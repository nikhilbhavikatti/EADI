import os
import glob
import math
import random
import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from safetensors.torch import save_file, load_file
import numpy as np


# Dataset
class CounterfactualDataset(Dataset):
    """
    Expected files per sample ID:
        {sid}_I.jpg  - factual image (with object)
        {sid}_B.jpg  - counterfactual image (without object)
        {sid}_O.png  - binary object segmentation mask M_o
    """
    def __init__(self, folder: str, res: int):
        self.folder = folder
        self.res = res
        self.I_paths = sorted(glob.glob(os.path.join(folder, "*_I.jpg")))
        if len(self.I_paths) == 0:
            raise FileNotFoundError(f"No *_I.jpg found in: {folder}")
        self.ids = [os.path.basename(p)[: -len("_I.jpg")] for p in self.I_paths]

        self.img_tf = transforms.Compose([
            transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((res, res), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_rgb(self, path: str) -> torch.Tensor:
        return self.img_tf(Image.open(path).convert("RGB"))

    def _load_mask(self, path: str) -> torch.Tensor:
        m = self.mask_tf(Image.open(path).convert("L"))
        return (m > 0.5).float()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        I_path = os.path.join(self.folder, f"{sid}_I.jpg")
        B_path = os.path.join(self.folder, f"{sid}_B.jpg")
        O_path = os.path.join(self.folder, f"{sid}_O.png")
        for p in [I_path, B_path, O_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        I = self._load_rgb(I_path) * 2.0 - 1.0
        B = self._load_rgb(B_path) * 2.0 - 1.0
        M = self._load_mask(O_path)
        return {"I": I, "B": B, "M": M, "id": sid}

# VAE helper
@torch.no_grad()
def encode_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(images).latent_dist.sample()
    return latents * vae.config.scaling_factor


# Validation loss computation
@torch.no_grad()
def run_validation(unet, vae, noise_sched, val_dl, empty_emb, accelerator):
    """
    Compute mean noise-prediction MSE loss over the full validation set.
    UNet is temporarily set to eval() then restored to train().
    """
    unet.eval()
    total_loss  = 0.0
    total_steps = 0

    for batch in val_dl:
        I = batch["I"]
        B = batch["B"]
        M = batch["M"]

        B_lat = encode_latents(vae, B)
        I_lat = encode_latents(vae, I)

        noise   = torch.randn_like(B_lat)
        t       = torch.randint(0, noise_sched.config.num_train_timesteps,
                                (B_lat.shape[0],), device=B_lat.device).long()
        noisy_B = noise_sched.add_noise(B_lat, noise, t)

        lat_h, lat_w = B_lat.shape[2], B_lat.shape[3]
        mask_lat = F.interpolate(M, size=(lat_h, lat_w), mode="nearest")
        model_in = torch.cat([noisy_B, I_lat, mask_lat], dim=1)

        pred = unet(
            model_in, t,
            encoder_hidden_states=empty_emb.repeat(B_lat.shape[0], 1, 1),
        ).sample
        loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

        total_loss  += loss.item()
        total_steps += 1

    unet.train()
    return total_loss / max(total_steps, 1)


# Plot helper
def save_loss_plot(train_log, val_log, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Training loss ---
    tr_steps  = [x["step"] for x in train_log]
    tr_losses = [x["loss"] for x in train_log]
    axes[0].plot(tr_steps, tr_losses, color="#2196F3", linewidth=0.8,
                 alpha=0.4, label="Train loss (raw)")
    if len(tr_losses) >= 50:
        window = 50
        smooth = [
            sum(tr_losses[max(0, i - window): i + 1]) /
            len(tr_losses[max(0, i - window): i + 1])
            for i in range(len(tr_losses))
        ]
        axes[0].plot(tr_steps, smooth, color="#0D47A1", linewidth=1.8,
                     label=f"Train loss (smooth {window})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Validation loss
    if val_log:
        val_steps  = [x["step"] for x in val_log]
        val_losses = [x["loss"] for x in val_log]
        axes[1].plot(val_steps, val_losses, color="#F44336", linewidth=2.0,
                     marker="o", markersize=5, label="Val loss")
        best_idx = val_losses.index(min(val_losses))
        axes[1].scatter(
            [val_steps[best_idx]], [val_losses[best_idx]],
            color="#B71C1C", zorder=5, s=80,
            label=f"Best: {val_losses[best_idx]:.4f} @ step {val_steps[best_idx]}"
        )
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("MSE Loss")
        axes[1].set_title("Validation Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No validation data",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


# Training
def train(args):
    print("Starting ObjectDrop removal training (SD2-inpainting)...")

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    save_steps = {int(x.strip()) for x in args.save_steps.split(",") if x.strip()}
    val_steps  = {int(x.strip()) for x in args.val_steps.split(",")  if x.strip()}

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(args.pretrained, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(args.pretrained, subfolder="unet")
    noise_sched  = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        unet.load_state_dict(load_file(args.resume_from), strict=True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("xformers not enabled:", e)

    empty_tokens = tokenizer(
        [""], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt",
    )

    # Train DataLoader
    train_folder = os.path.join(args.data_root, args.train_folder)
    train_ds = CounterfactualDataset(train_folder, args.res)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Validation DataLoader
    val_dl = None
    if args.val_folder:
        val_folder = os.path.join(args.data_root, args.val_folder)
        val_ds = CounterfactualDataset(val_folder, args.res)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        print(f"Validation set: {len(val_ds)} samples from {val_folder}")

    accelerator = Accelerator(
        mixed_precision=args.mixed,
        gradient_accumulation_steps=args.grad_accum,
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    epochs = math.ceil(args.max_steps / steps_per_epoch)

    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=args.max_steps,
    )

    unet, optimizer, train_dl, lr_sched, text_encoder, vae = accelerator.prepare(
        unet, optimizer, train_dl, lr_sched, text_encoder, vae
    )
    if val_dl is not None:
        val_dl = accelerator.prepare(val_dl)

    empty_ids = empty_tokens.input_ids.to(accelerator.device)
    with torch.no_grad():
        empty_emb = text_encoder(empty_ids)[0]  # (1, 77, 1024) for SD2

    # Loss tracking
    train_log = []
    val_log   = []

    global_step = 0
    for ep in range(epochs):
        for batch in train_dl:
            with accelerator.accumulate(unet):
                I = batch["I"]
                B = batch["B"]
                M = batch["M"]

                with torch.no_grad():
                    B_lat = encode_latents(vae, B)
                    I_lat = encode_latents(vae, I)

                noise   = torch.randn_like(B_lat)
                t       = torch.randint(0, noise_sched.config.num_train_timesteps,
                                        (B_lat.shape[0],), device=B_lat.device).long()
                noisy_B = noise_sched.add_noise(B_lat, noise, t)

                lat_h, lat_w = B_lat.shape[2], B_lat.shape[3]
                mask_lat = F.interpolate(M, size=(lat_h, lat_w), mode="nearest")
                model_in = torch.cat([noisy_B, I_lat, mask_lat], dim=1)

                pred = unet(
                    model_in, t,
                    encoder_hidden_states=empty_emb.repeat(B_lat.shape[0], 1, 1),
                ).sample
                loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                train_log.append({"step": global_step, "loss": loss.item()})
                if global_step % 100 == 0:
                    print(f"step={global_step}  train_loss={loss.item():.4f}  "
                          f"lr={lr_sched.get_last_lr()[0]:.2e}")

            # Validation
            if accelerator.is_main_process and val_dl is not None \
                    and global_step in val_steps:
                val_loss = run_validation(
                    accelerator.unwrap_model(unet), vae, noise_sched,
                    val_dl, empty_emb, accelerator
                )
                val_log.append({"step": global_step, "loss": val_loss})
                print(f"  >> VAL  step={global_step}  val_loss={val_loss:.4f}")

            # ---- Checkpoint ----
            if accelerator.is_main_process and global_step in save_steps:
                unet_unwrapped = accelerator.unwrap_model(unet)
                ckpt = os.path.join(args.out_dir, f"unet_step{global_step}.safetensors")
                save_file(unet_unwrapped.state_dict(), ckpt)
                print(f"Saved checkpoint: {ckpt}")

            global_step += 1
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    # Final validation pass
    if accelerator.is_main_process and val_dl is not None:
        val_loss = run_validation(
            accelerator.unwrap_model(unet), vae, noise_sched,
            val_dl, empty_emb, accelerator
        )
        val_log.append({"step": global_step, "loss": val_loss})
        print(f"  >> FINAL VAL  step={global_step}  val_loss={val_loss:.4f}")

    # Save final model
    if accelerator.is_main_process and args.save_final:
        unet_unwrapped = accelerator.unwrap_model(unet)
        final = os.path.join(args.out_dir, "unet_final.safetensors")
        save_file(unet_unwrapped.state_dict(), final)
        print("Saved final model:", final)

    # Save loss log + plot
    if accelerator.is_main_process:
        log_path = os.path.join(args.out_dir, "loss_log.json")
        with open(log_path, "w") as f:
            json.dump({"train": train_log, "val": val_log}, f, indent=2)
        print("Saved loss log:", log_path)

        plot_path = save_loss_plot(train_log, val_log, args.out_dir)
        print("Saved loss plot:", plot_path)


# Argument parser
def parse_args():
    ap = argparse.ArgumentParser(description="ObjectDrop - SD2-inpainting Training")
    ap.add_argument("--data_root",    type=str, required=True)
    ap.add_argument("--train_folder", type=str, default="Train")
    ap.add_argument("--val_folder",   type=str, default="Validation",
                    help="Validation folder inside data_root. Set to '' to disable.")
    ap.add_argument("--out_dir",      type=str, required=True)
    ap.add_argument("--pretrained",   type=str,
                    default="sd2-community/stable-diffusion-2-inpainting")
    ap.add_argument("--res",          type=int,   default=768,
                    help="512 or 768 — SD2 supports both natively")
    ap.add_argument("--batch",        type=int,   default=2)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--lr",           type=float, default=1e-5)
    ap.add_argument("--wd",           type=float, default=1e-2)
    ap.add_argument("--max_steps",    type=int,   default=120_000)
    ap.add_argument("--warmup",       type=int,   default=500)
    ap.add_argument("--grad_accum",   type=int,   default=64,
                    help="Set so batch * grad_accum = 128 effective batch size")
    ap.add_argument("--mixed",        type=str,   default="fp16",
                    choices=["fp16", "bf16", "no"])
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--save_steps",   type=str,
                    default="20000,40000,60000,80000,100000,120000")
    ap.add_argument("--val_steps",    type=str,
                    default="5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000",
                    help="Steps at which to compute validation loss")
    ap.add_argument("--save_final",   action="store_true")
    ap.add_argument("--resume_from",  type=str,   default=None,
                    help="Path to unet .safetensors to resume training from")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)