import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from safetensors.torch import load_file


# Helpers

@torch.inference_mode()
def encode_latents(vae: AutoencoderKL, image_tensor: torch.Tensor) -> torch.Tensor:
    lat = vae.encode(image_tensor).latent_dist.sample()
    return lat * vae.config.scaling_factor


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    latents = latents / vae.config.scaling_factor
    img = vae.decode(latents).sample
    return (img / 2.0 + 0.5).clamp(0, 1)


def load_rgb(path: str, size: int) -> torch.Tensor:
    """Load RGB image → (1, 3, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load_mask(path: str, size: int) -> torch.Tensor:
    """Load mask → (1, 1, H, W) float tensor in {0.0, 1.0}. White = inpaint."""
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = (np.array(m) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def load_ti_embedding(ti_path: str, tokenizer: CLIPTokenizer,
                      text_encoder: CLIPTextModel, device, dtype):
    """Load a diffusers-format TI embedding (folder or .bin/.safetensors file)."""
    if os.path.isdir(ti_path):
        sf_path  = os.path.join(ti_path, "learned_embeds.safetensors")
        bin_path = os.path.join(ti_path, "learned_embeds.bin")
        if os.path.exists(sf_path):
            ti_path = sf_path
        elif os.path.exists(bin_path):
            ti_path = bin_path
        else:
            raise FileNotFoundError(f"No learned_embeds file found in {ti_path}")

    embeds_dict = (load_file(ti_path) if ti_path.endswith(".safetensors")
                   else torch.load(ti_path, map_location="cpu"))

    loaded_tokens = []
    for token_str, embedding in embeds_dict.items():
        embedding = embedding.squeeze()
        expected_dim = text_encoder.get_input_embeddings().weight.shape[1]
        if embedding.shape[0] != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch: got {embedding.shape[0]}, "
                f"expected {expected_dim}. "
                f"SD2 expects 1024 — did you train with SD1.x instead?")
        num_added = tokenizer.add_tokens(token_str)
        if num_added == 0:
            print(f"  Warning: token '{token_str}' already in vocabulary.")
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[token_id] = \
                embedding.to(device=device, dtype=dtype)
        loaded_tokens.append(token_str)
        print(f"  Loaded TI token '{token_str}' (dim={embedding.shape[0]})")
    return loaded_tokens


def encode_prompt(prompt: str, tokenizer: CLIPTokenizer,
                  text_encoder: CLIPTextModel, device) -> torch.Tensor:
    tok = tokenizer(
        [prompt], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.inference_mode():
        return text_encoder(tok.input_ids.to(device))[0]   # (1, 77, dim)


def get_timesteps(scheduler, steps: int, strength: float, device):
    """
    Return the sliced timestep schedule corresponding to the given strength.
    strength=1.0 → all steps (pure noise init).
    strength=0.85 → skip first 15% of steps (partial noise init).
    """
    init_timestep = min(int(steps * strength), steps)
    t_start       = max(steps - init_timestep, 0)
    timesteps     = scheduler.timesteps[t_start:]
    return timesteps, t_start


# Denoising loop

def run_denoising(unet, vae, scheduler, I, M,
                  cond_emb, uncond_emb,
                  steps, guidance_scale, ddim_eta, strength,
                  device, dtype,
                  num_samples: int = 4,
                  seed: int = 42) -> list:
    """
    DDIM denoising with:
      - strength-based partial noise initialisation (matches pipeline behaviour)
      - per-step latent blending for smooth boundary transitions
      - classifier-free guidance (CFG)
      - batched num_samples with per-sample seed offsets for diverse outputs
      - final hard latent composite for pixel-perfect background preservation
    """
    with torch.inference_mode():
        # Encode original image
        I_n      = I * 2.0 - 1.0                                    # [0,1] → [-1,1]
        I_lat    = encode_latents(vae, I_n)                          # (1, 4, h, w)
        lat_h, lat_w = I_lat.shape[2], I_lat.shape[3]

        # Downsample mask to latent resolution
        mask_lat = F.interpolate(M, size=(lat_h, lat_w), mode="nearest")  # (1,1,h,w)

        # Tile to batch size
        I_lat_b    = I_lat.repeat(num_samples, 1, 1, 1)             # (N, 4, h, w)
        mask_lat_b = mask_lat.repeat(num_samples, 1, 1, 1)          # (N, 1, h, w)

        # Strength: get sliced timesteps + init noise level
        timesteps, t_start = get_timesteps(scheduler, steps, strength, device)

        # Each sample gets a different seed for diverse outputs
        latents_list = []
        for i in range(num_samples):
            gen   = torch.Generator(device=device).manual_seed(seed + i)
            noise = torch.randn(1, 4, lat_h, lat_w,
                                generator=gen, device=device, dtype=dtype)
            # Init at the noise level of the first active timestep
            t_init  = timesteps[0:1]                                 # shape (1,)
            init_lt = scheduler.add_noise(I_lat, noise, t_init)
            latents_list.append(init_lt)
        latents = torch.cat(latents_list, dim=0)                     # (N, 4, h, w)

        # Tile prompt embeddings
        cond_b   = cond_emb.repeat(num_samples, 1, 1)               # (N, 77, dim)
        uncond_b = uncond_emb.repeat(num_samples, 1, 1)             # (N, 77, dim)
        use_cfg  = (guidance_scale != 1.0)

        # Denoising loop
        for t in timesteps:
            # Latent blending: re-noise original at this timestep,
            # paste into unmasked region so boundary transitions smoothly.
            # Use per-step fresh noise — must NOT reuse init noise here.
            blend_noise  = torch.randn_like(I_lat_b)
            t_vec        = t.unsqueeze(0)                            # scalar → (1,)
            I_lat_noisy  = scheduler.add_noise(I_lat_b, blend_noise, t_vec)
            latents      = I_lat_noisy * (1.0 - mask_lat_b) + latents * mask_lat_b

            # 9-channel UNet input: [noisy_latent | image_latent | mask]
            model_in = torch.cat([latents, I_lat_b, mask_lat_b], dim=1)  # (N, 9, h, w)

            if use_cfg:
                # Batch both passes: [uncond, cond] → chunk back
                model_in_2x = torch.cat([model_in, model_in], dim=0)     # (2N, 9, h, w)
                emb_2x      = torch.cat([uncond_b, cond_b],   dim=0)     # (2N, 77, dim)
                eps_both    = unet(model_in_2x, t,
                                   encoder_hidden_states=emb_2x).sample
                eps_uncond, eps_cond = eps_both.chunk(2)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = unet(model_in, t, encoder_hidden_states=cond_b).sample

            latents = scheduler.step(eps, t, latents, eta=ddim_eta).prev_sample

        # Final hard composite in latent space
        # Guarantees background outside mask is pixel-perfect original.
        latents = I_lat_b * (1.0 - mask_lat_b) + latents * mask_lat_b

        # Decode each sample to PIL Image
        out_imgs = []
        for i in range(num_samples):
            out = decode_latents(vae, latents[i].unsqueeze(0))       # (1, 3, H, W)
            arr = (out[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            out_imgs.append(Image.fromarray(arr))

    return out_imgs


# Argument parser

def parse_args():
    ap = argparse.ArgumentParser(
        description="SD2 Inpainting inference — fine-tuned UNet + latent blending + CFG + TI")
    ap.add_argument("--data_root",       type=str, required=True)
    ap.add_argument("--folder",          type=str, required=True)
    ap.add_argument("--checkpoint",      type=str, required=True,
                    help="Path to fine-tuned UNet .safetensors")
    ap.add_argument("--out_dir",         type=str, required=True)
    ap.add_argument("--sample_id",       type=str, default=None,
                    help="Single sample ID (e.g. V11_00148). "
                         "Omit to process all _I images in --folder.")
    ap.add_argument("--res",             type=int, default=512,
                    help="Inference resolution (512 or 768). "
                         "Outputs are always resized back to original dimensions.")
    ap.add_argument("--steps",           type=int,   default=50)
    ap.add_argument("--guidance_scale",  type=float, default=7.5)
    ap.add_argument("--ddim_eta",        type=float, default=0.0)
    ap.add_argument("--strength",        type=float, default=0.85,
                    help="Denoising strength (0.0–1.0). "
                         "0.85 matches the inpaint_sd2.py pipeline default.")
    ap.add_argument("--num_samples",     type=int,   default=4)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--mixed",           type=str,   default="fp16",
                    choices=["fp16", "bf16"])
    ap.add_argument("--ti_embedding",    type=str,   default=None,
                    help="Path to TI folder or .bin/.safetensors. "
                         "Token is automatically placed in negative prompt.")
    ap.add_argument("--positive_prompt", type=str,   default=None)
    ap.add_argument("--negative_prompt", type=str,   default=None)
    return ap.parse_args()


# Per-sample processing

def process_sample(img_id, folder_path, args,
                   unet, vae, scheduler,
                   cond_emb, uncond_emb,
                   device, dtype) -> bool:

    # Find input image (jpg or png)
    image_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(folder_path, f"{img_id}_I{ext}")
        if os.path.exists(candidate):
            image_path = candidate
            break
    if image_path is None:
        print(f"⚠️  Image not found for {img_id}, skipping.")
        return False

    mask_path = os.path.join(folder_path, f"{img_id}_C.png")
    if not os.path.exists(mask_path):
        print(f"⚠️  Mask not found for {img_id} (expected {img_id}_C.png), skipping.")
        return False

    print(f"Processing {img_id}...")

    # Read original size BEFORE any resize — used for final upscale
    orig_size = Image.open(image_path).size   # (W, H)

    I = load_rgb(image_path, args.res).to(device, dtype)
    M = load_mask(mask_path,  args.res).to(device, dtype)

    results = run_denoising(
        unet, vae, scheduler, I, M,
        cond_emb, uncond_emb,
        steps          = args.steps,
        guidance_scale = args.guidance_scale,
        ddim_eta       = args.ddim_eta,
        strength       = args.strength,
        device         = device,
        dtype          = dtype,
        num_samples    = args.num_samples,
        seed           = args.seed,
    )

    pair_dir = os.path.join(args.out_dir, img_id)
    os.makedirs(pair_dir, exist_ok=True)

    for i, result in enumerate(results):
        result_resized = result.resize(orig_size, Image.LANCZOS)
        out_path = os.path.join(pair_dir, f"{img_id}_result_{i}.png")
        result_resized.save(out_path)

    print(f"✅  {img_id} → {args.num_samples} results saved to '{pair_dir}'\n")
    return True


# Main

def main():
    args   = parse_args()
    device = "cuda"
    dtype  = torch.float16 if args.mixed == "fp16" else torch.bfloat16
    base   = "sd2-community/stable-diffusion-2-inpainting"

    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Text encoder
    tokenizer    = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder").to(device, dtype)
    text_encoder.eval()

    # Load TI embedding
    ti_tokens = []
    if args.ti_embedding is not None:
        print(f"Loading TI embedding from: {args.ti_embedding}")
        ti_tokens = load_ti_embedding(
            args.ti_embedding, tokenizer, text_encoder, device, dtype)

    # Build prompts
    positive_prompt = (args.positive_prompt
                       if args.positive_prompt is not None
                       else "a clean surface, even lighting, high quality photo")

    default_keywords = ("shadow, dark area, cast shadow, object, "
                        "harsh lighting, low quality")
    if args.negative_prompt is not None:
        negative_prompt = args.negative_prompt
    elif ti_tokens:
        negative_prompt = f"{' '.join(ti_tokens)}, {default_keywords}"
    else:
        negative_prompt = default_keywords

    print(f"Positive prompt : '{positive_prompt}'")
    print(f"Negative prompt : '{negative_prompt}'")
    print(f"Guidance scale  : {args.guidance_scale}")
    print(f"Strength        : {args.strength}\n")

    cond_emb   = encode_prompt(positive_prompt, tokenizer, text_encoder, device)
    uncond_emb = encode_prompt(negative_prompt, tokenizer, text_encoder, device)

    # Free text encoder from GPU
    text_encoder.to("cpu")
    del text_encoder
    torch.cuda.empty_cache()

    # VAE
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(device, dtype)
    vae.eval()

    # UNet — load pretrained then overwrite with fine-tuned weights
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to(device, dtype)
    unet.eval()

    # Verify this is a 9-channel inpainting UNet before loading weights
    assert unet.config.in_channels == 9, (
        f"Expected 9-channel inpainting UNet, got {unet.config.in_channels}. "
        f"Make sure --checkpoint was trained on the SD2 inpainting backbone.")

    print(f"Loading fine-tuned UNet weights from: {args.checkpoint}")
    unet.load_state_dict(load_file(args.checkpoint), strict=True)
    print("UNet weights loaded.\n")

    # Scheduler
    scheduler = DDIMScheduler.from_pretrained(base, subfolder="scheduler")
    scheduler.set_timesteps(args.steps, device=device)

    folder_path = os.path.join(args.data_root, args.folder)

    # Build list of sample IDs
    if args.sample_id is not None:
        sample_ids = [args.sample_id]
    else:
        valid_ext  = {".png", ".jpg", ".jpeg"}
        sample_ids = sorted([
            os.path.splitext(f)[0][:-2]           # strip "_I"
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in valid_ext
            and os.path.splitext(f)[0].endswith("_I")
        ])
        print(f"Found {len(sample_ids)} images in '{folder_path}'\n")

    # Process
    success, skipped = 0, []
    for img_id in sample_ids:
        ok = process_sample(
            img_id, folder_path, args,
            unet, vae, scheduler,
            cond_emb, uncond_emb,
            device, dtype)
        if ok:
            success += 1
        else:
            skipped.append(img_id)

    print(f"Done. {success} images processed.")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()