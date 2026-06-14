# Fine-Tuning: SD2 Inpainting U-Net on Counterfactual Pairs

This folder contains the training and inference scripts for the **SD2 Fine-Tuned** variant of EADI.
The U-Net backbone of `stable-diffusion-2-inpainting` is fine-tuned on counterfactual image pairs
so the model learns to reconstruct a clean background from image content alone, without relying
on text conditioning. At inference, prompt engineering and an optional Textual Inversion (TI)
token are layered on top of the fine-tuned weights.

---

## Method Overview

### Training Objective

The U-Net is trained with a **noise-prediction MSE loss** on counterfactual image pairs.
For each training sample the model receives:

- `B_lat`: VAE latent of the clean background image (the prediction target)
- `I_lat`: VAE latent of the factual image (with the object present)
- `mask_lat`: Binary object mask downsampled to latent resolution

The 9-channel U-Net input at each denoising step is:

```
model_input = [ noisy_B_lat  |  I_lat  |  mask_lat ]
              (4 channels)    (4 ch.)   (1 ch.)   → 9 ch. total
```

**Text conditioning is intentionally empty (`""`) during training.**
This forces the model to reconstruct the background from visual context alone.
Prompt engineering is added only at inference, keeping the training signal clean.

### Frozen Components

Only the U-Net weights are updated. The VAE and CLIP text encoder are frozen
throughout training to preserve the latent encoding space and text alignment.

| Component | Status |
|-----------|--------|
| U-Net | ✅ Trained |
| VAE | ❌ Frozen |
| CLIP Text Encoder | ❌ Frozen |

### Inference Pipeline

At inference the script runs a standard DDIM loop with three additions:

1. **Strength-based partial noise init** (`--strength 0.85`) : skips the first 15 % of
   timesteps and initialises from a partially noised version of the original image rather
   than pure Gaussian noise, preserving more original structure.
2. **Per-step latent blending** : at every denoising step the unmasked latent region is
   re-composited from the original image, ensuring smooth boundary transitions without seams.
3. **Hard latent composite** after the final step : pixel-perfect background preservation
   outside the mask is guaranteed in latent space before decoding.

If a TI embedding is provided via `--ti_embedding`, the learned token (e.g. `<shadowobject>`)
is automatically prepended to the negative prompt to suppress shadow residuals.

---

## Environment Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv sd2_ft
source sd2_ft/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install PyTorch 2.8 + torchvision for CUDA 12.8

```bash
pip install torch==2.8.0 torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install xformers (pinned to torch 2.8 + cu128)

```bash
pip install xformers==0.0.30 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install HuggingFace stack and core dependencies

```bash
pip install diffusers transformers accelerate safetensors \
            pillow matplotlib numpy
```

### 6. Install bitsandbytes (CUDA 12.8)

```bash
pip install bitsandbytes
```

### 7. Log in to HuggingFace

The base model (`sd2-community/stable-diffusion-2-inpainting`) is downloaded automatically
on first run. A HuggingFace account with access to the model is required.

```bash
huggingface-cli login
```

### 8. Configure Accelerate

```bash
accelerate config
```

Select your hardware (single GPU / multi-GPU / mixed precision) when prompted.
For a single-GPU fp16 setup, accept defaults and set mixed precision to `fp16`.

---

## Dataset Format

`train.py` uses `CounterfactualDataset`, which expects the following file triplet
per sample ID `{sid}` inside the train and validation folders:

| File | Description |
|------|-------------|
| `{sid}_I.jpg` | Factual image (object present) |
| `{sid}_B.jpg` | Counterfactual image (clean background ground truth) |
| `{sid}_O.png` | Binary object segmentation mask (white = inpaint region) |

The inference script expects a different naming convention for test data:

| File | Description |
|------|-------------|
| `{sid}_I.jpg / .png` | Input image (object present) |
| `{sid}_C.png` | Combined mask (object + shadow region, white = inpaint) |

Expected directory layout:

```
data/
└── Quad_DS_TVT/
    ├── Train/
    │   ├── V01_00001_I.jpg
    │   ├── V01_00001_B.jpg
    │   ├── V01_00001_O.png
    │   └── ...
    ├── Validation/
    │   └── ...
    └── Test/          (used at inference only)
        └── ...
```

---

## Training

```bash
python train.py \
  --data_root ./data/Quad_DS_TVT \
  --train_folder Train \
  --val_folder Validation \
  --out_dir ./checkpoints_sd2_ft \
  --res 768 \
  --batch 2 \
  --grad_accum 64 \
  --lr 1e-5 \
  --max_steps 120000 \
  --mixed fp16 \
  --save_final \
  --val_steps 5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000 \
  --save_steps 20000,40000,60000,80000,100000,120000
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `N/A` | Root directory containing train/val subfolders |
| `--train_folder` | `Train` | Subfolder name for training data |
| `--val_folder` | `Validation` | Subfolder name for validation data. Set to `""` to disable. |
| `--out_dir` | `N/A` | Output directory for checkpoints and logs |
| `--pretrained` | `sd2-community/stable-diffusion-2-inpainting` | HuggingFace model ID |
| `--res` | `768` | Training resolution (512 or 768) |
| `--batch` | `2` | Per-device batch size |
| `--grad_accum` | `64` | Gradient accumulation steps. Effective batch size = `batch × grad_accum = 128` |
| `--lr` | `1e-5` | AdamW learning rate |
| `--wd` | `1e-2` | AdamW weight decay |
| `--max_steps` | `120000` | Total optimiser steps |
| `--warmup` | `500` | Linear warmup steps before cosine decay |
| `--mixed` | `fp16` | Mixed precision: `fp16`, `bf16`, or `no` |
| `--save_steps` | `20000,...` | Comma-separated steps at which to save a `.safetensors` checkpoint |
| `--val_steps` | `5000,...` | Comma-separated steps at which to compute validation MSE loss |
| `--save_final` | `False` | If set, saves `unet_final.safetensors` after the last step |
| `--resume_from` | `None` | Path to a `.safetensors` checkpoint to resume training from |
| `--seed` | `42` | Random seed for reproducibility |

### Outputs

After training the `--out_dir` will contain:

```
checkpoints_sd2_ft/
├── unet_step20000.safetensors
├── unet_step40000.safetensors
├── ...
├── unet_step120000.safetensors
├── unet_final.safetensors     (if --save_final is set)
├── loss_log.json              (train + validation loss per step)
└── loss_curves.png            (training and validation loss plots)
```

`loss_log.json` records every training step loss and all validation checkpoints.
`loss_curves.png` shows the raw training loss, a 50-step smoothed curve, and the
validation loss with the best-performing step highlighted.

---

## Inference

```bash
python inference.py \
    --data_root . \
    --folder COMPLETE_TEST \
    --checkpoint ./checkpoints_sd2/unet_final.safetensors \
    --out_dir ./results/sd2_ft_ti \
    --res 768 \
    --guidance_scale 7.5 \
    --strength 0.85 \
    --ti_embedding ./ti_shadow_sd2
```

### Key Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `N/A` | Root directory containing the test folder |
| `--folder` | `N/A` | Subfolder with test images |
| `--checkpoint` | `N/A` | Path to fine-tuned U-Net `.safetensors` |
| `--out_dir` | `N/A` | Output directory for results |
| `--sample_id` | `None` | Single sample ID (e.g. `V11_00148`). Omit to process all images. |
| `--res` | `512` | Inference resolution (512 or 768). Outputs are always upscaled back to original size. |
| `--steps` | `50` | Number of DDIM denoising steps |
| `--guidance_scale` | `7.5` | Classifier-free guidance (CFG) scale |
| `--strength` | `0.85` | Denoising strength. `0.85` skips the first 15 % of timesteps. |
| `--num_samples` | `4` | Number of diverse outputs per input (different seed offsets) |
| `--seed` | `42` | Base random seed; each sample uses `seed + i` |
| `--mixed` | `fp16` | Mixed precision: `fp16` or `bf16` |
| `--ti_embedding` | `None` | Path to TI folder or `.bin`/`.safetensors`. Token is auto-inserted into the negative prompt. |
| `--positive_prompt` | `"a clean surface, even lighting, high quality photo"` | Override the positive prompt |
| `--negative_prompt` | `None` | Override the negative prompt (default includes TI token if loaded) |
| `--ddim_eta` | `0.0` | DDIM stochasticity (0 = deterministic) |

### Default Prompts

| Prompt | Value |
|--------|-------|
| Positive | `"a clean surface, even lighting, high quality photo"` |
| Negative (no TI) | `"shadow, dark area, cast shadow, object, harsh lighting, low quality"` |
| Negative (with TI) | `"<shadowobject>, shadow, dark area, cast shadow, object, harsh lighting, low quality"` |

### Inference Output Structure

Each processed sample produces a subdirectory with `num_samples` PNG results,
all resized back to the original image dimensions:

```
results/sd2_ft_ti/
├── V11_00148/
│   ├── V11_00148_result_0.png
│   ├── V11_00148_result_1.png
│   ├── V11_00148_result_2.png
│   └── V11_00148_result_3.png
└── ...
```

---

## Memory and Hardware Notes

- Training at `--res 768` with `--batch 2 --grad_accum 64` was performed on a single NVIDIA
  A100 40 GB GPU. Reduce `--res` to `512` or `--batch` to `1` for smaller GPUs.
- `xformers` memory-efficient attention is enabled automatically if the package is installed,
  falling back gracefully if not available.
- Gradient checkpointing is always enabled during training to reduce activation memory.
- The VAE and text encoder are offloaded to CPU after the prompt embedding step at inference
  to free VRAM for the U-Net denoising loop.

---

## Notes

- The base model must be exactly `sd2-community/stable-diffusion-2-inpainting`
  (a 9-channel inpainting U-Net). Passing a standard SD2 checkpoint will raise an assertion error.
- TI tokens must have been trained against the same SD2 backbone.
  A dimension mismatch error will be raised if an SD1.x token is loaded.
- Inference outputs are always saved at the original image resolution regardless of `--res`,
  using Lanczos upsampling.
