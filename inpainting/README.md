# Textual Inversion: Shadow Token Training and Inpainting

This folder contains everything needed to train a **Textual Inversion (TI)** token that
represents the visual concept of a cast shadow, and to run inpainting inference across
four EADI variants: **SD1.5**, **SD1.5 + TI**, **SD2**, and **SD2 + TI**.

The trained token `<shadowobject>` is inserted into the **negative prompt** at inference
to suppress shadow residuals that prompt engineering alone cannot remove.

---

## Overview of Variants

| Variant | Base Model | TI Token | Embedding Format |
|---------|-----------|----------|-----------------|
| SD1.5 | `stable-diffusion-v1-5/stable-diffusion-inpainting` | ✗ | N/A |
| SD1.5 + TI | `stable-diffusion-v1-5/stable-diffusion-inpainting` | `<shadowobject>` | `shadow_sd15_diffusers.pt` |
| SD2 | `sd2-community/stable-diffusion-2-inpainting` | ✗ | N/A |
| SD2 + TI | `sd2-community/stable-diffusion-2-inpainting` | `<shadowobject>` | `learned_embeds.safetensors` |

> **Important:** SD1.5 and SD2 use different CLIP text encoders with different embedding
> dimensions (768 vs 1024). Tokens trained on one backbone **cannot** be used with the other.

---

## Part 1: Textual Inversion Training

### 1A. SD1.5 TI Training (rinongal LDM Repository)

SD1.5 TI training uses the original [rinongal/textual_inversion](https://github.com/rinongal/textual_inversion)
repository, which operates on the raw LDM checkpoint format.

#### Environment Setup

```bash
# 1. Clone the repo
mkdir -p /scratch/users/$USER/projects
cd /scratch/users/$USER/projects
git clone https://github.com/rinongal/textual_inversion.git
cd textual_inversion

# 2. Install micromamba
mkdir -p ~/.local/bin
cd ~/.local/bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv bin/micromamba micromamba
rmdir bin
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
micromamba --version

# 3. Create the environment
cd /scratch/users/$USER/projects/textual_inversion
micromamba create -y -n stab -f environment.yaml
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate stab

# 4. Install diffusers stack
pip install diffusers==0.27.2 accelerate==0.28.0 transformers==4.38.2 \
            ftfy tensorboard Pillow safetensors huggingface_hub==0.22.2
```

#### Download Pretrained Checkpoints

```bash
# SD1.5 checkpoint
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='runwayml/stable-diffusion-v1-5',
    filename='v1-5-pruned.ckpt',
    local_dir='models/ldm/stable-diffusion-v1/'
)
print('Downloaded to:', path)
"

# CLIP ViT-L/14 (required by v1-finetune.yaml)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openai/clip-vit-large-patch14',
    local_dir='models/clip-vit-large-patch14'
)
"
```

#### Config Edit (required)

In `configs/stable-diffusion/v1-finetune.yaml`, point the CLIP encoder to your
local download:

```yaml
cond_stage_config:
  target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
  params:
    version: /path/to/textual_inversion/models/clip-vit-large-patch14
```

#### Prepare Training Images

Shadow patch crops must be resized to 512×512 before training:

```bash
python -c "
from PIL import Image
import pathlib, os

src = '/path/to/shadow_dataset_cropped/'
dst = '/path/to/shadow_dataset_cropped_resized/'
os.makedirs(dst, exist_ok=True)

files = list(pathlib.Path(src).glob('*.jpg')) + list(pathlib.Path(src).glob('*.jpeg'))
print(f'Found {len(files)} images')
for f in files:
    img = Image.open(f).convert('RGB').resize((512, 512), Image.LANCZOS)
    img.save(os.path.join(dst, f.name), quality=95)
print(f'Done. Saved to {dst}')
"
```

#### Run Training

```bash
python main.py \
  --base configs/stable-diffusion/v1-finetune.yaml \
  -t \
  --actual_resume models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt \
  -n shadow_run_sd15 \
  --gpus 0, \
  --data_root /path/to/shadow_dataset_cropped_resized \
  --init_word shadow \
  --no-test
```

The trained embedding is saved at:
```
logs/shadow_run_sd15<timestamp>/checkpoints/embeddings_gs-XXXX.pt
```

#### Verify the Token

```bash
python scripts/txt2img.py \
  --ddim_eta 0.0 \
  --n_samples 4 \
  --n_iter 1 \
  --scale 10.0 \
  --ddim_steps 50 \
  --embedding_path logs/shadow_run_sd15*/checkpoints/embeddings_gs-XXXX.pt \
  --ckpt_path models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt \
  --prompt "a photo of *"
```

---

### 1B. Convert SD1.5 Embedding to Diffusers Format

The LDM `.pt` file produced by the rinongal repo uses an older format that is **not**
compatible with the diffusers `load_textual_inversion()` API used by the inpainting scripts.
Convert it before running inference:

```bash
# Edit EMBEDDING_PATH inside the script to point to your embeddings_gs-XXXX.pt, then:
python convert_embedding_sd15.py
# Produces: shadow_sd15_diffusers.pt
```

The script extracts the raw token vector (expected shape `[1, 768]`) and saves it
as `shadow_sd15_diffusers.pt` in diffusers-compatible format.

> If the assertion `vector.shape[-1] == 768` fails, the embedding was trained on the
> wrong backbone. SD1.5 requires dim 768; SD2 requires dim 1024.

---

### 1C. SD2 TI Training (Diffusers Repository)

SD2 TI training uses the official HuggingFace `diffusers` `textual_inversion.py` script,
which works directly with the diffusers model format; no conversion step is needed.

#### Environment Setup

```bash
micromamba create -n ti_sd2 python=3.8 -y
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate ti_sd2

# PyTorch — CUDA 11.8, last build supporting Python 3.8
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

pip install diffusers==0.27.2 accelerate==0.28.0 transformers==4.38.2 \
            ftfy tensorboard Pillow safetensors huggingface_hub==0.22.2

# Download the official TI training script (pinned to diffusers v0.27.2)
wget https://raw.githubusercontent.com/huggingface/diffusers/v0.27.2/examples/textual_inversion/textual_inversion.py
pip install -r https://raw.githubusercontent.com/huggingface/diffusers/main/examples/textual_inversion/requirements.txt
```

#### Configure Accelerate

```bash
accelerate config
```

Recommended answers for a single A100 GPU:

| Question | Answer |
|----------|--------|
| Compute environment | This machine |
| Machine type | No distributed training |
| CPU only? | NO |
| Torch dynamo? | NO |
| DeepSpeed? | NO |
| Which GPU(s)? | all |
| Mixed precision | BF16 *(A100 supports BF16 natively)* |

#### Prepare Training Images

Same resize step as SD1.5 (512×512):

```bash
python -c "
from PIL import Image
import pathlib, os

src = '/path/to/shadow_dataset_cropped/'
dst = '/path/to/shadow_dataset_cropped_resized/'
os.makedirs(dst, exist_ok=True)

files = list(pathlib.Path(src).glob('*.jpg')) + list(pathlib.Path(src).glob('*.jpeg'))
print(f'Found {len(files)} images')
for f in files:
    img = Image.open(f).convert('RGB').resize((512, 512), Image.LANCZOS)
    img.save(os.path.join(dst, f.name), quality=95)
print(f'Done. Saved to {dst}')
"
```

#### Run Training

```bash
accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path="Manojb/stable-diffusion-2-base" \
  --train_data_dir="/path/to/shadow_dataset_cropped_resized" \
  --learnable_property="object" \
  --placeholder_token="<shadowobject>" \
  --initializer_token="shadow" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=25000 \
  --learning_rate=5e-4 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="bf16" \
  --output_dir="./logs/shadow_ti_sd2" \
  --save_steps=5000 \
  --validation_prompt="a photo with a <shadowobject> on white background" \
  --num_validation_images=4 \
  --validation_steps=5000 \
  --checkpointing_steps=5000
```

#### Key Training Arguments

| Argument | Value | Description |
|----------|-------|-------------|
| `--placeholder_token` | `<shadowobject>` | New token added to the CLIP vocabulary |
| `--initializer_token` | `shadow` | Existing token used to initialise the embedding |
| `--learnable_property` | `object` | Optimisation target: `object` or `style` |
| `--max_train_steps` | `25000` | Total optimiser steps |
| `--learning_rate` | `5e-4` | Base learning rate (scaled by `--scale_lr`) |
| `--save_steps` | `5000` | Checkpoint interval |
| `--validation_steps` | `5000` | Interval for generating validation images |

The final embedding is saved as:
```
logs/shadow_ti_sd2/learned_embeds.safetensors
```

No conversion step is needed — this file is directly loadable by `inpaint_sd2.py`.

---

## Part 2: Inpainting Inference

Both inpainting scripts use `StableDiffusionInpaintPipeline` from diffusers and share
identical inference logic. The only differences are the base model ID, the embedding
format, and the native resolution.

### Input File Convention

Both scripts scan a directory for files named `{id}_I.{ext}` and expect a corresponding
combined mask at `{id}_C.png`. The combined mask covers both the object and shadow region.

| File | Description |
|------|-------------|
| `{id}_I.jpg / .png` | Input image with object present |
| `{id}_C.png` | Combined mask (object + shadow, white = inpaint region) |

### Shared Inference Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PROMPT` | `"a clean surface, even lighting, high quality photo"` | Positive prompt |
| `NEGATIVE_PROMPT` | see per-variant below | Shadow suppression keywords |
| `STEPS` | `50` | Number of DDIM denoising steps |
| `GUIDANCE_SCALE` | `7.5` | Classifier-free guidance (CFG) scale |
| `STRENGTH` | `0.85` | Denoising strength; skips the first 15% of timesteps |
| `SEED` | `42` | Random seed |
| `NUM_IMAGES` | `4` | Outputs generated per input |

---

### 2A. SD1.5 (No TI)

Edit the config block at the top of `inpaint_sd15.py`:

```python
MODEL_ID       = "stable-diffusion-v1-5/stable-diffusion-inpainting"
EMBEDDING_PATH = ""   # not used — comment out load_textual_inversion below
IMAGES_DIR     = "COMPLETE_TEST/"
MASKS_DIR      = "COMPLETE_TEST/"
OUTPUT_DIR     = "inpaint_results_sd15/"
NEGATIVE_PROMPT = "shadow, dark area, cast shadow, object, harsh lighting, low quality"
```

Also comment out the `pipe.load_textual_inversion(...)` line. Then run:

```bash
python inpaint_sd15.py
```

---

### 2B. SD1.5 + TI

Ensure you have run `convert_embedding_sd15.py` first to produce `shadow_sd15_diffusers.pt`.

Edit the config block in `inpaint_sd15.py`:

```python
MODEL_ID       = "stable-diffusion-v1-5/stable-diffusion-inpainting"
EMBEDDING_PATH = "shadow_sd15_diffusers.pt"
IMAGES_DIR     = "COMPLETE_TEST/"
MASKS_DIR      = "COMPLETE_TEST/"
OUTPUT_DIR     = "inpaint_results_sd15_ti/"
NEGATIVE_PROMPT = "<shadowobject>, shadow, dark area, cast shadow, object, harsh lighting, low quality"
```

Run:

```bash
python inpaint_sd15.py
```

The script calls `pipe.load_textual_inversion(EMBEDDING_PATH, token="<shadowobject>")` to
inject the learned embedding before inference.

---

### 2C. SD2 (No TI)

Edit the config block in `inpaint_sd2.py`:

```python
MODEL_ID       = "sd2-community/stable-diffusion-2-inpainting"
EMBEDDING_PATH = ""   # not used; comment out load_textual_inversion below
IMAGES_DIR     = "COMPLETE_TEST/"
MASKS_DIR      = "COMPLETE_TEST/"
OUTPUT_DIR     = "inpaint_results_sd2/"
NEGATIVE_PROMPT = "shadow, dark area, cast shadow, object, harsh lighting, low quality"
```

Comment out the `pipe.load_textual_inversion(...)` line. Run:

```bash
python inpaint_sd2.py
```

---

### 2D. SD2 + TI

Edit the config block in `inpaint_sd2.py`:

```python
MODEL_ID       = "sd2-community/stable-diffusion-2-inpainting"
EMBEDDING_PATH = "logs/shadow_ti_sd2/learned_embeds.safetensors"
IMAGES_DIR     = "COMPLETE_TEST/"
MASKS_DIR      = "COMPLETE_TEST/"
OUTPUT_DIR     = "inpaint_results_sd2_ti/"
NEGATIVE_PROMPT = "<shadowobject>, shadow, dark area, cast shadow, object, harsh lighting, low quality"
```

Run:

```bash
python inpaint_sd2.py
```

---

## Output Structure

Each script creates one subdirectory per input sample containing `NUM_IMAGES` PNG results,
all upscaled back to the original image resolution via Lanczos resampling:

```
inpaint_results_sd2_ti/
├── V11_00148/
│   ├── V11_00148_result_0.png
│   ├── V11_00148_result_1.png
│   ├── V11_00148_result_2.png
│   └── V11_00148_result_3.png
└── ...
```

---

## Embedding Compatibility Reference

| Embedding file | Trained on | Dim | Compatible with |
|----------------|-----------|-----|-----------------|
| `embeddings_gs-XXXX.pt` (LDM format) | SD1.5 backbone | 768 | rinongal repository only (must be converted) |
| `shadow_sd15_diffusers.pt` | SD1.5 backbone | 768 | `inpaint_sd15.py` |
| `learned_embeds.safetensors` | SD2 backbone | 1024 | `inpaint_sd2.py` |

> Loading a 768-dim token into an SD2 pipeline (or vice versa) will raise a
> `ValueError: Embedding dim mismatch`. Always match embedding to backbone.

---

## Notes

- Both inpainting scripts resize inputs to **512×512** internally and restore the original
  resolution with Lanczos upsampling before saving. No manual resizing is needed.
- The `<shadowobject>` token is placed only in the **negative prompt**. It was trained to
  represent the visual appearance of shadows so the model suppresses shadow-like regions
  during generation.
- Samples with no matching `_C.png` mask are skipped automatically and reported at the
  end of each run.
