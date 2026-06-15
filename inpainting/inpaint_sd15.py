import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID       = "stable-diffusion-v1-5/stable-diffusion-inpainting"
EMBEDDING_PATH = "Path to shadow_sd15_diffusers.pt"
IMAGES_DIR     = "COMPLETE_TEST/"
MASKS_DIR      = "COMPLETE_TEST/"
OUTPUT_DIR     = "inpaint_results/"

PROMPT          = "a clean surface, even lighting, high quality photo"
NEGATIVE_PROMPT = "<shadowobject>, shadow, dark area, cast shadow, object, harsh lighting, low quality"
STEPS           = 50
GUIDANCE_SCALE  = 7.5
STRENGTH        = 0.85
SEED            = 42
NUM_IMAGES      = 4
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_textual_inversion(EMBEDDING_PATH, token="<shadowobject>")
print("Model loaded.\n")

valid_ext = {".png", ".jpg", ".jpeg"}
image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if os.path.splitext(f)[1].lower() in valid_ext
    and os.path.splitext(f)[0].endswith("_I")
])

print(f"Found {len(image_files)} images in '{IMAGES_DIR}'\n")

success, skipped = 0, []

for img_filename in image_files:
    stem, ext = os.path.splitext(img_filename)
    img_id    = stem[:-2]

    mask_filename = f"{img_id}_C.png"
    image_path    = os.path.join(IMAGES_DIR, img_filename)
    mask_path     = os.path.join(MASKS_DIR, mask_filename)

    if not os.path.exists(mask_path):
        print(f"⚠️  Mask not found for {img_id} (expected {mask_filename}), skipping.")
        skipped.append(img_id)
        continue

    print(f"Processing {img_id}...")

    image     = Image.open(image_path).convert("RGB")
    mask      = Image.open(mask_path).convert("RGB")
    orig_size = image.size

    image_resized = image.resize((512, 512))
    mask_resized  = mask.resize((512, 512))

    generator = torch.Generator("cuda").manual_seed(SEED)

    results = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_resized,
        mask_image=mask_resized,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        strength=STRENGTH,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES,
    ).images

    pair_dir = os.path.join(OUTPUT_DIR, img_id)
    os.makedirs(pair_dir, exist_ok=True)

    for i, result in enumerate(results):
        result_resized = result.resize(orig_size, Image.LANCZOS)
        out_path = os.path.join(pair_dir, f"{img_id}_result_{i}.png")
        result_resized.save(out_path)

    print(f"✅  {img_id} → {NUM_IMAGES} results saved to '{pair_dir}'\n")
    success += 1

print(f"Done. {success} pairs processed.")
if skipped:
    print(f"Skipped (no mask found): {skipped}")

