# Computes masked PSNR, SSIM, LPIPS for all generated images.
#
# Folder structure:
#   A (GT):    V0_*.png, V1_*.png        — one ground truth per scene (10 total)
#   B (gen):   subfolders V0_00000/,     — each subfolder has 4 generated samples
#              V0_00001/, V1_00000/, ...
#   C (masks): V0_00000_C.png, ...       — one combined mask per pair
#
# Output: metrics.xlsx with 4 sheets
#   Sheet 1 — All Samples      : one row per generated image
#   Sheet 2 — Averaged per Pair: avg of 4 samples per test image
#   Sheet 3 — Averaged by Scene: avg per GT category
#   Sheet 4 — Best per Pair    : best-scoring sample per test image

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# CONFIG
GT_DIR        = "A/"
GENERATED_DIR = "B/"
MASKS_DIR     = "C/"
OUTPUT_EXCEL  = "metrics.xlsx"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# AlexNet in LPIPS has 5 max-pool layers (stride 2 each).
# 64px is the safe minimum spatial size to survive all layers without error.
LPIPS_MIN = 64

print(f"Using device: {DEVICE}")
print("Loading LPIPS model...")
loss_fn = lpips.LPIPS(net="alex").to(DEVICE)
loss_fn.eval()

VALID_EXT = {".png", ".jpg", ".jpeg"}

def get_scene_id(name):
    """V0_00000 -> V0"""
    return name.split("_")[0]

def get_pair_id(name):
    """V0_00000_C.png -> V0_00000"""
    parts = os.path.splitext(name)[0].split("_")
    return f"{parts[0]}_{parts[1]}"

# Build GT map: scene_id -> full path
gt_map = {}
for f in os.listdir(GT_DIR):
    if os.path.splitext(f)[1].lower() in VALID_EXT:
        gt_map[get_scene_id(f)] = os.path.join(GT_DIR, f)
print(f"Found {len(gt_map)} GT scenes: {sorted(gt_map.keys())}")

# Build mask map: pair_id -> full path
mask_map = {}
for f in os.listdir(MASKS_DIR):
    if os.path.splitext(f)[1].lower() in VALID_EXT:
        mask_map[get_pair_id(f)] = os.path.join(MASKS_DIR, f)
print(f"Found {len(mask_map)} masks")

# Collect pair subfolders from B/
pair_folders = sorted([
    d for d in os.listdir(GENERATED_DIR)
    if os.path.isdir(os.path.join(GENERATED_DIR, d))
])
print(f"Found {len(pair_folders)} pair folders\n")

# Masked metric computation
def compute_metrics(gen_path, gt_path, mask_path):
    gen      = np.array(Image.open(gen_path).convert("RGB"))
    gt       = np.array(Image.open(gt_path).convert("RGB"))
    mask_img = Image.open(mask_path).convert("L")

    # Resize GT and mask to match generated image size if needed
    if gen.shape[:2] != gt.shape[:2]:
        gt = np.array(Image.open(gt_path).convert("RGB").resize(
            (gen.shape[1], gen.shape[0]), Image.LANCZOS))
    if (mask_img.size[1], mask_img.size[0]) != gen.shape[:2]:
        mask_img = mask_img.resize((gen.shape[1], gen.shape[0]), Image.NEAREST)
    mask = np.array(mask_img) > 127

    # Crop to bounding box of mask
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None, None, None

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    gen_crop = gen[y1:y2+1, x1:x2+1].astype(np.float32) / 255.0
    gt_crop  = gt[y1:y2+1,  x1:x2+1].astype(np.float32) / 255.0

    # PSNR
    psnr = peak_signal_noise_ratio(gt_crop, gen_crop, data_range=1.0)

    # SSIM
    ssim = structural_similarity(gt_crop, gen_crop, channel_axis=2, data_range=1.0)

    # LPIPS
    # AlexNet requires a minimum spatial size of 64x64 to survive all
    # max-pool layers without a "output size too small" RuntimeError.
    # Small crops are padded symmetrically using reflect mode so that no
    # artificial hard edges are introduced that would bias the score.
    # PSNR and SSIM always use the raw unpadded crop.
    crop_h, crop_w = gen_crop.shape[:2]

    if crop_h < LPIPS_MIN or crop_w < LPIPS_MIN:
        pad_h      = max(0, LPIPS_MIN - crop_h)
        pad_w      = max(0, LPIPS_MIN - crop_w)
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        gen_lpips  = np.pad(gen_crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
        gt_lpips   = np.pad(gt_crop,  ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
    else:
        gen_lpips = gen_crop
        gt_lpips  = gt_crop

    gen_t = torch.tensor(gen_lpips).permute(2, 0, 1).unsqueeze(0).to(DEVICE) * 2 - 1
    gt_t  = torch.tensor(gt_lpips).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  * 2 - 1
    with torch.no_grad():
        lpips_val = loss_fn(gen_t, gt_t).item()

    return round(psnr, 4), round(ssim, 4), round(lpips_val, 4)

# Main evaluation loop
rows = []
skipped_pairs = []
total_processed = 0

for i, pair_id in enumerate(pair_folders):
    scene_id  = get_scene_id(pair_id)
    gt_path   = gt_map.get(scene_id)
    mask_path = mask_map.get(pair_id)

    if gt_path is None:
        print(f"Warning: No GT found for scene {scene_id!r} (pair {pair_id}), skipping.")
        skipped_pairs.append(pair_id)
        continue
    if mask_path is None:
        print(f"Warning: No mask found for pair {pair_id!r}, skipping.")
        skipped_pairs.append(pair_id)
        continue

    pair_dir     = os.path.join(GENERATED_DIR, pair_id)
    sample_files = sorted([
        f for f in os.listdir(pair_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    ])

    if not sample_files:
        print(f"Warning: No images found in {pair_dir}, skipping.")
        skipped_pairs.append(pair_id)
        continue

    for sample_file in sample_files:
        gen_path = os.path.join(pair_dir, sample_file)
        psnr, ssim, lpips_val = compute_metrics(gen_path, gt_path, mask_path)

        if psnr is None:
            print(f"Warning: Empty mask crop for {pair_id}/{sample_file}, skipping.")
            continue

        rows.append({
            "pair_id":     pair_id,
            "scene_id":    scene_id,
            "sample_file": sample_file,
            "psnr":        psnr,
            "ssim":        ssim,
            "lpips":       lpips_val,
            "gt_file":     os.path.basename(gt_path),
            "mask_file":   os.path.basename(mask_path),
        })
        total_processed += 1

    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(pair_folders)} pairs ({total_processed} images)...")

print(f"\nEvaluated {total_processed} images across {len(pair_folders) - len(skipped_pairs)} pairs.")
if skipped_pairs:
    print(f"Skipped pairs: {skipped_pairs}")

# Build DataFrames
df_all = pd.DataFrame(rows, columns=[
    "pair_id", "scene_id", "sample_file",
    "psnr", "ssim", "lpips",
    "gt_file", "mask_file",
])

# Sheet 2 — average of N samples per pair
df_pair_avg = (
    df_all.groupby(["scene_id", "pair_id"])[["psnr", "ssim", "lpips"]]
    .agg(
        psnr_mean  = ("psnr",  "mean"),
        ssim_mean  = ("ssim",  "mean"),
        lpips_mean = ("lpips", "mean"),
        n_samples  = ("psnr",  "count"),
    )
    .reset_index()
    .sort_values(["scene_id", "pair_id"])
    .round(4)
)

# Sheet 3 — averaged by scene (pair-averages first, then scene-level mean)
df_scene = (
    df_pair_avg
    .groupby("scene_id")[["psnr_mean", "ssim_mean", "lpips_mean"]]
    .agg(
        psnr_mean  = ("psnr_mean",  "mean"),
        psnr_std   = ("psnr_mean",  "std"),
        ssim_mean  = ("ssim_mean",  "mean"),
        ssim_std   = ("ssim_mean",  "std"),
        lpips_mean = ("lpips_mean", "mean"),
        lpips_std  = ("lpips_mean", "std"),
        n_pairs    = ("psnr_mean",  "count"),
    )
    .reset_index()
    .sort_values("scene_id")
    .round(4)
)

overall = pd.DataFrame([{
    "scene_id":   "OVERALL",
    "psnr_mean":  round(df_pair_avg["psnr_mean"].mean(), 4),
    "psnr_std":   round(df_pair_avg["psnr_mean"].std(), 4),
    "ssim_mean":  round(df_pair_avg["ssim_mean"].mean(), 4),
    "ssim_std":   round(df_pair_avg["ssim_mean"].std(), 4),
    "lpips_mean": round(df_pair_avg["lpips_mean"].mean(), 4),
    "lpips_std":  round(df_pair_avg["lpips_mean"].std(), 4),
    "n_pairs":    len(df_pair_avg),
}])
df_scene = pd.concat([df_scene, overall], ignore_index=True)

# Sheet 4 — best sample per pair, selected by lowest LPIPS
idx_best = df_all.groupby("pair_id")["lpips"].idxmin()
df_best  = (
    df_all.loc[idx_best]
    .rename(columns={
        "psnr":  "psnr_best",
        "ssim":  "ssim_best",
        "lpips": "lpips_best",
    })
    [["pair_id", "scene_id", "sample_file", "psnr_best", "ssim_best", "lpips_best"]]
    .sort_values(["scene_id", "pair_id"])
    .reset_index(drop=True)
)

# Write Excel
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    df_all.to_excel(      writer, sheet_name="1 - All Samples",       index=False)
    df_pair_avg.to_excel( writer, sheet_name="2 - Averaged per Pair", index=False)
    df_scene.to_excel(    writer, sheet_name="3 - Averaged by Scene", index=False)
    df_best.to_excel(     writer, sheet_name="4 - Best per Pair",     index=False)

print(f"\nSaved -> {OUTPUT_EXCEL}")
print(f"  Sheet 1 rows : {len(df_all)}")
print(f"  Sheet 2 rows : {len(df_pair_avg)}")
print(f"  Sheet 3 rows : {len(df_scene)}")
print(f"  Sheet 4 rows : {len(df_best)}")
print("\nScene summary:")
print(df_scene.to_string(index=False))