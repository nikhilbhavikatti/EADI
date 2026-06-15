# Context-Aware Feature Deviation (CFD) Score : based on OmniPaint paper (arXiv:2503.08677)
# CFD = d_context + d_hallucination (lower is better)

import os
import csv
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoModel, AutoImageProcessor


RESULTS_DIR  = ""   # folder: results/V38_00138/V38_00138_result_0.png
MASKS_DIR    = ""   # combined masks: V38_00138_C.png
OUTPUT_CSV   = "cfd_scores.csv"
SAM_CKPT     = "../models/sam_vit_h_4b8939.pth"
DEVICE       = "cuda"
NUM_RESULTS  = 4    # number of candidates per image pair


# Load SAM 
print("Loading SAM...")
sam      = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(DEVICE)
mask_gen = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
)

# Load DINOv2 via transformers (Python 3.8 compatible)
print("Loading DINOv2...")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino           = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
dino.eval()

# Feature extraction
def extract_features(img_np, region_mask):
    """
    Extract DINOv2 features for a masked region via mean-pooled patch tokens.
    Paper uses f(Omega) as region feature — mean pooling patch tokens is most
    faithful to this spatial region representation (vs CLS token which is global).
    """
    ys, xs = np.where(region_mask)
    if len(ys) == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    if y2 - y1 < 2 or x2 - x1 < 2:
        return None

    crop   = Image.fromarray(img_np[y1:y2+1, x1:x2+1]).convert("RGB")
    inputs = dino_processor(images=crop, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = dino(**inputs)
        # Skip CLS token (index 0), mean pool all patch tokens — eq. f(Omega) in paper
        feat = outputs.last_hidden_state[:, 1:, :].mean(dim=1)

    return feat / feat.norm(dim=-1, keepdim=True)


# CFD computation
def compute_cfd(inpainted_pil, mask_pil):
    """
    Compute CFD = d_context + d_hallucination as defined in OmniPaint paper Eq (14-16).

    Args:
        inpainted_pil : PIL Image (RGB) — the inpainted result
        mask_pil      : PIL Image (L)   — white=removed region, black=keep

    Returns:
        (cfd, d_context, d_hallucination) — all rounded to 4 decimal places
    """
    img_np  = np.array(inpainted_pil.convert("RGB"))
    mask_np = np.array(mask_pil.convert("L")) > 127  # bool
    H, W    = mask_np.shape

    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        return 0.0, 0.0, 0.0

    # Bounding box B of mask M
    y1, y2  = int(ys.min()), int(ys.max())
    x1, x2  = int(xs.min()), int(xs.max())
    bbox_mask = np.zeros((H, W), dtype=bool)
    bbox_mask[y1:y2+1, x1:x2+1] = True

    # SAM segmentation
    sam_masks = mask_gen.generate(img_np)
    seg_masks = [m["segmentation"] for m in sam_masks]

    # Classify masks near M (paper: focus on masks near M)
    nested, overlapping = [], []
    for seg in seg_masks:
        inter = np.logical_and(seg, mask_np).sum()
        if inter == 0:
            continue
        if inter == seg.sum():
            # Omega_M^n: fully inside mask → candidate hallucination (paper Eq 13)
            nested.append(seg)
        else:
            # Omega_M^o: crosses mask boundary → likely real background
            overlapping.append(seg)

    # Hallucination penalty d_hallucination (paper Eq 14)
    d_hallucination  = 0.0
    total_nested_px  = sum(s.sum() for s in nested)

    if total_nested_px > 0:
        for n_mask in nested:
            # Paper: "adj(M_j^o, M_i^n) = 1 if masks share boundary pixel
            # OR their one-pixel dilation overlaps" — apply symmetric dilation
            n_t        = torch.tensor(n_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            n_dilated  = torch.nn.functional.max_pool2d(n_t, 3, stride=1, padding=1)
            n_dilated_np = n_dilated.squeeze().numpy() > 0.5

            paired = np.zeros((H, W), dtype=bool)
            for o_mask in overlapping:
                # Symmetric adjacency: dilated-nested touches o_mask OR dilated-o_mask touches n_mask
                o_t         = torch.tensor(o_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                o_dilated_np = torch.nn.functional.max_pool2d(
                    o_t, 3, stride=1, padding=1).squeeze().numpy() > 0.5

                if (np.logical_and(n_dilated_np, o_mask).any() or
                        np.logical_and(o_dilated_np, n_mask).any()):
                    # Merge into paired overlapping mask (paper Eq 13)
                    paired = np.logical_or(paired, o_mask)

            if paired.sum() == 0:
                continue

            f_n = extract_features(img_np, n_mask)
            f_o = extract_features(img_np, paired)
            if f_n is None or f_o is None:
                continue

            # Eq (14): omega_i * (1 - f(nested)^T f(paired))
            sim = (f_n * f_o).sum().item()
            sim = max(-1.0, min(1.0, sim))   # clamp here too
            weight          = float(n_mask.sum()) / float(total_nested_px)
            d_hallucination += weight * (1.0 - sim)

    # Context coherence d_context (paper Eq 15)
    # Compare inpainted region Omega_M vs bounding-box-minus-mask Omega_{B\M}
    
    background_mask = np.logical_and(bbox_mask, ~mask_np)
    f_inp = extract_features(img_np, mask_np)
    f_bg  = extract_features(img_np, background_mask)

    if f_inp is not None and f_bg is not None:
        similarity = (f_inp * f_bg).sum().item()
        similarity = max(-1.0, min(1.0, similarity))   # clamp to valid cosine range
        d_context  = float(1.0 - similarity)
    else:
        d_context = 0.0

    # CFD = d_context + d_hallucination (paper Eq 16)
    cfd = d_context + d_hallucination
    return round(cfd, 4), round(d_context, 4), round(d_hallucination, 4)


# Main loop
rows = []
pair_dirs = sorted([
    d for d in os.listdir(RESULTS_DIR)
    if os.path.isdir(os.path.join(RESULTS_DIR, d))
])

print(f"\nFound {len(pair_dirs)} image pairs. Computing CFD scores...\n")

for img_id in pair_dirs:
    mask_path = os.path.join(MASKS_DIR, f"{img_id}_C.png")
    if not os.path.exists(mask_path):
        print(f"⚠️  Mask not found for {img_id}, skipping.")
        continue

    mask_pil = Image.open(mask_path).convert("L")
    pair_scores = []

    for i in range(NUM_RESULTS):
        result_path = os.path.join(RESULTS_DIR, img_id, f"{img_id}_result_{i}.png")
        if not os.path.exists(result_path):
            print(f"⚠️  Missing: {result_path}")
            continue

        result_pil = Image.open(result_path).convert("RGB")

        # Resize mask to match result if sizes differ
        if result_pil.size != mask_pil.size:
            mask_resized = mask_pil.resize(result_pil.size, Image.NEAREST)
        else:
            mask_resized = mask_pil

        cfd, d_ctx, d_hal = compute_cfd(result_pil, mask_resized)
        pair_scores.append((i, cfd))

        rows.append({
            "image_id":        img_id,
            "result_idx":      i,
            "result_path":     result_path,
            "cfd":             cfd,
            "d_context":       d_ctx,
            "d_hallucination": d_hal,
            "is_best":         False
        })
        print(f"  {img_id} result_{i} → CFD={cfd:.4f}  (ctx={d_ctx:.4f}, hal={d_hal:.4f})")

    # Mark best (lowest CFD) per pair
    if pair_scores:
        best_idx = min(pair_scores, key=lambda x: x[1])[0]
        for row in rows:
            if row["image_id"] == img_id and row["result_idx"] == best_idx:
                row["is_best"] = True
        print(f"  ✅ Best for {img_id}: result_{best_idx}\n")

# Save CSV
fieldnames = ["image_id", "result_idx", "result_path", "cfd",
              "d_context", "d_hallucination", "is_best"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved CFD scores for {len(rows)} results → {OUTPUT_CSV}")

# Print summary table
print(f"\n{'Image ID':<20} {'Best':<10} {'CFD':<8} {'d_context':<12} {'d_halluc'}")
print("-" * 62)
for row in rows:
    if row["is_best"]:
        print(f"{row['image_id']:<20} result_{row['result_idx']:<3}    "
              f"{row['cfd']:<8} {row['d_context']:<12} {row['d_hallucination']}")
