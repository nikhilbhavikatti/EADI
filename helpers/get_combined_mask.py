# combine_masks.py
import os
import re
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# CONFIG
MASK_DIR    = "COMPLETE_TEST/"      # folder containing all mask images
OUTPUT_DIR  = "COMPLETE_TEST/"      # folder to save combined masks
DILATE_ITER = 5                 # iterations to bridge gap between object+shadow

os.makedirs(OUTPUT_DIR, exist_ok=True)

pattern = re.compile(r"^(.+)_O(\..+)$")

combined_count = 0
missing_shadow = []

for filename in sorted(os.listdir(MASK_DIR)):
    match = pattern.match(filename)
    if not match:
        continue

    img_id = match.group(1)   # e.g. V38_00138
    ext    = match.group(2)   # e.g. .png

    object_path = os.path.join(MASK_DIR, f"{img_id}_O{ext}")
    shadow_path = os.path.join(MASK_DIR, f"{img_id}_S{ext}")

    if not os.path.exists(shadow_path):
        print(f"⚠️  Shadow mask missing for {img_id}, skipping.")
        missing_shadow.append(img_id)
        continue

    obj_img    = Image.open(object_path).convert("L")
    shadow_img = Image.open(shadow_path).convert("L")

    # Use object mask size as reference — resize shadow if sizes differ
    if obj_img.size != shadow_img.size:
        print(f"⚠️  Size mismatch for {img_id}: O={obj_img.size} S={shadow_img.size} — resizing S to match O")
        shadow_img = shadow_img.resize(obj_img.size, Image.NEAREST)

    obj_np    = np.array(obj_img)
    shadow_np = np.array(shadow_img)

    # Combine with OR
    combined = np.clip(obj_np.astype(np.uint16) + shadow_np.astype(np.uint16), 0, 255).astype(np.uint8)

    # Dilate to bridge any gap between object base and shadow
    combined_bool = binary_dilation(combined > 127, iterations=DILATE_ITER)
    combined_out  = (combined_bool * 255).astype(np.uint8)

    out_path = os.path.join(OUTPUT_DIR, f"{img_id}_C{ext}")
    Image.fromarray(combined_out).save(out_path)
    print(f"✅  {img_id}_O{ext} + {img_id}_S{ext} → {img_id}_C{ext} | size={obj_img.size}")
    combined_count += 1

print(f"\nDone. {combined_count} combined masks saved to '{OUTPUT_DIR}'")
if missing_shadow:
    print(f"Skipped (no shadow mask): {missing_shadow}")
