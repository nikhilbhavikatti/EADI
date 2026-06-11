import cv2
import numpy as np
from pathlib import Path

# CONFIG
DATASET_ROOT = Path("Final_dataset")
IMG_DIRNAME = "shadow_images"
MASK_DIRNAME = "shadow_masks"

# NEW output folder (inside each V*)
OUT_DIRNAME = "cropped_shadow_images"

PAD = 2  # pixels padding around bbox

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_v_folder(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("V") and p.name[1:].isdigit()

def find_mask_for_frame(mask_dir: Path, frame_id: str):
    for ext in IMG_EXTS:
        p = mask_dir / f"{frame_id}_shadow_mask{ext}"
        if p.exists():
            return p
    hits = list(mask_dir.glob(f"{frame_id}_shadow_mask.*"))
    return hits[0] if hits else None

def find_image_for_frame(img_dir: Path, frame_id: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{frame_id}_shadow_only{ext}"
        if p.exists():
            return p
    hits = list(img_dir.glob(f"{frame_id}_shadow_only.*"))
    return hits[0] if hits else None

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

def clamp_bbox(x0, y0, x1, y1, W, H, pad):
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)
    return x0, y0, x1, y1

def main():
    v_folders = sorted(
        [p for p in DATASET_ROOT.iterdir() if is_v_folder(p)],
        key=lambda p: int(p.name[1:])
    )

    total = 0

    for v_dir in v_folders:
        img_dir = v_dir / IMG_DIRNAME
        mask_dir = v_dir / MASK_DIRNAME
        out_dir = v_dir / OUT_DIRNAME

        if not img_dir.exists() or not mask_dir.exists():
            print(f"[SKIP] {v_dir.name}")
            continue

        out_dir.mkdir(exist_ok=True)

        mask_files = [p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

        count = 0

        for mpath in mask_files:
            if not mpath.stem.endswith("_shadow_mask"):
                continue

            frame_id = mpath.stem.replace("_shadow_mask", "")

            ipath = find_image_for_frame(img_dir, frame_id)
            if ipath is None:
                continue

            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(str(ipath))

            if mask is None or img is None:
                continue

            H, W = img.shape[:2]

            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            bb = bbox_from_mask(mask)
            if bb is None:
                continue

            x0, y0, x1, y1 = clamp_bbox(*bb, W, H, PAD)

            cropped = img[y0:y1, x0:x1]

            out_path = out_dir / f"{frame_id}_shadow_only_cropped{ipath.suffix}"
            cv2.imwrite(str(out_path), cropped)

            count += 1
            total += 1

        print(f"[DONE] {v_dir.name}: {count} cropped")

    print(f"\nTotal cropped images: {total}")

if __name__ == "__main__":
    main()
