import os
import cv2
import numpy as np
from pathlib import Path

# CONFIG
DATASET_ROOT = Path("Final_dataset")   # contains V1, V2, ..., V160
OUT_FOLDER_NAME = "shadow_images"
EPS = 1e-3  # to avoid division by zero

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_video_folder(p: Path) -> bool:
    # Accepts folders like V1, V24, V160
    return p.is_dir() and p.name.startswith("V") and p.name[1:].isdigit()


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


# LOOP OVER V* FOLDERS
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT.resolve()}")

video_folders = sorted(
    [p for p in DATASET_ROOT.iterdir() if is_video_folder(p)],
    key=lambda p: int(p.name[1:])
)

if not video_folders:
    raise RuntimeError(f"No V* folders found in: {DATASET_ROOT.resolve()}")

for root in video_folders:
    bg_dir = root / "bg"
    obj_dir = root / "obj"
    sm_dir  = root / "shadow_masks"
    out_dir = root / OUT_FOLDER_NAME

    # Basic folder checks
    if not bg_dir.is_dir() or not obj_dir.is_dir() or not sm_dir.is_dir():
        print(f"[SKIP] {root.name}: missing required folder(s). Need bg/, obj/, shadow_masks/")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)

    # LOAD BACKGROUND IMAGE
    bg_files = list_images(bg_dir)
    if len(bg_files) != 1:
        print(f"[SKIP] {root.name}: bg/ must contain exactly 1 image, found {len(bg_files)}")
        continue

    B = cv2.imread(str(bg_files[0]))
    if B is None:
        print(f"[SKIP] {root.name}: failed to read background image: {bg_files[0].name}")
        continue

    B = B.astype(np.float32) / 255.0
    H, W = B.shape[:2]

    # PROCESS EACH FRAME
    obj_files = list_images(obj_dir)
    if not obj_files:
        print(f"[SKIP] {root.name}: no image files in obj/")
        continue

    ok_count, skip_count = 0, 0

    for obj_path in obj_files:
        frame_id = obj_path.stem  # e.g. "00001"
        sm_path = sm_dir / f"{frame_id}_shadow_mask.png"

        if not sm_path.exists():
            print(f"[{root.name}] [SKIP] Missing shadow mask: {sm_path.name}")
            skip_count += 1
            continue

        I = cv2.imread(str(obj_path))
        if I is None:
            print(f"[{root.name}] [SKIP] Failed to read obj image: {obj_path.name}")
            skip_count += 1
            continue
        I = I.astype(np.float32) / 255.0

        M = cv2.imread(str(sm_path), cv2.IMREAD_GRAYSCALE)
        if M is None:
            print(f"[{root.name}] [SKIP] Failed to read shadow mask: {sm_path.name}")
            skip_count += 1
            continue

        # Resize if needed (safety)
        if I.shape[:2] != (H, W):
            I = cv2.resize(I, (W, H), interpolation=cv2.INTER_LINEAR)
        if M.shape != (H, W):
            M = cv2.resize(M, (W, H), interpolation=cv2.INTER_NEAREST)

        # Binary shadow mask (1 = shadow)
        M = (M > 0).astype(np.float32)
        M3 = M[..., None]  # (H, W, 1)

        # Shadow attenuation
        attenuation = I / (B + EPS)
        attenuation = np.clip(attenuation, 0.0, 1.0)

        # Shadow-only image
        S = B * (M3 * attenuation + (1.0 - M3))

        out_path = out_dir / f"{frame_id}_shadow_only.jpg"
        cv2.imwrite(str(out_path), (S * 255.0).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        ok_count += 1

    print(f"[DONE] {root.name}: saved {ok_count} image(s) to {out_dir.name}/ (skipped {skip_count})")

print("All done.")
