"""Stroke Texture Diagnostic: Laplacian Variance Analysis.

Measures edge sharpness / texture proxy (Laplacian variance) across
dataset categories and compares against digital canvas drawings.
Also saves a 5-image grid of post-preprocess real samples to outputs/debug_real_samples.png.
"""

import sys
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
PROJECT_ROOT = Path("/Users/miteshsingh/Documents/projects/telgu-hcr-v4")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_image, IMAGE_SIZE
from src.data.dataset import resolve_dataset_root
from app import crop_to_content


def analyze_stroke_texture(samples_per_cat: int = 50, seed: int = 42) -> None:
    dataset_root = resolve_dataset_root()
    if dataset_root is None or not dataset_root.exists():
        home = Path.home()
        candidates = [
            home / "Downloads/telgu_dataset/Test1",
            home / "Documents/projects/telugu-hcr-v3/data/Final Dataset of Telugu Handwritten Chararcters/Test1",
        ]
        for c in candidates:
            if c.exists():
                dataset_root = c
                break

    if dataset_root is None or not dataset_root.exists():
        print("ERROR: Could not locate dataset root.")
        return

    print(f"Using dataset root: {dataset_root}\n")

    categories = ["achulu", "hallulu", "Guninthamulu", "othulu"]
    cat_results: Dict[str, List[float]] = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    np.random.seed(seed)

    grid_images = []

    for cat in categories:
        cat_dir = None
        for child in dataset_root.iterdir():
            if child.is_dir() and child.name.lower() == cat.lower():
                cat_dir = child
                break

        if cat_dir is None or not cat_dir.exists():
            print(f"WARNING: Category directory '{cat}' not found under {dataset_root}")
            continue

        all_imgs = [
            p for p in cat_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        if not all_imgs:
            continue

        n_sample = min(samples_per_cat, len(all_imgs))
        sampled_imgs = np.random.choice(all_imgs, size=n_sample, replace=False)

        laplacian_vars = []
        for idx, img_path in enumerate(sampled_imgs):
            try:
                img = Image.open(img_path)
                img_arr = np.array(img)
                prep_tensor = preprocess_image(img_arr, img_size=IMAGE_SIZE)
                prep_np = prep_tensor.numpy()
                
                gray_channel = prep_np[..., 0].astype(np.float64)

                if cat.lower() in ("achulu", "hallulu", "guninthamulu") and len(grid_images) < 5:
                    u8_img = np.clip(gray_channel, 0, 255).astype(np.uint8)
                    grid_images.append(u8_img)

                lap = cv2.Laplacian(gray_channel, cv2.CV_64F)
                lap_var = float(lap.var())
                laplacian_vars.append(lap_var)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        cat_results[cat] = laplacian_vars

    if grid_images:
        out_dir = PROJECT_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_np = np.hstack(grid_images[:5])
        grid_pil = Image.fromarray(grid_np)
        grid_path = out_dir / "debug_real_samples.png"
        grid_pil.save(str(grid_path))
        print(f"Saved 5-sample grid to: {grid_path} (shape={grid_np.shape})\n")

    canvas = Image.new("RGBA", (450, 450), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.line([(160, 160), (290, 160)], fill=(0, 0, 0, 255), width=16)
    draw.line([(225, 135), (225, 160)], fill=(0, 0, 0, 255), width=16)
    draw.arc([(170, 160), (280, 310)], start=0, end=360, fill=(0, 0, 0, 255), width=16)
    draw.line([(150, 235), (300, 235)], fill=(0, 0, 0, 255), width=16)

    canvas_arr = np.array(canvas)
    canvas_cropped = crop_to_content(canvas_arr)
    prep_canvas = preprocess_image(canvas_cropped, img_size=IMAGE_SIZE).numpy()
    canvas_gray = prep_canvas[..., 0].astype(np.float64)
    canvas_lap = cv2.Laplacian(canvas_gray, cv2.CV_64F)
    canvas_lap_var = float(canvas_lap.var())

    print("=" * 68)
    print(f"{'Category':<16} | {'Count':<6} | {'Min Var':<12} | {'Mean Var':<12} | {'Max Var':<12}")
    print("-" * 68)

    base_vars = []
    for cat in categories:
        if cat in cat_results and cat_results[cat]:
            vals = cat_results[cat]
            if cat.lower() != "othulu":
                base_vars.extend(vals)
            print(f"{cat:<16} | {len(vals):<6} | {np.min(vals):<12.2f} | {np.mean(vals):<12.2f} | {np.max(vals):<12.2f}")

    print("=" * 68)

    print("\n" + "=" * 68)
    print(f"{'Group / Sample':<28} | {'Count':<6} | {'Min Var':<12} | {'Mean Var':<12} | {'Max Var':<12}")
    print("-" * 68)

    if base_vars:
        print(f"{'Base Classes (Ach+Hal+Gun)':<28} | {len(base_vars):<6} | {np.min(base_vars):<12.2f} | {np.mean(base_vars):<12.2f} | {np.max(base_vars):<12.2f}")

    if "othulu" in cat_results and cat_results["othulu"]:
        oth_vals = cat_results["othulu"]
        print(f"{'Othulu Alone':<28} | {len(oth_vals):<6} | {np.min(oth_vals):<12.2f} | {np.mean(oth_vals):<12.2f} | {np.max(oth_vals):<12.2f}")

    print("-" * 68)
    print(f"{'Single Canvas Sample':<28} | {'1':<6} | {canvas_lap_var:<12.2f} | {canvas_lap_var:<12.2f} | {canvas_lap_var:<12.2f}")
    print("=" * 68)


if __name__ == "__main__":
    analyze_stroke_texture(samples_per_cat=50)
