"""Ink Density Diagnostic by Category.

Measures the percentage of ink pixels (pct_pixels_below_240) across
raw dataset categories using the unmodified preprocess_image() pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path("/Users/miteshsingh/Documents/projects/telgu-hcr-v4")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_image, IMAGE_SIZE
from src.data.dataset import resolve_dataset_root


def measure_ink_density(samples_per_cat: int = 50, seed: int = 42) -> None:
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
            print(f"WARNING: No images found in {cat_dir}")
            continue

        n_sample = min(samples_per_cat, len(all_imgs))
        sampled_imgs = np.random.choice(all_imgs, size=n_sample, replace=False)

        densities = []
        for img_path in sampled_imgs:
            try:
                img = Image.open(img_path)
                img_arr = np.array(img)
                preprocessed_tensor = preprocess_image(img_arr, img_size=IMAGE_SIZE)
                preprocessed_np = preprocessed_tensor.numpy()
                
                pct_ink = float((preprocessed_np[..., 0] < 240.0).mean() * 100.0)
                densities.append(pct_ink)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        cat_results[cat] = densities

    print("=" * 65)
    print(f"{'Category':<16} | {'Count':<6} | {'Min (%)':<10} | {'Mean (%)':<10} | {'Max (%)':<10}")
    print("-" * 65)

    base_densities = []
    for cat in categories:
        if cat in cat_results and cat_results[cat]:
            vals = cat_results[cat]
            if cat.lower() != "othulu":
                base_densities.extend(vals)
            print(f"{cat:<16} | {len(vals):<6} | {np.min(vals):<10.2f} | {np.mean(vals):<10.2f} | {np.max(vals):<10.2f}")

    print("=" * 65)

    print("\n" + "=" * 65)
    print(f"{'Group':<28} | {'Count':<6} | {'Min (%)':<10} | {'Mean (%)':<10} | {'Max (%)':<10}")
    print("-" * 65)

    if base_densities:
        print(f"{'Base Classes (Ach+Hal+Gun)':<28} | {len(base_densities):<6} | {np.min(base_densities):<10.2f} | {np.mean(base_densities):<10.2f} | {np.max(base_densities):<10.2f}")

    if "othulu" in cat_results and cat_results["othulu"]:
        oth_vals = cat_results["othulu"]
        print(f"{'Othulu Alone':<28} | {len(oth_vals):<6} | {np.min(oth_vals):<10.2f} | {np.mean(oth_vals):<10.2f} | {np.max(oth_vals):<10.2f}")

    print("=" * 65)


if __name__ == "__main__":
    measure_ink_density(samples_per_cat=50)
