"""Stratified Dataset Split Generator.

Generates a frozen, reproducible 80/10/10 train/val/test split across all classes
and exports train.csv, val.csv, test.csv, and label_maps.json.

WARNING — DATA LEAKAGE LIMITATION:
    This module performs a class-stratified random split WITHOUT writer-identity
    awareness. In handwritten character recognition, images from the same writer
    may appear in both train and test splits, artificially inflating validation
    accuracy. If writer IDs become available, the split should be refactored to
    group by writer ID to produce honest generalization metrics.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data.decomposition import (
    decompose_class_name,
    build_and_validate_label_maps
)


def collect_dataset_images(dataset_root: str, relative_paths: bool = True) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Recursively collects all images and extracts canonical class names."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{dataset_root}' does not exist.")
        
    records = []
    class_set = set()
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        cat = cat_dir.name
        
        if cat.lower() == "guninthamulu":
            for c_dir in sorted(cat_dir.iterdir()):
                if not c_dir.is_dir() or c_dir.name.startswith("."):
                    continue
                for v_dir in sorted(c_dir.iterdir()):
                    if not v_dir.is_dir() or v_dir.name.startswith("."):
                        continue
                    class_name = f"{cat}__{c_dir.name}__{v_dir.name}"
                    class_set.add(class_name)
                    for img_file in sorted(v_dir.iterdir()):
                        if img_file.is_file() and img_file.suffix.lower() in valid_extensions:
                            fp = str(img_file.relative_to(root)) if relative_paths else str(img_file.resolve())
                            records.append({
                                "file_path": fp,
                                "class_name": class_name,
                                "category": cat
                            })
        else:
            for c_dir in sorted(cat_dir.iterdir()):
                if not c_dir.is_dir() or c_dir.name.startswith("."):
                    continue
                class_name = f"{cat}__{c_dir.name}"
                class_set.add(class_name)
                for img_file in sorted(c_dir.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in valid_extensions:
                        fp = str(img_file.relative_to(root)) if relative_paths else str(img_file.resolve())
                        records.append({
                            "file_path": fp,
                            "class_name": class_name,
                            "category": cat
                        })
                        
    return records, sorted(list(class_set))


def create_frozen_splits(dataset_root: str,
                         output_dir: str = "outputs",
                         train_ratio: float = 0.8,
                         val_ratio: float = 0.1,
                         test_ratio: float = 0.1,
                         relative_paths: bool = True,
                         seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates stratified train/val/test CSV splits and saves label maps."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning dataset from: {dataset_root}")
    records, class_names = collect_dataset_images(dataset_root, relative_paths=relative_paths)
    print(f"Found {len(records):,} total images across {len(class_names)} classes.")
    
    label_maps_path = out_dir / "label_maps.json"
    label_maps = build_and_validate_label_maps(class_names, output_path=str(label_maps_path))
    print(f"Saved label maps to {label_maps_path}")
    print(f"Vocabulary: {label_maps['num_base_classes']} base, "
          f"{label_maps['num_modifier_classes']} modifier, "
          f"{label_maps['num_vattu_classes']} vattu primitives.")
    print(f"Unique structural compound combinations: {label_maps['num_unique_combinations']} (across {len(class_names)} folders).")
    
    df = pd.DataFrame(records)
    base_map = label_maps["base_map"]
    mod_map = label_maps["mod_map"]
    vattu_map = label_maps["vattu_map"]
    
    base_letters, mod_letters, vattu_letters = [], [], []
    base_indices, mod_indices, vattu_indices = [], [], []
    
    for cname in df["class_name"]:
        b_str, m_str, v_str = decompose_class_name(cname)
        base_letters.append(b_str)
        mod_letters.append(m_str)
        vattu_letters.append(v_str)
        base_indices.append(base_map[b_str])
        mod_indices.append(mod_map[m_str])
        vattu_indices.append(vattu_map[v_str])
        
    df["base_letter"] = base_letters
    df["vowel_modifier"] = mod_letters
    df["vattu"] = vattu_letters
    df["base_idx"] = base_indices
    df["modifier_idx"] = mod_indices
    df["vattu_idx"] = vattu_indices
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    rng = np.random.RandomState(seed)
    
    for cname, group in df.groupby("class_name"):
        n = len(group)
        if n < 3:
            train_dfs.append(group)
        else:
            shuffled = group.sample(frac=1.0, random_state=seed)
            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = 1
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                    
            train_dfs.append(shuffled.iloc[:n_train])
            val_dfs.append(shuffled.iloc[n_train:n_train + n_val])
            test_dfs.append(shuffled.iloc[n_train + n_val:])
            
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Split created successfully:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df):.1%}) -> {train_path}")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df):.1%}) -> {val_path}")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df):.1%}) -> {test_path}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate frozen stratified splits for Telugu HCR v4")
    parser.add_argument("--data_dir", type=str, 
                        default="/Users/miteshsingh/Documents/projects/telugu-hcr-v3/data/Final Dataset of Telugu Handwritten Chararcters/Test1",
                        help="Path to raw dataset root")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for CSVs and JSON")
    parser.add_argument("--absolute_paths", action="store_true", help="Store absolute file paths instead of relative")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    create_frozen_splits(dataset_root=args.data_dir, output_dir=args.output_dir, relative_paths=not args.absolute_paths, seed=args.seed)
