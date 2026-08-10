"""High-Throughput tf.data Dataset Pipeline.

Strictly imports preprocessing from src.data.preprocessing and augmentation from
src.data.augmentation. Computes per-head class weights and formats multi-head targets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import tensorflow as tf

from src.data.preprocessing import preprocess_image, IMAGE_SIZE
from src.data.augmentation import build_augmentation_pipeline, apply_cutmix
from src.models.losses import compute_normalized_class_weights


def load_label_maps(label_maps_path_or_dict: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Loads label maps dictionary from disk or passes through dictionary."""
    if isinstance(label_maps_path_or_dict, dict):
        return label_maps_path_or_dict
    with open(label_maps_path_or_dict, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_head_weights_from_df(df: pd.DataFrame, 
                                 label_maps: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Computes normalized class weight vectors for each classification head."""
    num_base = label_maps["num_base_classes"]
    num_mod = label_maps["num_modifier_classes"]
    num_vattu = label_maps["num_vattu_classes"]
    
    base_counts = np.bincount(df["base_idx"].values, minlength=num_base)
    mod_counts = np.bincount(df["modifier_idx"].values, minlength=num_mod)
    vattu_counts = np.bincount(df["vattu_idx"].values, minlength=num_vattu)
    
    w_base = compute_normalized_class_weights(base_counts)
    w_mod = compute_normalized_class_weights(mod_counts)
    w_vattu = compute_normalized_class_weights(vattu_counts)
    
    return {
        "base_output": w_base,
        "modifier_output": w_mod,
        "vattu_output": w_vattu
    }


def resolve_dataset_root(candidate_paths: Optional[list] = None) -> Optional[Path]:
    """Finds existing dataset root among standard candidate locations or auto-discovers on Kaggle/Local."""
    import os
    home = Path.home()
    defaults = [
        home / "Documents/projects/telugu-hcr-v3/data/Final Dataset of Telugu Handwritten Chararcters/Test1",
        home / "Downloads/telgu_dataset/Test1",
        Path("/kaggle/input/telugu-handwritten-character-dataset/Final Dataset of Telugu Handwritten Chararcters/Test1"),
        Path("/kaggle/input/telugu-hcr/Final Dataset of Telugu Handwritten Chararcters/Test1"),
        Path("/kaggle/input/telugu-dataset/Test1"),
        Path("data/Final Dataset of Telugu Handwritten Chararcters/Test1"),
        Path("data/Test1"),
    ]
    candidates = (candidate_paths or []) + defaults
    for p in candidates:
        if p and Path(p).exists() and Path(p).is_dir():
            return Path(p)
            
    # Auto-discover under /kaggle/input (recursive search for full dataset with Guninthamulu and Achulu)
    for search_base in (Path("/kaggle/input"), Path("data"), home / "Downloads", home / "Documents/projects"):
        if search_base.exists():
            for root, dirs, _ in os.walk(str(search_base)):
                # Skip sample folders
                if "data_samples" in root:
                    continue
                dirs_lower = {d.lower(): d for d in dirs}
                if "guninthamulu" in dirs_lower and "achulu" in dirs_lower:
                    return Path(root)
                    
    return None


def remap_file_paths(file_paths: np.ndarray, dataset_root: Optional[Union[str, Path]] = None) -> np.ndarray:
    """Remaps file paths if the saved paths do not exist on the current machine (e.g. on Kaggle/Colab)."""
    if len(file_paths) == 0:
        return file_paths
        
    sample_path = Path(file_paths[0])
    if sample_path.is_file():
        return file_paths
        
    target_root = Path(dataset_root) if dataset_root else resolve_dataset_root()
    if target_root is None or not target_root.exists():
        # Path does not exist and no fallback found; return as is
        return file_paths
        
    # Extract relative path starting from known categories or 'Test1'
    known_cats = ("achulu", "hallulu", "guninthamulu", "othulu")
    remapped = []
    
    for fp in file_paths:
        p_str = str(fp).replace("\\", "/")
        p_lower = p_str.lower()
        
        # Check if already relative starting with a known category
        starts_with_cat = False
        for cat in known_cats:
            if p_lower.startswith(f"{cat}/"):
                new_path = str(target_root / p_str)
                starts_with_cat = True
                break
                
        if not starts_with_cat:
            idx = -1
            for cat in known_cats:
                pos = p_lower.find(f"/{cat}/")
                if pos != -1:
                    idx = pos + 1
                    break
            if idx != -1:
                rel_part = p_str[idx:]
                new_path = str(target_root / rel_part)
            else:
                new_path = str(target_root / p_str)
                
        remapped.append(new_path)
        
    return np.array(remapped, dtype=object)


def create_telugu_dataset(csv_path_or_df: Union[str, Path, pd.DataFrame],
                          label_maps_or_path: Union[str, Path, Dict[str, Any]],
                          dataset_root: Optional[Union[str, Path]] = None,
                          batch_size: int = 128,
                          is_training: bool = True,
                          use_augmentation: bool = True,
                          rotation_degrees: float = 5.0,
                          translation_factor: float = 0.05,
                          zoom_factor: float = 0.05,
                          use_cutmix: bool = True,
                          cutmix_alpha: float = 0.4,
                          cutmix_probability: float = 0.5,
                          label_smoothing: float = 0.1,
                          img_size: int = IMAGE_SIZE,
                          shuffle_buffer: int = 10000) -> Tuple[tf.data.Dataset, int, Dict[str, np.ndarray]]:
    """Builds a high-performance tf.data input pipeline for multi-head training or evaluation.
    
    Args:
        csv_path_or_df: Path to CSV or pandas DataFrame with file_path and head index columns.
        label_maps_or_path: Label maps dictionary or path to label_maps.json.
        dataset_root: Optional override path for dataset root (auto-remaps paths on Kaggle/Colab).
        batch_size: Batch size (default 128).
        is_training: If True, shuffles and applies augmentations/CutMix.
        use_augmentation: If True, applies spatial rotation/translation/zoom.
        rotation_degrees: Rotation angle range (+/- degrees).
        translation_factor: Translation fraction (+/- fraction).
        zoom_factor: Zoom fraction (+/- fraction).
        use_cutmix: If True, applies multi-head CutMix blending.
        cutmix_alpha: Beta distribution parameter for CutMix.
        cutmix_probability: Fraction of training batches CutMix is applied to (default 0.5).
        label_smoothing: Label smoothing factor (applied to one-hot vectors).
        img_size: Image dimension (default 128).
        shuffle_buffer: Buffer size for training shuffle.
        
    Returns:
        (tf_dataset, steps_per_epoch, class_weights_dict)
    """
    label_maps = load_label_maps(label_maps_or_path)
    num_base = label_maps["num_base_classes"]
    num_mod = label_maps["num_modifier_classes"]
    num_vattu = label_maps["num_vattu_classes"]
    
    if isinstance(csv_path_or_df, (str, Path)):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()
        
    raw_paths = df["file_path"].astype(str).values
    file_paths = remap_file_paths(raw_paths, dataset_root=dataset_root)
    base_indices = df["base_idx"].astype(np.int32).values
    mod_indices = df["modifier_idx"].astype(np.int32).values
    vattu_indices = df["vattu_idx"].astype(np.int32).values
    
    num_samples = len(file_paths)
    steps_per_epoch = max(1, num_samples // batch_size) if is_training else max(1, int(np.ceil(num_samples / batch_size)))
    
    class_weights = compute_head_weights_from_df(df, label_maps)
    
    dataset = tf.data.Dataset.from_tensor_slices((
        file_paths, base_indices, mod_indices, vattu_indices
    ))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        
    @tf.function
    def _load_and_preprocess(fpath, b_idx, m_idx, v_idx):
        img_bytes = tf.io.read_file(fpath)
        img = preprocess_image(img_bytes, img_size=img_size)
        
        b_one_hot = tf.one_hot(b_idx, depth=num_base, dtype=tf.float32)
        m_one_hot = tf.one_hot(m_idx, depth=num_mod, dtype=tf.float32)
        v_one_hot = tf.one_hot(v_idx, depth=num_vattu, dtype=tf.float32)
        
        if is_training and label_smoothing > 0.0:
            b_one_hot = b_one_hot * (1.0 - label_smoothing) + (label_smoothing / float(num_base))
            m_one_hot = m_one_hot * (1.0 - label_smoothing) + (label_smoothing / float(num_mod))
            v_one_hot = v_one_hot * (1.0 - label_smoothing) + (label_smoothing / float(num_vattu))
            
        return img, b_one_hot, m_one_hot, v_one_hot
        
    dataset = dataset.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    
    if is_training and use_augmentation:
        aug_pipeline = build_augmentation_pipeline(
            img_size=img_size,
            rotation_degrees=rotation_degrees,
            translation_factor=translation_factor,
            zoom_factor=zoom_factor
        )
        @tf.function
        def _apply_spatial_aug(imgs, b_labels, m_labels, v_labels):
            augmented_imgs = tf.cast(aug_pipeline(imgs, training=True), tf.float32)
            return augmented_imgs, b_labels, m_labels, v_labels
        dataset = dataset.map(_apply_spatial_aug, num_parallel_calls=tf.data.AUTOTUNE)
        
    if is_training and use_cutmix:
        @tf.function
        def _apply_batch_cutmix(imgs, b_labels, m_labels, v_labels):
            imgs_f32 = tf.cast(imgs, tf.float32)
            b_f32 = tf.cast(b_labels, tf.float32)
            m_f32 = tf.cast(m_labels, tf.float32)
            v_f32 = tf.cast(v_labels, tf.float32)

            def _mixed():
                return apply_cutmix(imgs_f32, b_f32, m_f32, v_f32, alpha=cutmix_alpha)

            def _unmixed():
                return imgs_f32, b_f32, m_f32, v_f32

            apply_this_batch = tf.random.uniform([], minval=0.0, maxval=1.0) < cutmix_probability
            return tf.cond(apply_this_batch, _mixed, _unmixed)
        dataset = dataset.map(_apply_batch_cutmix, num_parallel_calls=tf.data.AUTOTUNE)
        
    @tf.function
    def _format_outputs(imgs, b_labels, m_labels, v_labels):
        return imgs, {
            "base_output": b_labels,
            "modifier_output": m_labels,
            "vattu_output": v_labels
        }
        
    dataset = dataset.map(_format_outputs, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, steps_per_epoch, class_weights
