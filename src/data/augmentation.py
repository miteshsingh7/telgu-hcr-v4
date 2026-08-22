"""Single Source of Truth Image Augmentation and Multi-Head CutMix Pipeline.

Augmentation parameters and fill values are defined HERE ONLY.
All training scripts and data loaders import and use this module.
"""

from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers

from src.data.preprocessing import BACKGROUND_FILL_VALUE, IMAGE_SIZE, NUM_CHANNELS


def build_augmentation_pipeline(img_size: int = IMAGE_SIZE, 
                                rotation_degrees: float = 5.0,
                                translation_factor: float = 0.05,
                                zoom_factor: float = 0.05) -> tf.keras.Sequential:
    """Builds standard spatial augmentation pipeline for Telugu handwritten glyphs.
    
    Uses BACKGROUND_FILL_VALUE from preprocessing.py for all edge padding.
    """
    rot_factor = rotation_degrees / 360.0
    
    return tf.keras.Sequential([
        layers.RandomRotation(
            factor=(-rot_factor, rot_factor),
            fill_mode="constant",
            fill_value=BACKGROUND_FILL_VALUE,
            name="aug_rotation"
        ),
        layers.RandomTranslation(
            height_factor=(-translation_factor, translation_factor),
            width_factor=(-translation_factor, translation_factor),
            fill_mode="constant",
            fill_value=BACKGROUND_FILL_VALUE,
            name="aug_translation"
        ),
        layers.RandomZoom(
            height_factor=(-zoom_factor, zoom_factor),
            width_factor=(-zoom_factor, zoom_factor),
            fill_mode="constant",
            fill_value=BACKGROUND_FILL_VALUE,
            name="aug_zoom"
        ),
    ], name="telugu_augmentation_pipeline")


def _sample_beta_distribution(alpha: float, shape: tf.TensorShape) -> tf.Tensor:
    """Samples from Beta(alpha, alpha) using two Gamma distributions."""
    gamma_1 = tf.random.gamma(shape, alpha=alpha)
    gamma_2 = tf.random.gamma(shape, alpha=alpha)
    return gamma_1 / (gamma_1 + gamma_2 + 1e-8)


@tf.function
def apply_cutmix(images: tf.Tensor,
                 base_labels: tf.Tensor,
                 mod_labels: tf.Tensor,
                 vattu_labels: tf.Tensor,
                 alpha: float = 0.4) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Applies batch-level CutMix mixing image patches and all 3 multi-head label targets.
    
    Args:
        images: (B, H, W, C) float32 image batch.
        base_labels: (B, num_base) float32 probability/smoothed label vectors.
        mod_labels: (B, num_mod) float32 probability/smoothed label vectors.
        vattu_labels: (B, num_vattu) float32 probability/smoothed label vectors.
        alpha: Beta distribution parameter (default 0.4).
        
    Returns:
        (mixed_images, mixed_base, mixed_mod, mixed_vattu)
    """
    batch_size = tf.shape(images)[0]
    h = tf.cast(tf.shape(images)[1], tf.float32)
    w = tf.cast(tf.shape(images)[2], tf.float32)
    
    # 1. Sample lambda from Beta(alpha, alpha)
    lam = _sample_beta_distribution(alpha, [1])[0]
    
    # 2. Compute cut bounding box dimensions
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(w * cut_ratio, tf.int32)
    cut_h = tf.cast(h * cut_ratio, tf.int32)
    
    # 3. Sample random bounding box center
    cx = tf.random.uniform([], minval=0, maxval=tf.cast(w, tf.int32), dtype=tf.int32)
    cy = tf.random.uniform([], minval=0, maxval=tf.cast(h, tf.int32), dtype=tf.int32)
    
    # 4. Clamp bounding box coordinates
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, tf.cast(w, tf.int32))
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, tf.cast(h, tf.int32))
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, tf.cast(w, tf.int32))
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, tf.cast(h, tf.int32))
    
    # 5. Compute exact adjusted lambda based on actual cut area
    actual_cut_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    total_area = h * w
    lam_adjusted = 1.0 - (actual_cut_area / (total_area + 1e-8))
    
    # 6. Shuffle indices across batch
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    
    # 7. Construct spatial mask: 1.0 where original image stays, 0.0 where cut from shuffled image
    mask_top = tf.ones([tf.shape(images)[0], y1, tf.cast(w, tf.int32), tf.shape(images)[3]], dtype=tf.float32)
    mask_bottom = tf.ones([tf.shape(images)[0], tf.cast(h, tf.int32) - y2, tf.cast(w, tf.int32), tf.shape(images)[3]], dtype=tf.float32)
    
    middle_left = tf.ones([tf.shape(images)[0], y2 - y1, x1, tf.shape(images)[3]], dtype=tf.float32)
    middle_cut = tf.zeros([tf.shape(images)[0], y2 - y1, x2 - x1, tf.shape(images)[3]], dtype=tf.float32)
    middle_right = tf.ones([tf.shape(images)[0], y2 - y1, tf.cast(w, tf.int32) - x2, tf.shape(images)[3]], dtype=tf.float32)
    
    middle_strip = tf.concat([middle_left, middle_cut, middle_right], axis=2)
    mask = tf.concat([mask_top, middle_strip, mask_bottom], axis=1)
    
    mixed_images = images * mask + shuffled_images * (1.0 - mask)
    
    # 8. Mix all three label heads simultaneously with the exact same adjusted lambda
    shuffled_base = tf.gather(base_labels, indices)
    shuffled_mod = tf.gather(mod_labels, indices)
    shuffled_vattu = tf.gather(vattu_labels, indices)
    
    mixed_base = lam_adjusted * base_labels + (1.0 - lam_adjusted) * shuffled_base
    mixed_mod = lam_adjusted * mod_labels + (1.0 - lam_adjusted) * shuffled_mod
    mixed_vattu = lam_adjusted * vattu_labels + (1.0 - lam_adjusted) * shuffled_vattu
    
    return mixed_images, mixed_base, mixed_mod, mixed_vattu
