"""Single Source of Truth Image Preprocessing Module.

All preprocessing logic for training, evaluation, TTA, and Streamlit inference
MUST import and use this exact module. No file is permitted to redefine image
resizing, padding, channel replication, or normalization constants.
"""

from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Canonical Image Dimensions
IMAGE_SIZE: int = 128
NUM_CHANNELS: int = 3

# Compute BACKGROUND_FILL_VALUE dynamically from pure white (255.0) via preprocess_input
# For EfficientNetV2 preprocess_input: x in [0, 255] -> (x / 127.5) - 1.0 -> 255.0 maps to 1.0
_SAMPLE_WHITE = tf.constant([[[255.0, 255.0, 255.0]]], dtype=tf.float32)
BACKGROUND_FILL_VALUE: float = float(preprocess_input(_SAMPLE_WHITE)[0, 0, 0].numpy())


def pad_to_square(image: tf.Tensor, pad_value: float = 255.0) -> tf.Tensor:
    """Pads a 2D or 3D grayscale/RGB image tensor to square aspect ratio with pad_value.
    
    Args:
        image: (H, W) or (H, W, C) float32 tensor in range [0, 255].
        pad_value: Pixel value for padding (default 255.0 for white paper background).
        
    Returns:
        Square tensor (S, S, C) where S = max(H, W).
    """
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    
    max_dim = tf.maximum(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if tf.rank(image) == 2:
        image = tf.expand_dims(image, axis=-1)
        
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    padded = tf.pad(image, paddings, mode="CONSTANT", constant_values=pad_value)
    return padded


def _ensure_single_channel_grayscale(img: tf.Tensor) -> tf.Tensor:
    """Converts 2D/3D/4D image tensors of arbitrary channel count to single-channel (H, W, 1)."""
    shape = tf.shape(img)
    rank = tf.rank(img)
    
    if rank == 2:
        return tf.expand_dims(img, axis=-1)
        
    c = shape[-1]
    if c == 1:
        return img
    elif c == 3:
        return tf.image.rgb_to_grayscale(img)
    elif c == 4:
        rgb = tf.slice(img, [0, 0, 0], [shape[0], shape[1], 3])
        alpha = tf.slice(img, [0, 0, 3], [shape[0], shape[1], 1]) / 255.0
        blended = rgb * alpha + 255.0 * (1.0 - alpha)
        return tf.image.rgb_to_grayscale(blended)
    else:
        # Fallback to first channel
        return tf.slice(img, [0, 0, 0], [shape[0], shape[1], 1])


@tf.function
def preprocess_image(raw_image_bytes_or_array: Union[tf.Tensor, np.ndarray, bytes], 
                     img_size: int = IMAGE_SIZE) -> tf.Tensor:
    """The ONLY place image preprocessing logic is allowed to live.
    
    Pipeline:
      1. Decode raw bytes or convert array -> tf.float32 [0, 255]
      2. Convert RGB/RGBA to single grayscale channel (H, W, 1)
      3. Aspect-ratio preserving pad-to-square with white (255.0) background
      4. Bilinear resize to (img_size, img_size)
      5. Replicate 1 channel to 3 channels (img_size, img_size, 3)
      6. EfficientNetV2 preprocess_input normalization
      
    Args:
        raw_image_bytes_or_array: String tensor of encoded image bytes, or (H, W) / (H, W, C) array.
        img_size: Target square dimension (default 128).
        
    Returns:
        (img_size, img_size, 3) float32 tensor ready for model input.
    """
    if isinstance(raw_image_bytes_or_array, (bytes, bytearray)):
        img = tf.io.decode_image(raw_image_bytes_or_array, channels=1, expand_animations=False)
        img = tf.cast(img, tf.float32)
    elif isinstance(raw_image_bytes_or_array, tf.Tensor) and raw_image_bytes_or_array.dtype == tf.string:
        img = tf.io.decode_image(raw_image_bytes_or_array, channels=1, expand_animations=False)
        img = tf.cast(img, tf.float32)
    else:
        tensor = tf.convert_to_tensor(raw_image_bytes_or_array)
        img = tf.cast(tensor, tf.float32)
        img = _ensure_single_channel_grayscale(img)
            
    # Pad to square with white (255.0) background
    padded = pad_to_square(img, pad_value=255.0)
    
    # Bilinear resize to (img_size, img_size)
    resized = tf.image.resize(padded, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    
    # Replicate single grayscale channel to 3 channels for EfficientNet ImageNet weights
    replicated = tf.repeat(resized, repeats=3, axis=-1)
    
    # Normalize via EfficientNetV2 official preprocess_input
    normalized = preprocess_input(replicated)
    
    return normalized
