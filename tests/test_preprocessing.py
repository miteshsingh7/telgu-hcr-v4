"""Unit tests for single source of truth preprocessing module."""

import numpy as np
import tensorflow as tf
from PIL import Image
import io

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from src.data.preprocessing import (
    preprocess_image,
    pad_to_square,
    BACKGROUND_FILL_VALUE,
    IMAGE_SIZE,
    NUM_CHANNELS
)


def test_background_fill_value():
    """Confirms BACKGROUND_FILL_VALUE matches EfficientNetV2 preprocess_input white."""
    assert isinstance(BACKGROUND_FILL_VALUE, float)
    sample_white = tf.constant([[[255.0, 255.0, 255.0]]], dtype=tf.float32)
    expected = float(preprocess_input(sample_white)[0, 0, 0].numpy())
    assert np.isclose(BACKGROUND_FILL_VALUE, expected, atol=1e-3)


def test_pad_to_square():
    """Verifies aspect-ratio preserving square padding."""
    rect_img = tf.ones((60, 100, 1), dtype=tf.float32) * 128.0
    padded = pad_to_square(rect_img, pad_value=255.0)
    assert padded.shape == (100, 100, 1)
    # Check that top/bottom borders were padded with 255.0
    assert float(padded[0, 50, 0]) == 255.0
    # Check that center retained original value
    assert float(padded[50, 50, 0]) == 128.0


def test_preprocess_numpy_array():
    """Verifies preprocessing from arbitrary shape numpy array."""
    raw_img = np.random.randint(0, 256, (80, 110, 3), dtype=np.uint8)
    processed = preprocess_image(raw_img, img_size=128)
    
    assert processed.shape == (128, 128, 3)
    assert processed.dtype == tf.float32
    # Pixel values are within valid preprocessed range
    assert tf.reduce_min(processed) >= 0.0 - 1e-4
    assert tf.reduce_max(processed) <= 255.0 + 1e-4


def test_preprocess_encoded_bytes():
    """Verifies preprocessing from raw image bytes."""
    img = Image.new("RGB", (90, 70), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    
    processed = preprocess_image(raw_bytes, img_size=IMAGE_SIZE)
    assert processed.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    assert processed.dtype == tf.float32
    # Pure white image should normalize to BACKGROUND_FILL_VALUE (255.0)
    assert np.allclose(processed.numpy(), BACKGROUND_FILL_VALUE, atol=0.1)


if __name__ == "__main__":
    test_background_fill_value()
    test_pad_to_square()
    test_preprocess_numpy_array()
    test_preprocess_encoded_bytes()
    print("All preprocessing tests passed!")
