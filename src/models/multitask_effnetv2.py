"""Multi-Head EfficientNetV2 Telugu Grapheme Classifier.

Shared ImageNet-pretrained backbone with 3 independent dense head branches:
  1. Base Akshara (num_base classes)
  2. Vowel Modifier (num_mod classes)
  3. Conjunct Vattu (num_vattu classes)

Dynamically configured from empirical label maps without hardcoded primitives.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2S

from src.data.preprocessing import IMAGE_SIZE, NUM_CHANNELS


def build_multitask_effnetv2(variant: str = "B0",
                             num_base: int = 47,
                             num_mod: int = 16,
                             num_vattu: int = 32,
                             input_shape: Tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
                             weights: Optional[str] = "imagenet",
                             backbone_trainable: bool = False,
                             dropout_rate: float = 0.3) -> Model:
    """Builds multi-head EfficientNetV2 classifier.
    
    Args:
        variant: 'B0' or 'S' (default 'B0' for fast iteration, 'S' for scaled training).
        num_base: Number of base akshara classes (empirically derived).
        num_mod: Number of vowel modifier classes (empirically derived).
        num_vattu: Number of conjunct/vattu classes (empirically derived).
        input_shape: (128, 128, 3) 3-channel input tensor.
        weights: 'imagenet' or None.
        backbone_trainable: Initial trainability state of backbone layers.
        dropout_rate: Dropout rate for dense heads.
        
    Returns:
        tf.keras.Model with outputs [base_output, modifier_output, vattu_output].
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    variant_clean = variant.strip().upper()
    if variant_clean in ("B0", "V2B0", "EFFICIENTNETV2B0"):
        backbone = EfficientNetV2B0(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling=None
        )
    elif variant_clean in ("S", "V2S", "EFFICIENTNETV2S"):
        backbone = EfficientNetV2S(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling=None
        )
    else:
        raise ValueError(f"Unsupported EfficientNetV2 variant '{variant}'. Use 'B0' or 'S'.")
        
    backbone.trainable = backbone_trainable
    
    features = backbone.output
    pooled = layers.GlobalAveragePooling2D(name="backbone_gap")(features)
    pooled = layers.BatchNormalization(name="backbone_bn")(pooled)
    pooled = layers.Dropout(dropout_rate, name="backbone_dropout")(pooled)
    
    base_h = layers.Dense(256, activation="relu", name="base_dense")(pooled)
    base_h = layers.Dropout(dropout_rate, name="base_dropout")(base_h)
    base_out = layers.Dense(
        num_base,
        activation="softmax",
        dtype="float32",
        name="base_output"
    )(base_h)
    
    mod_h = layers.Dense(128, activation="relu", name="modifier_dense")(pooled)
    mod_h = layers.Dropout(dropout_rate, name="modifier_dropout")(mod_h)
    mod_out = layers.Dense(
        num_mod,
        activation="softmax",
        dtype="float32",
        name="modifier_output"
    )(mod_h)
    
    vattu_h = layers.Dense(128, activation="relu", name="vattu_dense")(pooled)
    vattu_h = layers.Dropout(dropout_rate, name="vattu_dropout")(vattu_h)
    vattu_out = layers.Dense(
        num_vattu,
        activation="softmax",
        dtype="float32",
        name="vattu_output"
    )(vattu_h)
    
    model = Model(
        inputs=inputs,
        outputs=[base_out, mod_out, vattu_out],
        name=f"multitask_effnetv2_{variant_clean.lower()}"
    )
    
    return model


def parse_model_prediction_outputs(raw_outputs: Union[List[tf.Tensor], Dict[str, tf.Tensor], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Safely extracts (base_probs, mod_probs, vattu_probs) across dict or list return types.
    
    Addresses pitfall #9: handles both Keras 2/3 and SavedModel return formats consistently.
    """
    if isinstance(raw_outputs, dict):
        base_p = raw_outputs["base_output"]
        mod_p = raw_outputs["modifier_output"]
        vattu_p = raw_outputs["vattu_output"]
    elif isinstance(raw_outputs, (list, tuple)):
        base_p = raw_outputs[0]
        mod_p = raw_outputs[1]
        vattu_p = raw_outputs[2]
    else:
        raise TypeError(f"Unexpected prediction output type: {type(raw_outputs)}")
        
    if isinstance(base_p, tf.Tensor):
        base_p = base_p.numpy()
    if isinstance(mod_p, tf.Tensor):
        mod_p = mod_p.numpy()
    if isinstance(vattu_p, tf.Tensor):
        vattu_p = vattu_p.numpy()
        
    return np.asarray(base_p), np.asarray(mod_p), np.asarray(vattu_p)
