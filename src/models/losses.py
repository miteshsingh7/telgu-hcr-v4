"""Custom Loss Functions for Multi-Head Grapheme Classification with Class Weighting.

Bakes normalized class weights directly into categorical cross-entropy loss, ensuring
seamless compatibility with CutMix-blended soft targets and label smoothing.
"""

from typing import Optional, Union, Dict, Any
import numpy as np
import tensorflow as tf


def compute_normalized_class_weights(class_counts: np.ndarray, 
                                     clip_min: float = 0.1, 
                                     clip_max: float = 10.0) -> np.ndarray:
    """Computes balanced class frequency weights normalized to mean 1.0.
    
    Formula:
        w_c = total_samples / (num_classes * count_c)
        normalized so that sum(w_c * count_c) / total_samples == 1.0
        
    Args:
        class_counts: 1D array of count per class index.
        clip_min: Lower bound clipping.
        clip_max: Upper bound clipping.
        
    Returns:
        1D float32 array of shape (num_classes,)
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    total_samples = np.sum(counts)
    num_classes = len(counts)
    
    # Avoid zero division
    safe_counts = np.maximum(counts, 1.0)
    weights = total_samples / (num_classes * safe_counts)
    
    # Normalize so expectation under empirical distribution is 1.0
    mean_w = np.sum(weights * counts) / total_samples
    normalized_weights = weights / (mean_w + 1e-8)
    
    # Clip extreme weights to prevent gradient destabilization
    clipped = np.clip(normalized_weights, clip_min, clip_max).astype(np.float32)
    return clipped


@tf.keras.utils.register_keras_serializable(package="TeluguHCR")
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Categorical Cross-Entropy with per-class weight multipliers.
    
    Fully compatible with continuous/soft CutMix targets and label smoothing:
        L(y, p) = - sum_c w_c * y_c * log(p_c + eps)
    """
    
    def __init__(self, 
                 class_weights: Optional[Union[np.ndarray, list, tf.Tensor]] = None,
                 label_smoothing: float = 0.0,
                 from_logits: bool = False,
                 name: str = "weighted_categorical_crossentropy",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)
        
        if class_weights is not None:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        else:
            self.class_weights = None

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        
        # Apply label smoothing if configured and not already smoothed
        if self.label_smoothing > 0.0:
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
            
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
        # Safe log
        eps = tf.keras.backend.epsilon()
        y_pred_safe = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        
        # Per-sample, per-class cross entropy
        ce = -y_true * tf.math.log(y_pred_safe)
        
        if self.class_weights is not None:
            # Broadcast weights across batch
            ce = ce * self.class_weights
            
        # Sum across class dimension -> (Batch,)
        sample_loss = tf.reduce_sum(ce, axis=-1)
        return tf.reduce_mean(sample_loss)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        weights_list = None
        if self.class_weights is not None:
            try:
                weights_list = self.class_weights.numpy().tolist()
            except Exception:
                weights_list = tf.keras.backend.get_value(self.class_weights).tolist()
        config.update({
            "class_weights": weights_list,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits
        })
        return config
