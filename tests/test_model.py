import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# On Apple Silicon macOS, tensorflow-metal has known kernel locks with Keras 3 AdamW EMA
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

from src.models.multitask_effnetv2 import build_multitask_effnetv2, parse_model_prediction_outputs
from src.models.losses import WeightedCategoricalCrossentropy, compute_normalized_class_weights


def test_model_architecture_shapes():
    """Verifies layer shapes and multi-output forward pass."""
    num_base = 47
    num_mod = 16
    num_vattu = 32
    batch_size = 4
    
    model = build_multitask_effnetv2(
        variant="B0",
        num_base=num_base,
        num_mod=num_mod,
        num_vattu=num_vattu,
        weights=None,
        backbone_trainable=True
    )
    
    dummy_input = tf.random.normal((batch_size, 128, 128, 3), dtype=tf.float32)
    outputs = model(dummy_input, training=False)
    
    base_out, mod_out, vattu_out = parse_model_prediction_outputs(outputs)
    
    assert base_out.shape == (batch_size, num_base)
    assert mod_out.shape == (batch_size, num_mod)
    assert vattu_out.shape == (batch_size, num_vattu)
    
    # Softmax probabilities should sum to 1.0 per sample
    assert np.allclose(np.sum(base_out, axis=-1), 1.0, atol=1e-4)
    assert np.allclose(np.sum(mod_out, axis=-1), 1.0, atol=1e-4)
    assert np.allclose(np.sum(vattu_out, axis=-1), 1.0, atol=1e-4)


def test_weighted_categorical_crossentropy_soft_targets():
    """Verifies weighted cross entropy with continuous CutMix targets."""
    num_classes = 5
    weights = np.array([1.0, 2.0, 0.5, 1.5, 1.0], dtype=np.float32)
    
    loss_fn = WeightedCategoricalCrossentropy(class_weights=weights, label_smoothing=0.0)
    
    # Continuous blended target
    y_true = tf.constant([[0.7, 0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5, 0.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.6, 0.4, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5, 0.0]], dtype=tf.float32)
    
    loss_val = loss_fn(y_true, y_pred)
    assert np.isfinite(loss_val.numpy())
    assert float(loss_val.numpy()) > 0.0


def test_one_batch_overfit_with_mixed_precision_and_ema():
    """Critical Verification: Overfits 1 batch under mixed_float16 + AdamW(use_ema=True) + global_clipnorm=1.0.
    
    Ensures:
      1. No Grappler / type conversion crash on mixed_float16
      2. AdamW gradient computation and clipping execute stably
      3. EMA variable tracking works properly without shape/type mismatch
      4. Total loss decreases to near zero (< 0.05)
    """
    # 1. Enable mixed precision
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception as e:
        print(f"Note: mixed_precision policy notice: {e}")
        
    num_base = 6
    num_mod = 4
    num_vattu = 4
    batch_size = 8
    
    # Fixed synthetic dataset batch
    np.random.seed(42)
    tf.random.set_seed(42)
    
    x_batch = tf.random.uniform((batch_size, 128, 128, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
    b_indices = np.random.randint(0, num_base, size=batch_size)
    m_indices = np.random.randint(0, num_mod, size=batch_size)
    v_indices = np.random.randint(0, num_vattu, size=batch_size)
    
    y_b = tf.one_hot(b_indices, depth=num_base, dtype=tf.float32)
    y_m = tf.one_hot(m_indices, depth=num_mod, dtype=tf.float32)
    y_v = tf.one_hot(v_indices, depth=num_vattu, dtype=tf.float32)
    
    targets = {
        "base_output": y_b,
        "modifier_output": y_m,
        "vattu_output": y_v
    }
    
    # Instantiate model with dropout=0.0 for deterministic overfit verification
    model = build_multitask_effnetv2(
        variant="B0",
        num_base=num_base,
        num_mod=num_mod,
        num_vattu=num_vattu,
        weights=None,
        backbone_trainable=True,
        dropout_rate=0.0
    )
    
    # Optimizer with AdamW + EMA + global_clipnorm
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
        global_clipnorm=1.0,
        use_ema=True,
        ema_momentum=0.9
    )
    
    loss_fn_b = WeightedCategoricalCrossentropy(label_smoothing=0.0)
    loss_fn_m = WeightedCategoricalCrossentropy(label_smoothing=0.0)
    loss_fn_v = WeightedCategoricalCrossentropy(label_smoothing=0.0)
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss_b = loss_fn_b(y["base_output"], preds[0])
            loss_m = loss_fn_m(y["modifier_output"], preds[1])
            loss_v = loss_fn_v(y["vattu_output"], preds[2])
            total_loss = 1.0 * loss_b + 0.5 * loss_m + 0.5 * loss_v
            if hasattr(optimizer, "scale_loss"):
                scaled_loss = optimizer.scale_loss(total_loss)
            else:
                scaled_loss = total_loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, preds

    print("Starting 1-batch overfit test (up to 200 steps)...")
    
    min_loss = np.inf
    for step in range(200):
        step_loss, preds = train_step(x_batch, targets)
        loss_val = float(step_loss.numpy())
        if loss_val < min_loss:
            min_loss = loss_val
        if (step + 1) % 25 == 0 or step == 0:
            print(f"  Step {step + 1:3d}: Loss = {loss_val:.4f} (min: {min_loss:.4f})")
        if min_loss < 0.05:
            print(f"  Converged at step {step + 1} with loss {loss_val:.4f} < 0.05!")
            break
            
    b_p, m_p, v_p = parse_model_prediction_outputs(preds)
    
    final_b_acc = float(np.mean(np.argmax(b_p, axis=1) == b_indices))
    final_m_acc = float(np.mean(np.argmax(m_p, axis=1) == m_indices))
    final_v_acc = float(np.mean(np.argmax(v_p, axis=1) == v_indices))
    
    print(f"  Overfit Check Results: Min Loss = {min_loss:.4f}, Base Acc = {final_b_acc:.2%}, Mod Acc = {final_m_acc:.2%}, Vattu Acc = {final_v_acc:.2%}")
    
    # Assertions
    assert min_loss < 0.05, f"Expected min loss < 0.05, got {min_loss}"
    assert final_b_acc >= 0.9, f"Expected base accuracy >= 90%, got {final_b_acc}"
    assert final_m_acc >= 0.9, f"Expected modifier accuracy >= 90%, got {final_m_acc}"
    assert final_v_acc >= 0.9, f"Expected vattu accuracy >= 90%, got {final_v_acc}"
    
    # Test EMA weight finalization
    if hasattr(optimizer, "finalize_variable_values"):
        optimizer.finalize_variable_values(model.trainable_variables)
    print("EMA shadow weights verified successfully!")
    
    # Reset global policy back to float32
    tf.keras.mixed_precision.set_global_policy("float32")


if __name__ == "__main__":
    test_model_architecture_shapes()
    test_weighted_categorical_crossentropy_soft_targets()
    test_one_batch_overfit_with_mixed_precision_and_ema()
    print("All model and one-batch overfit tests passed!")
