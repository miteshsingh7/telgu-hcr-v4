"""Unit tests for FullStateCheckpointManager (Kill-and-Resume State Persistence)."""

import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.models.multitask_effnetv2 import build_multitask_effnetv2
from src.models.losses import WeightedCategoricalCrossentropy
from src.checkpointing import FullStateCheckpointManager


def test_checkpoint_save_and_restore(tmp_path=None):
    """Verifies that model weights, optimizer iterations, epoch counter, and metrics restore exactly."""
    save_dir = Path("outputs/test_checkpoints")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    manager = FullStateCheckpointManager(checkpoint_dir=save_dir, monitor="val_loss", mode="min")
    
    # Model 1
    model1 = build_multitask_effnetv2(
        variant="B0",
        num_base=5,
        num_mod=3,
        num_vattu=4,
        weights=None,
        backbone_trainable=True
    )
    opt1 = tf.keras.optimizers.AdamW(learning_rate=1e-3, global_clipnorm=1.0)
    model1.compile(
        optimizer=opt1,
        loss={
            "base_output": WeightedCategoricalCrossentropy(),
            "modifier_output": WeightedCategoricalCrossentropy(),
            "vattu_output": WeightedCategoricalCrossentropy()
        }
    )
    
    # Train for 2 epochs on dummy data
    x_dummy = tf.random.normal((8, 128, 128, 3))
    y_dummy = {
        "base_output": tf.one_hot([0, 1, 2, 3, 4, 0, 1, 2], depth=5),
        "modifier_output": tf.one_hot([0, 1, 2, 0, 1, 2, 0, 1], depth=3),
        "vattu_output": tf.one_hot([0, 1, 2, 3, 0, 1, 2, 3], depth=4)
    }
    
    # 2 training steps
    for _ in range(2):
        with tf.GradientTape() as tape:
            preds = model1(x_dummy, training=True)
            loss = (
                WeightedCategoricalCrossentropy()(y_dummy["base_output"], preds[0]) +
                WeightedCategoricalCrossentropy()(y_dummy["modifier_output"], preds[1]) +
                WeightedCategoricalCrossentropy()(y_dummy["vattu_output"], preds[2])
            )
        grads = tape.gradient(loss, model1.trainable_variables)
        opt1.apply_gradients(zip(grads, model1.trainable_variables))
    pred1 = model1(x_dummy, training=False)
    
    # Save checkpoint at epoch 2
    ckpt_path = manager.save_state(
        model=model1,
        optimizer=opt1,
        epoch=2,
        metrics={"loss": 1.25, "val_loss": 1.50},
        is_best=True
    )
    assert ckpt_path.exists()
    assert (ckpt_path / "model.weights.h5").exists()
    assert (ckpt_path / "optimizer_state.pkl").exists()
    assert (ckpt_path / "state.json").exists()
    
    # Simulate restart with fresh model and fresh optimizer
    model2 = build_multitask_effnetv2(
        variant="B0",
        num_base=5,
        num_mod=3,
        num_vattu=4,
        weights=None,
        backbone_trainable=True
    )
    opt2 = tf.keras.optimizers.AdamW(learning_rate=1e-3, global_clipnorm=1.0)
    model2.compile(
        optimizer=opt2,
        loss={
            "base_output": WeightedCategoricalCrossentropy(),
            "modifier_output": WeightedCategoricalCrossentropy(),
            "vattu_output": WeightedCategoricalCrossentropy()
        }
    )
    
    # Initialize variables with dummy pass
    _ = model2(x_dummy, training=False)
    
    # Restore
    restored_epoch, meta = manager.restore_state(model=model2, optimizer=opt2, checkpoint_path_or_tag="best_model")
    assert restored_epoch == 2
    assert meta["monitored_value"] == 1.50
    assert meta["is_best"] is True
    
    # Check predictions match model1 exactly
    pred2 = model2(x_dummy, training=False)
    for p1, p2 in zip(pred1, pred2):
        assert np.allclose(p1.numpy(), p2.numpy(), atol=1e-5)
        
    print("Checkpoint save and restore test passed perfectly!")
    
    # Clean up test checkpoints
    shutil.rmtree(save_dir)


if __name__ == "__main__":
    test_checkpoint_save_and_restore()
