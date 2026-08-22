"""Unified Training Pipeline for Multi-Head Telugu Handwritten Character Recognizer.

Two-phase training:
  Phase 1: Frozen backbone warmup (5 epochs)
  Phase 2: Unfrozen fine-tuning (40-50 epochs) with AdamW + EMA + Cosine Decay + Gradient Clipping.

Includes Early Timing & Kaggle Session Extrapolation Callback and Full-State Checkpointing.
"""

import os
import sys
import time
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf

from src.data.preprocessing import IMAGE_SIZE
from src.data.dataset import create_telugu_dataset, load_label_maps
from src.models.multitask_effnetv2 import build_multitask_effnetv2
from src.models.losses import WeightedCategoricalCrossentropy
from src.checkpointing import FullStateCheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TeluguHCR.Train")


class TimingExtrapolationCallback(tf.keras.callbacks.Callback):
    """Measures epoch duration early and extrapolates projected total runtime against session limits."""
    
    def __init__(self, 
                 total_warmup_epochs: int, 
                 total_finetune_epochs: int, 
                 max_hours: float = 11.5,
                 timing_epochs: int = 2):
        super().__init__()
        self.total_warmup = total_warmup_epochs
        self.total_finetune = total_finetune_epochs
        self.max_hours = max_hours
        self.timing_epochs = timing_epochs
        self.epoch_times = []
        self.epoch_start = 0.0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()
        
    def on_epoch_end(self, epoch, logs=None):
        duration = time.perf_counter() - self.epoch_start
        self.epoch_times.append(duration)
        avg_sec = np.mean(self.epoch_times[-self.timing_epochs:])
        logger.info(f"Epoch {epoch + 1} completed in {duration:.1f}s (rolling avg: {avg_sec:.1f}s/epoch)")
        
        # Extrapolate when sufficient epochs are measured
        if len(self.epoch_times) == self.timing_epochs:
            total_epochs = self.total_warmup + self.total_finetune
            remaining_epochs = max(0, total_epochs - (epoch + 1))
            projected_remaining_hours = (remaining_epochs * avg_sec) / 3600.0
            projected_total_hours = (total_epochs * avg_sec) / 3600.0
            
            logger.info("=" * 70)
            logger.info("KAGGLE RUNTIME BUDGET EXTRAPOLATION REPORT:")
            logger.info(f"  Measured speed:          {avg_sec:.1f} sec/epoch ({avg_sec/60.0:.2f} min/epoch)")
            logger.info(f"  Total planned epochs:    {total_epochs} ({self.total_warmup} warmup + {self.total_finetune} fine-tune)")
            logger.info(f"  Remaining epochs:        {remaining_epochs}")
            logger.info(f"  Est. remaining duration: {projected_remaining_hours:.2f} hours (Total: {projected_total_hours:.2f}h | Limit: {self.max_hours:.1f}h)")
            
            if projected_remaining_hours > self.max_hours:
                logger.warning(
                    f"  [ALERT] Projected remaining duration ({projected_remaining_hours:.2f}h) exceeds safe session budget ({self.max_hours:.1f}h)! "
                    f"Consider lowering finetune_epochs or using EfficientNetV2B0."
                )
            else:
                logger.info(f"  [OK] Remaining training fits comfortably within session limits.")
            logger.info("=" * 70)


class FullStateCheckpointCallback(tf.keras.callbacks.Callback):
    """Persists full training state (model weights + optimizer + epoch metadata) after every epoch."""
    
    def __init__(self, manager: FullStateCheckpointManager, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.manager = manager
        self.monitor = monitor
        self.mode = mode
        self.best_val = np.inf if mode == "min" else -np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_metric = logs.get(self.monitor, None)
        is_best = False
        if current_metric is not None:
            if self.mode == "min" and current_metric < self.best_val:
                self.best_val = current_metric
                is_best = True
            elif self.mode == "max" and current_metric > self.best_val:
                self.best_val = current_metric
                is_best = True
                
        # Save state atomically
        self.manager.save_state(
            model=self.model,
            optimizer=self.model.optimizer,
            epoch=epoch + 1,
            metrics=logs,
            is_best=is_best
        )


def create_lr_schedule(base_lr: float,
                       min_lr: float,
                       total_epochs: int,
                       warmup_epochs: int,
                       steps_per_epoch: int) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Constructs linear warmup + cosine decay learning rate schedule."""
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps
    
    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self):
            super().__init__()
            self.warmup_steps = float(warmup_steps)
            self.decay_steps = float(decay_steps)
            self.base_lr = float(base_lr)
            self.min_lr = float(min_lr)
            
        def __call__(self, step):
            step_f = tf.cast(step, tf.float32)
            
            # Linear Warmup
            warmup_lr = self.base_lr * (step_f / tf.maximum(1.0, self.warmup_steps))
            
            # Cosine Decay
            decay_step = tf.maximum(0.0, step_f - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * tf.minimum(decay_step, self.decay_steps) / tf.maximum(1.0, self.decay_steps)))
            decayed_lr = (self.base_lr - self.min_lr) * cosine_decay + self.min_lr
            
            return tf.where(step_f < self.warmup_steps, warmup_lr, decayed_lr)
            
        def get_config(self):
            return {
                "base_lr": self.base_lr,
                "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps,
                "decay_steps": self.decay_steps
            }
            
    return WarmupCosineDecay()


def run_training(config_path: str,
                 resume: bool = False,
                 resume_path: Optional[str] = None,
                 dataset_root: Optional[str] = None,
                 custom_epochs: Optional[int] = None,
                 custom_batch_size: Optional[int] = None,
                 custom_variant: Optional[str] = None):
    """Executes the full end-to-end multi-head training regimen."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    b_cfg = cfg.get("timing_budget", {})
    
    variant = custom_variant or m_cfg["variant"]
    batch_size = custom_batch_size or d_cfg["batch_size"]
    img_size = m_cfg["img_size"]
    effective_data_root = dataset_root or d_cfg.get("dataset_root", None)
    
    warmup_epochs = t_cfg["warmup_epochs"]
    finetune_epochs = custom_epochs or t_cfg["finetune_epochs"]
    total_epochs = warmup_epochs + finetune_epochs
    
    # 1. Mixed Precision Setup
    if t_cfg.get("mixed_precision", True):
        logger.info("Enabling mixed_float16 global policy.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        
    # 2. Load Label Maps
    label_maps = load_label_maps(d_cfg["label_maps"])
    num_base = label_maps["num_base_classes"]
    num_mod = label_maps["num_modifier_classes"]
    num_vattu = label_maps["num_vattu_classes"]
    logger.info(f"Loaded label maps: {num_base} base, {num_mod} modifier, {num_vattu} vattu classes.")
    
    # 3. Create Datasets
    logger.info(f"Building training dataset from {d_cfg['train_csv']}...")
    train_ds, train_steps, class_weights = create_telugu_dataset(
        csv_path_or_df=d_cfg["train_csv"],
        label_maps_or_path=label_maps,
        dataset_root=effective_data_root,
        batch_size=batch_size,
        is_training=True,
        use_augmentation=d_cfg.get("use_augmentation", True),
        use_cutmix=d_cfg.get("use_cutmix", True),
        cutmix_alpha=d_cfg.get("cutmix_alpha", 0.4),
        label_smoothing=d_cfg.get("label_smoothing", 0.1),
        img_size=img_size
    )
    
    logger.info(f"Building validation dataset from {d_cfg['val_csv']}...")
    val_ds, val_steps, _ = create_telugu_dataset(
        csv_path_or_df=d_cfg["val_csv"],
        label_maps_or_path=label_maps,
        dataset_root=effective_data_root,
        batch_size=batch_size,
        is_training=False,
        use_augmentation=False,
        use_cutmix=False,
        label_smoothing=0.0,
        img_size=img_size
    )
        img_size=img_size
    )
    logger.info(f"Dataset ready: {train_steps} train steps/epoch, {val_steps} val steps/epoch (batch={batch_size}).")
    
    # 4. Instantiate Checkpoint Manager
    checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_manager = FullStateCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=t_cfg.get("max_checkpoints_to_keep", 3),
        monitor="val_loss",
        mode="min"
    )
    
    # 5. Build Model
    logger.info(f"Instantiating EfficientNetV2-{variant} multi-head model...")
    model = build_multitask_effnetv2(
        variant=variant,
        num_base=num_base,
        num_mod=num_mod,
        num_vattu=num_vattu,
        input_shape=(img_size, img_size, 3),
        weights=m_cfg.get("weights", "imagenet"),
        backbone_trainable=False,  # Start frozen for Phase 1
        dropout_rate=m_cfg.get("dropout_rate", 0.3)
    )
    
    # 6. Build LR Schedule & Optimizer
    lr_schedule = create_lr_schedule(
        base_lr=t_cfg.get("learning_rate", 1e-4),
        min_lr=t_cfg.get("min_learning_rate", 1e-6),
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        steps_per_epoch=train_steps
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=t_cfg.get("weight_decay", 1e-4),
        global_clipnorm=t_cfg.get("global_clipnorm", 1.0),
        use_ema=t_cfg.get("use_ema", True),
        ema_momentum=t_cfg.get("ema_momentum", 0.999)
    )
    
    # 7. Compile Model with Weighted Losses
    losses = {
        "base_output": WeightedCategoricalCrossentropy(class_weights=class_weights["base_output"]),
        "modifier_output": WeightedCategoricalCrossentropy(class_weights=class_weights["modifier_output"]),
        "vattu_output": WeightedCategoricalCrossentropy(class_weights=class_weights["vattu_output"])
    }
    loss_weights = t_cfg.get("loss_weights", {"base_output": 1.0, "modifier_output": 0.5, "vattu_output": 0.5})
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "base_output": ["accuracy"],
            "modifier_output": ["accuracy"],
            "vattu_output": ["accuracy"]
        }
    )
    
    # 8. Resume Checkpoint if Requested
    start_epoch = 0
    if resume:
        start_epoch, meta = ckpt_manager.restore_state(model, optimizer, checkpoint_path_or_tag=resume_path)
        logger.info(f"Resumed from epoch {start_epoch} (initial val_loss: {meta.get('monitored_value')})")
        
    # 9. Callbacks
    timing_cb = TimingExtrapolationCallback(
        total_warmup_epochs=warmup_epochs,
        total_finetune_epochs=finetune_epochs,
        max_hours=b_cfg.get("max_session_hours", 11.5),
        timing_epochs=b_cfg.get("early_timing_epochs", 2)
    )
    ckpt_cb = FullStateCheckpointCallback(ckpt_manager, monitor="val_loss", mode="min")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=t_cfg.get("early_stopping_patience", 10),
        restore_best_weights=False,  # Managed cleanly by FullStateCheckpointManager
        verbose=1
    )
    
    callbacks = [timing_cb, ckpt_cb, early_stopping]
    
    # 10. Phase 1: Warmup with Frozen Backbone
    if start_epoch < warmup_epochs:
        logger.info(f"=== Phase 1: Warmup (Epochs {start_epoch + 1} -> {warmup_epochs}) [Backbone Frozen] ===")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            initial_epoch=start_epoch,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks
        )
        start_epoch = warmup_epochs
        
    # 11. Phase 2: Unfreeze Backbone and Fine-Tune
    logger.info(f"=== Phase 2: Fine-Tuning (Epochs {start_epoch + 1} -> {total_epochs}) [Backbone Unfrozen] ===")
    
    # Unfreeze backbone layers
    for layer in model.layers:
        layer.trainable = True
        
    # Recompile after modifying layer trainability
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "base_output": ["accuracy"],
            "modifier_output": ["accuracy"],
            "vattu_output": ["accuracy"]
        }
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=start_epoch,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks
    )
    
    # 12. Finalize EMA weights for final best checkpoint
    if hasattr(optimizer, "finalize_variable_values"):
        logger.info("Finalizing EMA shadow weights into model parameters...")
        optimizer.finalize_variable_values(model.trainable_variables)
        
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Telugu HCR v4 Multi-Head Classifier")
    parser.add_argument("--config", type=str, default="configs/multitask_effnetv2.yaml", help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Specific checkpoint tag or path to resume")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root directory override for path remapping")
    parser.add_argument("--epochs", type=int, default=None, help="Override finetune epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--variant", type=str, default=None, help="Override model variant (B0 or S)")
    args = parser.parse_args()
    
    run_training(
        config_path=args.config,
        resume=args.resume,
        resume_path=args.resume_path,
        dataset_root=args.dataset_root,
        custom_epochs=args.epochs,
        custom_batch_size=args.batch_size,
        custom_variant=args.variant
    )
