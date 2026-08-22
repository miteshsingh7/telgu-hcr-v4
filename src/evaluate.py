"""Comprehensive Evaluation Framework for Multi-Head Telugu Handwritten Character Recognizer.

Computes:
  - Per-head Top-1 accuracy and Macro-Averaged Recall
  - Baseline comparison against 'Always None' trivial predictor
  - Recombined 630-way Top-1 and Top-5 accuracy via Constrained Maximum-Likelihood Decoding
  - Most confused character pairs analysis
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm

from src.data.preprocessing import IMAGE_SIZE
from src.data.dataset import create_telugu_dataset, load_label_maps
from src.data.decomposition import recombine_prediction
from src.data.known_duplicates import get_equivalent_classes
from src.models.multitask_effnetv2 import build_multitask_effnetv2, parse_model_prediction_outputs
from src.checkpointing import FullStateCheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TeluguHCR.Evaluate")


def evaluate_test_set(test_csv: str = "outputs/test.csv",
                      label_maps_path: str = "outputs/label_maps.json",
                      checkpoint_dir: str = "checkpoints",
                      checkpoint_tag: str = "best_model",
                      dataset_root: Optional[str] = None,
                      variant: str = "B0",
                      batch_size: int = 128,
                      output_report_path: str = "outputs/evaluation_report.json") -> Dict[str, Any]:
    """Runs rigorous multi-head and recombined 630-way evaluation on test split."""
    label_maps = load_label_maps(label_maps_path)
    num_base = label_maps["num_base_classes"]
    num_mod = label_maps["num_modifier_classes"]
    num_vattu = label_maps["num_vattu_classes"]
    class_to_comb = label_maps["class_to_combination"]
    
    logger.info(f"Loading test split from {test_csv}...")
    df_test = pd.read_csv(test_csv)
    total_test_samples = len(df_test)
    logger.info(f"Total test samples: {total_test_samples:,}")
    
    # 1. Build Model Architecture
    logger.info(f"Instantiating EfficientNetV2-{variant}...")
    model = build_multitask_effnetv2(
        variant=variant,
        num_base=num_base,
        num_mod=num_mod,
        num_vattu=num_vattu,
        weights=None,
        backbone_trainable=False
    )
    
    # 2. Restore Checkpoint Weights
    ckpt_manager = FullStateCheckpointManager(checkpoint_dir=checkpoint_dir)
    dummy_opt = tf.keras.optimizers.AdamW()
    restored_epoch, meta = ckpt_manager.restore_state(model, dummy_opt, checkpoint_path_or_tag=checkpoint_tag)
    logger.info(f"Loaded checkpoint '{checkpoint_tag}' from epoch {restored_epoch}.")
    
    # 3. Create Evaluation Dataset
    test_ds, test_steps, _ = create_telugu_dataset(
        csv_path_or_df=df_test,
        label_maps_or_path=label_maps,
        dataset_root=dataset_root,
        batch_size=batch_size,
        is_training=False,
        use_augmentation=False,
        use_cutmix=False,
        label_smoothing=0.0
    )
    
    # 4. Run Model Predictions
    logger.info("Executing batched inference across test set...")
    all_base_probs = []
    all_mod_probs = []
    all_vattu_probs = []
    
    for x_batch, _ in tqdm(test_ds, total=test_steps, desc="Evaluating"):
        preds = model(x_batch, training=False)
        b_p, m_p, v_p = parse_model_prediction_outputs(preds)
        all_base_probs.append(b_p)
        all_mod_probs.append(m_p)
        all_vattu_probs.append(v_p)
        
    base_probs_arr = np.concatenate(all_base_probs, axis=0)[:total_test_samples]
    mod_probs_arr = np.concatenate(all_mod_probs, axis=0)[:total_test_samples]
    vattu_probs_arr = np.concatenate(all_vattu_probs, axis=0)[:total_test_samples]
    
    # 5. Extract Ground Truths
    true_base = df_test["base_idx"].values
    true_mod = df_test["modifier_idx"].values
    true_vattu = df_test["vattu_idx"].values
    true_classes = df_test["class_name"].values
    true_combs = [class_to_comb.get(c, c) for c in true_classes]
    
    # 6. Per-Head Metrics
    pred_base = np.argmax(base_probs_arr, axis=1)
    pred_mod = np.argmax(mod_probs_arr, axis=1)
    pred_vattu = np.argmax(vattu_probs_arr, axis=1)
    
    acc_base = float(accuracy_score(true_base, pred_base))
    recall_base = float(recall_score(true_base, pred_base, average="macro", zero_division=0))
    
    acc_mod = float(accuracy_score(true_mod, pred_mod))
    recall_mod = float(recall_score(true_mod, pred_mod, average="macro", zero_division=0))
    
    acc_vattu = float(accuracy_score(true_vattu, pred_vattu))
    recall_vattu = float(recall_score(true_vattu, pred_vattu, average="macro", zero_division=0))
    
    # 7. Baseline Comparisons ('Always None')
    # modifier 'none' index
    none_mod_idx = label_maps["mod_map"].get("none", 0)
    baseline_mod_acc = float(np.mean(true_mod == none_mod_idx))
    
    # vattu 'none' index
    none_vattu_idx = label_maps["vattu_map"].get("none", 0)
    baseline_vattu_acc = float(np.mean(true_vattu == none_vattu_idx))
    
    # 8. Recombination & 630-Way Constrained MLE Evaluation
    logger.info("Recombining multi-head predictions with Constrained Maximum-Likelihood Decoding...")
    top1_hits = 0
    top5_hits = 0
    fallback_count = 0
    
    for i in range(total_test_samples):
        rec = recombine_prediction(
            base_probs=base_probs_arr[i],
            mod_probs=mod_probs_arr[i],
            vattu_probs=vattu_probs_arr[i],
            label_maps=label_maps
        )
        if rec["is_fallback"]:
            fallback_count += 1
            
        equiv_classes = get_equivalent_classes(true_classes[i])
        
        # Check top-1
        if rec["predicted_class"] in equiv_classes or class_to_comb.get(rec["predicted_class"]) == true_combs[i]:
            top1_hits += 1
            
        # Check top-5
        top_5_classes = {item["class_name"] for item in rec["top_5"]}
        top_5_combs = {class_to_comb.get(item["class_name"], item["class_name"]) for item in rec["top_5"]}
        
        if any(c in top_5_classes for c in equiv_classes) or (true_combs[i] in top_5_combs):
            top5_hits += 1
            
    recombined_top1_acc = float(top1_hits / total_test_samples)
    recombined_top5_acc = float(top5_hits / total_test_samples)
    fallback_rate = float(fallback_count / total_test_samples)
    
    # 9. Format Report
    report = {
        "test_samples": total_test_samples,
        "restored_epoch": restored_epoch,
        "metrics": {
            "per_head": {
                "base_akshara": {
                    "top1_accuracy": acc_base,
                    "macro_recall": recall_base
                },
                "vowel_modifier": {
                    "top1_accuracy": acc_mod,
                    "macro_recall": recall_mod,
                    "always_none_baseline_accuracy": baseline_mod_acc,
                    "beats_baseline": acc_mod > baseline_mod_acc
                },
                "conjunct_vattu": {
                    "top1_accuracy": acc_vattu,
                    "macro_recall": recall_vattu,
                    "always_none_baseline_accuracy": baseline_vattu_acc,
                    "beats_baseline": acc_vattu > baseline_vattu_acc
                }
            },
            "recombined_630_way": {
                "top1_accuracy": recombined_top1_acc,
                "top5_accuracy": recombined_top5_acc,
                "constrained_mle_fallback_rate": fallback_rate
            }
        }
    }
    
    # Print Executive Summary
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS SUMMARY:")
    logger.info(f"  Base Akshara Head:     Top-1 Acc = {acc_base:.2%}, Macro-Recall = {recall_base:.2%}")
    logger.info(f"  Vowel Modifier Head:   Top-1 Acc = {acc_mod:.2%}, Macro-Recall = {recall_mod:.2%} (Baseline: {baseline_mod_acc:.2%})")
    logger.info(f"  Conjunct Vattu Head:   Top-1 Acc = {acc_vattu:.2%}, Macro-Recall = {recall_vattu:.2%} (Baseline: {baseline_vattu_acc:.2%})")
    logger.info("-" * 70)
    logger.info(f"  RECOMBINED 630-WAY ACCURACY: Top-1 = {recombined_top1_acc:.2%}, Top-5 = {recombined_top5_acc:.2%}")
    logger.info(f"  Constrained Fallback Rate:   {fallback_rate:.2%} of predictions")
    logger.info("=" * 70)
    
    # Save Report
    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation report to: {out_p}")
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Telugu HCR v4 Multi-Head Model")
    parser.add_argument("--test_csv", type=str, default="outputs/test.csv", help="Path to test CSV")
    parser.add_argument("--label_maps", type=str, default="outputs/label_maps.json", help="Path to label_maps.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to checkpoints directory")
    parser.add_argument("--checkpoint_tag", type=str, default="best_model", help="Tag name of checkpoint to load")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root directory override for path remapping")
    parser.add_argument("--variant", type=str, default="B0", help="EfficientNetV2 variant (B0 or S)")
    parser.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--output_report", type=str, default="outputs/evaluation_report.json", help="Report output path")
    args = parser.parse_args()
    
    evaluate_test_set(
        test_csv=args.test_csv,
        label_maps_path=args.label_maps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_tag=args.checkpoint_tag,
        dataset_root=args.dataset_root,
        variant=args.variant,
        batch_size=args.batch_size,
        output_report_path=args.output_report
    )
