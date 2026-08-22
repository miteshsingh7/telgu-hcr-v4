"""Unit tests for dataset creation and CutMix probabilistic gating."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.data.dataset import create_telugu_dataset


def test_cutmix_probabilistic_gating():
    """Verifies that CutMix is applied with probability ~0.5 and not unconditionally."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a few dummy image files
        img_paths = []
        for i in range(16):
            img_file = tmp_path / f"img_{i}.png"
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(img_file)
            img_paths.append(str(img_file))
            
        df = pd.DataFrame({
            "file_path": img_paths,
            "class_name": [f"class_{i%4}" for i in range(16)],
            "base_idx": [i % 4 for i in range(16)],
            "modifier_idx": [0] * 16,
            "vattu_idx": [0] * 16
        })
        
        label_maps = {
            "num_base_classes": 4,
            "num_modifier_classes": 2,
            "num_vattu_classes": 2,
            "base_map": {f"b_{i}": i for i in range(4)},
            "mod_map": {"none": 0, "aa": 1},
            "vattu_map": {"none": 0, "k": 1}
        }
        
        label_smoothing = 0.1
        expected_unmixed_max = 1.0 - label_smoothing + (label_smoothing / 4.0)  # ~0.925
        
        ds, _, _ = create_telugu_dataset(
            csv_path_or_df=df,
            label_maps_or_path=label_maps,
            batch_size=4,
            is_training=True,
            use_augmentation=False,
            use_cutmix=True,
            cutmix_probability=0.5,
            label_smoothing=label_smoothing,
            shuffle_buffer=100
        )
        
        # Repeat to draw many batches
        ds_repeated = ds.repeat()
        
        unmixed_count = 0
        mixed_count = 0
        total_batches = 100
        
        for idx, (_, targets) in enumerate(ds_repeated.take(total_batches)):
            b_labels = targets["base_output"]
            max_probs = tf.reduce_max(b_labels, axis=-1).numpy()
            
            # An unmixed batch has all max probabilities equal to expected_unmixed_max
            is_unmixed = np.allclose(max_probs, expected_unmixed_max, atol=1e-3)
            if is_unmixed:
                unmixed_count += 1
            else:
                mixed_count += 1
                
        print(f"CutMix Gating Test: {unmixed_count} unmixed, {mixed_count} mixed out of {total_batches} batches.")
        
        # Confirm that some but not all batches are unmixed (roughly 50/50)
        assert unmixed_count > 10, f"Expected > 10 unmixed batches, got {unmixed_count}"
        assert mixed_count > 10, f"Expected > 10 mixed batches, got {mixed_count}"


if __name__ == "__main__":
    test_cutmix_probabilistic_gating()
    print("test_cutmix_probabilistic_gating passed successfully!")
