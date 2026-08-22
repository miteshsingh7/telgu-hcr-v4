# Telugu Handwritten Character Recognizer (v4)

A ground-up rebuild of the Telugu Handwritten Character Recognition system using a **shared-backbone Multi-Head EfficientNetV2 architecture with Constrained Maximum-Likelihood Decoding**.

---

## 1. Architectural Overview

### Structural Decomposition
Telugu compound characters are naturally composed of primitives:
- **Base Akshara**: 47 distinct standalone vowel and consonant glyphs (అ–అః, క–క్ష)
- **Vowel Modifier**: 16 canonical matras (తలకట్టు, దీర్ఘం, గుడి, కొమ్ము, ఎత్వం, etc.)
- **Conjunct Vattu**: 32 subscript consonants (వత్తులు)

Instead of a flat 630-way combinatorial softmax that treats every combination as an isolated class, the model predicts all three primitives independently and recombines them via **Constrained Maximum-Likelihood Decoding**:

$$\hat{y} = \arg\max_{(b, m, v) \in \mathcal{V}} \left( \log p_B[b] + \log p_M[m] + \log p_V[v] \right)$$

where $\mathcal{V}$ is the grammar of valid triples observed in the dataset.

```
                  ┌────────────────────────┐
                  │ 128x128 3-Ch Character │
                  └───────────┬────────────┘
                              │
               ┌──────────────▼──────────────┐
               │   EfficientNetV2 Backbone   │
               │   (ImageNet Pretrained)     │
               └──────────────┬──────────────┘
                              │ GAP + BN + Dropout(0.3)
               ┌──────────────┼──────────────┐
               │              │              │
        ┌──────▼──────┐┌──────▼──────┐┌──────▼──────┐
        │  Base Head  ││Modifier Head││  Vattu Head │
        │  Dense(256) ││  Dense(128) ││  Dense(128) │
        │  Softmax(47)││  Softmax(16)││  Softmax(32)│
        └──────┬──────┘└──────┬──────┘└──────┬──────┘
               │              │              │
               └──────────────┼──────────────┘
                              │
               ┌──────────────▼──────────────┐
               │   Constrained MLE Decoder   │
               │  (Recombined 630-Way Class) │
               └─────────────────────────────┘
```

---

## 2. Key Engineering Guarantees

1. **Single Source of Truth Preprocessing & Augmentation**:
   - `src/data/preprocessing.py`: The ONLY place image decoding, square-padding, bilinear resizing, and normalization occur. Exports `BACKGROUND_FILL_VALUE`.
   - `src/data/augmentation.py`: The ONLY place rotation ($\pm 5^\circ$), translation ($\pm 5\%$), and zoom ($\pm 5\%$) layers and CutMix are defined.
2. **Multi-Head CutMix with Normalized Loss Weighting**:
   - Batch CutMix simultaneously blends image patches and all 3 multi-head label vectors with identical $\lambda$.
   - Normalized per-head class frequency weights ($w_h(c)$) are baked directly into a custom cross-entropy loss, resolving class imbalance without Keras `class_weight` runtime clashes on soft targets.
3. **Safe Full-State Checkpointing**:
   - `src/checkpointing.py` saves model weights, optimizer state (including iterations counter for cosine decay), and epoch counters atomically. Checkpoint files are never mutated in place on load failure.
4. **Early Runtime Extrapolation**:
   - Training loop times early epochs and extrapolates total runtime to ensure safe execution within Kaggle's 12-hour session limits.

---

## 3. Quickstart Guide

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 1. Generate Frozen Dataset Splits
```bash
python src/data/split.py --data_dir "path/to/Test1" --output_dir "outputs"
```

### 2. Run Test Suite
```bash
python tests/test_preprocessing.py
python tests/test_decomposition.py
python tests/test_model.py
python tests/test_checkpointing.py
```

### 3. Train Multi-Head Model
```bash
# Two-phase training: Phase 1 frozen warmup (5 epochs) + Phase 2 fine-tuning (45 epochs)
python src/train.py --config configs/multitask_effnetv2.yaml --variant B0

# To resume interrupted training from latest checkpoint:
python src/train.py --resume
```

### 4. Evaluate Test Set
```bash
python src/evaluate.py --checkpoint_dir checkpoints --checkpoint_tag best_model
```

### 5. Launch Streamlit Web App
```bash
streamlit run app.py
```

---

## 4. Repository Structure

```
telgu-hcr-v4/
├── configs/
│   └── multitask_effnetv2.yaml        # Single training/model configuration
├── outputs/
│   ├── label_maps.json                # Derived vocabulary & reverse index
│   ├── train.csv / val.csv / test.csv # Frozen stratified splits
│   └── evaluation_report.json         # Evaluation metrics & summary
├── src/
│   ├── data/
│   │   ├── preprocessing.py           # Single preprocessing function & BACKGROUND_FILL_VALUE
│   │   ├── decomposition.py           # Akshara decomposition & constrained recombination
│   │   ├── split.py                   # Frozen split generator
│   │   ├── augmentation.py            # Augmentation pipeline & multi-head CutMix
│   │   └── dataset.py                 # tf.data input pipeline
│   ├── models/
│   │   ├── multitask_effnetv2.py      # Multi-head EfficientNetV2 model
│   │   └── losses.py                  # Weighted categorical cross-entropy
│   ├── checkpointing.py               # Full-state save and restore manager
│   ├── train.py                       # Training loop with timing extrapolation
│   └── evaluate.py                    # Multi-head & recombined evaluation
├── tests/
│   ├── test_preprocessing.py          # Preprocessing & fill value tests
│   ├── test_decomposition.py          # Full dataset decomposition audit
│   ├── test_model.py                  # One-batch overfit with mixed_float16 + EMA
│   └── test_checkpointing.py          # Checkpoint kill-and-resume tests
├── app.py                             # Interactive Streamlit inference app
├── notebooks/
│   └── kaggle_train.ipynb             # Standalone Kaggle execution notebook
├── pyproject.toml / requirements.txt  # Dependencies
└── README.md                          # Documentation
```
