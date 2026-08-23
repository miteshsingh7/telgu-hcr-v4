# Telugu Handwritten Character Recognizer (HCR) v4

A multi-task deep learning framework for recognizing **630 Telugu handwritten character classes** (Achulu, Hallulu, Guninthamulu, and Othulu) using a shared **EfficientNetV2** backbone and **Constrained Maximum-Likelihood Decoding**.

---

## 📌 Architecture Overview

Instead of a flat 630-way multi-class classification problem, this framework decomposes compound Telugu graphemes into three orthogonal linguistic primitives:

```
                          Input Image (128x128x3, [0, 255])
                                         │
                          ┌──────────────┴──────────────┐
                          │     EfficientNetV2-B0/S     │
                          │   (include_preprocessing)   │
                          └──────────────┬──────────────┘
                                         │ Global Average Pooling + BN + Dropout(0.3)
                          ┌──────────────┼──────────────┐
                          │ (Dense 256)  │ (Dense 128)  │ (Dense 128)
                          ▼              ▼              ▼
                    Base Akshara      Modifier        Vattu
                        Head            Head           Head
                    (52 classes)    (16 classes)   (37 classes)
                     [Softmax]       [Softmax]      [Softmax]
```

* **Base Akshara Head (52 classes):** Root consonant or independent vowel (e.g., క, ఖ, గ, అ, ఆ).
* **Vowel Modifier / Matra Head (16 classes):** Diacritic vowel attachment or standalone indicator (`none`, `aa` ా, `i` ి, `u` ు, `ee` ే, etc.).
* **Conjunct Subscript / Vattu Head (37 classes):** Subscript conjunct consonant or none (`none`, `k` ్క, `g` ్గ, `r` ్ర, `y` ్య, etc.).

Predictions are recombined into 630-way character classes using **Constrained Maximum-Likelihood Decoding**, which projects the multi-head joint probability distribution $P(\text{base}, \text{modifier}, \text{vattu}) = P(\text{base}) \cdot P(\text{modifier}) \cdot P(\text{vattu})$ strictly over the **596 unique valid compound combinations** defined in `outputs/label_maps.json`.

---

## 📈 Benchmarks & Results

Trained on 50,000+ handwritten Telugu character samples across 630 classes using Phase 1 warmup (5 epochs) and Phase 2 fine-tuning (45 epochs) with AdamW, Exponential Moving Average (EMA), Cosine Decay, and multi-head CutMix:

| Metric | Prior MobileNetV2 Baseline | Telugu HCR v4 (EfficientNetV2-B0) | Improvement |
| :--- | :---: | :---: | :---: |
| **Recombined 630-Way Top-1 Accuracy** | 85.64% | **94.72%** | **+9.08%** |
| **Recombined 630-Way Top-5 Accuracy** | 97.11% | **99.29%** | **+2.18%** |
| **Base Akshara Head Top-1 Accuracy** | ~89.20% | **96.57%** | +7.37% |
| **Vowel Modifier Head Top-1 Accuracy** | ~88.70% | **96.60%** | +7.90% |
| **Conjunct Vattu Head Top-1 Accuracy** | ~95.10% | **99.34%** | +4.24% |
| **Constrained MLE Fallback Rate** | N/A | **0.78%** | — |

*Per-head baselines*: The vowel modifier head achieves **96.60%** vs. the trivial "always none" baseline of 36.8%, and the conjunct vattu head achieves **99.34%** vs. the trivial baseline of 94.1%.

---

## 🏛️ Design Notes & Known Gotchas

### 1. Label Space & Duplicate Class Equivalence (`src/data/known_duplicates.py`)
* The raw dataset contains 630 class directories, but the actual structural combination space is **596 unique valid triples**.
* **34 pairs of flat class folders are true visual duplicates**, created by overlapping folder structures in the source dataset:
  * For example, the bare consonant folder in `hallulu/ka` and the unmodulated consonant folder in `Guninthamulu/kha/ka` both represent the glyph **క** (*ka*).
  * Similar identical pairs exist for **ఖ** (`Guninthamulu__khh__kha` / `hallulu__kha`), **గ** (`Guninthamulu__ga__ga` / `hallulu__g`), **ప** (`Guninthamulu__pa__p` / `hallulu__P`), **హ** (`Guninthamulu__ha__h` / `hallulu__h`), and **క్ష** (`Guninthamulu__ksh__ksh` / `hallulu__ks`).
* These 34 duplicate pairs were verified by direct visual and pixel-level comparison during development. During evaluation and scoring, predictions matching any canonical alias in the equivalence group are treated as correct via `get_equivalent_classes()`.

### 2. Internal Normalization Coupling (`include_preprocessing=True`)
* `src/models/multitask_effnetv2.py` instantiates EfficientNetV2 backbones with `include_preprocessing=True` (Keras standard default).
* Consequently, `src/data/preprocessing.py` intentionally outputs tensors in the raw `[0.0, 255.0]` float32 range and does **not** rescale to `[-1, 1]` or `[0, 1]`. The backbone's internal `Rescaling` and `Normalization` layers standardize the input.
* This coupling is explicitly pinned and guarded by unit tests:
  * `test_include_preprocessing_default_is_true` in `tests/test_model.py` validates that the backbone constructor parameters retain `include_preprocessing=True`.
  * `test_model_backbone_contains_internal_normalization` in `tests/test_model.py` inspects the instantiated model graph to ensure internal normalization layers are active.

### 3. Calibrated Gaussian Blur for Canvas Input in `app.py`
* Photographed paper-and-ink images in the training dataset exhibit natural pen spread and sensor anti-aliasing, with Laplacian edge sharpness variance ranging between **`9.38` and `256.21`** (mean ~`61.05`).
* In contrast, browser `<canvas>` vector rasterization produces razor-sharp binary edges with Laplacian variance exceeding **`5,545.0`** (~90× above training data).
* Without adjustment, this severe domain gap causes canvas-drawn character predictions to degrade to near-random, even though uploaded photographic images achieve >94% accuracy.
* In `app.py`, a calibrated Gaussian blur (`cv2.GaussianBlur(..., sigmaX=3.0)`) is applied to canvas strokes before preprocessing, bringing edge variance into the **`50.0–76.7`** range and matching the dataset distribution.

### 4. Gitignored Checkpoints & Weights Delivery
* The `checkpoints/` directory is gitignored to avoid checking multi-hundred-MB binary files into Git history.
* A fresh clone will not contain pre-trained weights. To run `app.py` or `src/evaluate.py`, copy the checkpoint files (`model.weights.h5`, `optimizer_state.pkl`, and `state.json`) into `checkpoints/best_model/` from your training environment (e.g., Kaggle working outputs or cloud storage).

---

## 🛠️ Setup & Requirements

```bash
# Clone the repository
git clone https://github.com/miteshsingh7/telgu-hcr-v4.git
cd telgu-hcr-v4

# Install dependencies
pip install -e .
```

**Environment Requirements:**
* Python >= 3.10
* TensorFlow >= 2.16.0
* Keras >= 3.0.0
* Streamlit >= 1.30.0
* streamlit-drawable-canvas >= 0.9.3

---

## 📁 Repository Structure

```
telgu-hcr-v4/
├── app.py                          # Streamlit 2-column interactive UI
├── configs/
│   └── multitask_effnetv2.yaml     # Model & training hyperparameters
├── outputs/
│   ├── label_maps.json             # Empirical vocabulary & 596 valid triples
│   ├── train.csv                   # Stratified training split
│   ├── val.csv                     # Stratified validation split
│   └── test.csv                    # Stratified test split
├── scripts/
│   ├── compare_ink_density.py      # Ink density diagnostic tool
│   ├── compare_stroke_texture.py   # Laplacian edge sharpness analysis
│   └── compress_images.py          # Image resizing utility
├── src/
│   ├── checkpointing.py            # Atomic full-state checkpoint manager
│   ├── evaluate.py                 # Multi-head and 630-way evaluation
│   ├── train.py                    # Two-phase training pipeline
│   ├── data/
│   │   ├── augmentation.py         # Multi-head CutMix & spatial augmentations
│   │   ├── dataset.py              # High-throughput tf.data pipeline
│   │   ├── decomposition.py        # Grapheme decomposition & decoding
│   │   ├── known_duplicates.py     # 34 verified duplicate class equivalence maps
│   │   ├── preprocessing.py        # Single source of truth preprocessing
│   │   └── split.py                # Stratified split generator
│   └── models/
│       ├── losses.py               # WeightedCategoricalCrossentropy
│       └── multitask_effnetv2.py   # Multi-head EfficientNetV2 architecture
└── tests/
    ├── test_checkpointing.py       # Checkpoint save/restore tests
    ├── test_dataset.py             # Dataset pipeline & CutMix tests
    ├── test_decomposition.py       # Decomposition audit & recombination tests
    ├── test_model.py               # Overfit & normalization coupling tests
    └── test_preprocessing.py       # Preprocessing tensor format tests
```

---

## 🚀 How to Run

### 1. Generate Stratified Splits
Decomposes folder names into base, modifier, and vattu indices and generates reproducible splits:
```bash
python src/data/split.py --data_dir "data/Final Dataset of Telugu Handwritten Chararcters/Test1" --output_dir outputs
```

### 2. Model Training
Runs two-phase training (Phase 1 frozen backbone warmup + Phase 2 unfrozen fine-tuning with EMA and AdamW):
```bash
python src/train.py --config configs/multitask_effnetv2.yaml
```

### 3. Evaluation
Evaluates the checkpoint on the test split:
```bash
python src/evaluate.py --checkpoint_dir checkpoints --checkpoint_tag best_model
```

### 4. Interactive Streamlit App
Launches the side-by-side technical UI:
```bash
streamlit run app.py
```
