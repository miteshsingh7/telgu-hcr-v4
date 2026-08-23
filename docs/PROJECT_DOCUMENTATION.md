# Telugu Handwritten Character Recognizer (Telugu HCR v4)

Telugu is a Dravidian language with an abugida writing system where characters (aksharas) are formed through structural combinations of base consonants, secondary vowel signs (guninthamulu), and subscript conjunct consonants (vattulu). The Telugu dataset contains 630 character directories, which collapse to 596 structurally unique compound graphemes across 34 confirmed duplicate pairs between the base consonant and guninthamulu folders. Monolithic 630-way flat classification models struggle on this task because data is heavily fragmented across hundreds of compound classes, and visual features share sub-character components.

Telugu HCR v4 addresses this by decomposing every compound character into three linguistic components: a Base Akshara (52 classes), a Vowel Modifier (16 classes), and a Subscript Conjunct Vattu (37 classes). An EfficientNetV2-B0 backbone extracts unified visual representations, while three independent dense heads predict the component distributions simultaneously. A Constrained Maximum Likelihood Decoding engine recombines these distributions over the 596 valid compound combinations, achieving 94.72% Top-1 and 99.29% Top-5 recombined accuracy on the test set. An interactive Streamlit web application with stroke-level Gaussian blur preprocessing provides real-time handwriting recognition.


## Scenarios

Scenario 1: Interactive Digitization of Complex Compound Graphemes: A user draws a complex conjunct character such as 'ksha' (క్ష) or 'thri' (త్రి) on the interactive drawing canvas. The application crops and centers the glyph, applies Gaussian blur smoothing to bridge the domain shift between synthetic digital canvas strokes and paper ink dispersion, runs multi-head inference, and presents the predicted Telugu Unicode character along with decomposed component confidence scores.

Scenario 2: Historical Manuscript Archival and Document OCR: An archival institution scans physical historical Telugu documents and palm-leaf manuscripts. The batch evaluation pipeline reads cropped character regions, normalizes them through the single source of truth preprocessing module, and outputs structured predictions with Top-5 candidate rankings, enabling automated transcription and human-in-the-loop verification.

Scenario 3: Language Learning and Calligraphic Feedback: A student learning the Telugu script writes aksharas on a tablet device. The system evaluates stroke structure and provides immediate visual feedback, showing whether mistakes occurred in the base letter, the vowel matra, or the subscript vattu, assisting learners in mastering complex handwriting orthography.


## Technical Architecture:

*Figure: Telugu HCR v4 Multi-Head EfficientNetV2 Architecture with Constrained MLE Decoding*

The technical architecture consists of four interconnected subsystems: the data preprocessing pipeline, the multi-head neural network backbone, the constrained decoding engine, and the Streamlit user interface.

- Input Pipeline: Single Source of Truth Preprocessing: Decodes raw image bytes or arrays, extracts single-channel grayscale, applies aspect-ratio preserving padding to square with white background (255.0), resizes to 128x128 pixels using bilinear interpolation, replicates to 3 channels, and normalizes according to EfficientNetV2 scaling.
- Backbone Network: EfficientNetV2-B0 backbone pretrained on ImageNet (5.9M parameters), followed by Global Average Pooling, Batch Normalization, and Dropout (0.3).
- Multi-Task Classification Heads: Three parallel dense branches: Base Head (Dense 256, Dropout 0.3, Softmax 52), Modifier Head (Dense 128, Dropout 0.3, Softmax 16), and Vattu Head (Dense 128, Dropout 0.3, Softmax 37).
- Recombination Engine: Computes greedy argmax combination, evaluates validity against label maps, and applies Constrained Maximum Likelihood Decoding over 596 valid triples with log-likelihood optimization if the greedy combination is invalid.
- Interactive Frontend: Streamlit web application with a 2-column matte utility layout, stroke smoothing via Gaussian blur (sigma=3.0), and real-time Telugu Unicode glyph assembly.

## Pre-requisites:

- Python Environment: Python 3.10 or higher (tested with Python 3.10.13 via pyenv).
- Deep Learning Frameworks: TensorFlow 2.16+ and Keras 3.0+ for multi-head model architecture, AdamW optimizer, EMA weight tracking, and tf.data pipelines.
- Web Application: Streamlit 1.30+ and streamlit-drawable-canvas 0.9.0+ for the browser-based drawing interface and prediction display.
- Supporting Libraries: OpenCV (opencv-python), NumPy, Pandas, Scikit-Learn, Pillow, PyYAML, and Tqdm for image processing, metrics calculation, and dataset handling.
- Telugu Dataset: Directory Test1 containing Achulu, Hallulu, Guninthamulu, and Othulu folders across 630 class directories.
- Hardware: Apple Silicon Mac (MPS / CPU) or Linux / Windows workstation with NVIDIA CUDA GPU.

## Project Workflow:


## Milestone 1: Data Preparation and Architecture Definition

Milestone 1 establishes the foundational data architecture for Telugu HCR v4. This involves auditing the dataset structure, identifying semantic class duplicates, implementing a centralized single source of truth preprocessing pipeline, and creating the multi-head grapheme decomposition engine that maps compound Telugu characters into linguistic primitives.


### Activity 1.1: Dataset Structure and Stratified Splitting

Step 1: Inspect Dataset Hierarchy: The raw dataset consists of four main categories: Achulu (vowels, 16 folders), Hallulu (consonants, 36 folders), Guninthamulu (vowel-consonant combinations, 542 folders across sub-directories), and Othulu (subscript vattu markers, 36 folders), yielding 630 total class directories.

Step 2: Generate Stratified Splits: An 80/10/10 stratified split is created using a fixed random seed (42) to ensure reproducibility. Each class directory is sampled proportionally across train.csv, val.csv, and test.csv.

Step 3: Export Split Manifests and Label Maps: The script parses all image paths, generates numeric label indices for base akshara, vowel modifier, and conjunct vattu, and writes outputs/label_maps.json.

Command to generate stratified splits and label maps:

Source Code: src/data/split.py (Dataset Collection and Split Generation):


### Activity 1.2: Class Collision and Semantic Equivalence Audit

Step 1: Audit Confirmed Duplicate Class Pairs: Careful visual and linguistic inspection revealed that 34 class directories in Guninthamulu (specifically the 'none' modifier sub-folder for each consonant) contain the exact same base consonant glyph as the corresponding bare consonant folder in Hallulu (Pattern A duplicates). For example, 'Guninthamulu__kha__ka' and 'hallulu__ka' represent the identical character 'క' (ka).

Step 2: Canonical Equivalence Grouping: Rather than treating these 34 pairs as distinct competing classes (which causes artificial confusion), src/data/known_duplicates.py maintains an equivalence group lookup that collapses 630 class folders to 596 unique compound combinations during evaluation and recombination.

*Figure: Sample Class Collision Audit: Confirmed Identical Character Pair (hallulu__b vs Guninthamulu__ba__b)*

Source Code: src/data/known_duplicates.py (Duplicate Resolution Engine):


### Activity 1.3: Single Source of Truth Image Preprocessing

Step 1: Pad to Square with White Background: Handwritten character crops have varying aspect ratios. Direct non-uniform resizing distorts character strokes and aspect ratios. The pad_to_square function computes the maximum dimension and pads symmetrically using constant value 255.0 (white paper background).

Step 2: Grayscale Extraction and EfficientNetV2 Normalization: Multichannel inputs (RGB or RGBA) are converted to single-channel grayscale via standard luminance weights, resized to 128x128 using bilinear interpolation, replicated across 3 channels, and scaled using EfficientNetV2 preprocess_input.

*Figure: Single Source of Truth Image Preprocessing Pipeline*

Source Code: src/data/preprocessing.py (Complete Module):


### Activity 1.4: Multi-Head Decomposition Engine

Step 1: Grapheme Primitives Parsing: Every folder class name is parsed by decompose_class_name into (base_letter, vowel_modifier, conjunct_vattu). For example, 'Guninthamulu__ka__kii' decomposes into base consonant 'క' (ka), vowel modifier 'ii' (ఈకారం), and vattu 'none'.

Step 2: Constrained Maximum Likelihood Decoding: During inference, the three classification heads output probability vectors p_base, p_mod, and p_vattu. recombine_prediction performs a fast greedy argmax lookup. If the resulting combination is valid in the dataset, it is returned directly. If the greedy combination is invalid, it evaluates log p_base(b) + log p_mod(m) + log p_vattu(v) over all 596 valid combinations and selects the argmax, guaranteeing 100% valid Telugu grapheme outputs.

Source Code: src/data/decomposition.py (Constrained MLE Decoding):


## Milestone 2: Model Architecture and Loss Design

Milestone 2 details the multi-task neural network architecture, custom class-weighted loss formulation, and regularized data augmentation pipeline designed specifically for compound character recognition.


### Activity 2.1: Multi-Head EfficientNetV2 Backbone Construction

Step 1: EfficientNetV2 Feature Extraction: EfficientNetV2-B0 provides an optimal trade-off between model capacity (5.9M parameters), inference speed, and fine-grained feature representation. The network accepts (128, 128, 3) image tensors and produces a 1280-dimensional feature vector after Global Average Pooling.

Step 2: Independent Classification Branches: A Batch Normalization layer and Dropout (0.3) stabilize backbone representations. Three separate dense heads branch off: Base Head (256 units + ReLU + Dropout -> 52 Softmax), Modifier Head (128 units + ReLU + Dropout -> 16 Softmax), and Vattu Head (128 units + ReLU + Dropout -> 37 Softmax).

Source Code: src/models/multitask_effnetv2.py (Model Architecture Builder):


### Activity 2.2: Class-Weighted Loss Formulation with Label Smoothing

Step 1: Compute Frequency-Balanced Inverse Class Weights: Certain modifiers and vattus occur infrequently in Telugu text. To prevent dominant classes from overpowering gradient updates, compute_normalized_class_weights calculates inverse frequency weights clipped between 0.1 and 10.0 and normalized to mean 1.0.

Step 2: Implement Weighted Categorical Cross-Entropy: Keras standard sample_weight mechanisms can experience instability with multi-head outputs. WeightedCategoricalCrossentropy bakes per-class weight multipliers and label smoothing directly into the tensor loss graph, supporting continuous soft targets from CutMix.

Source Code: src/models/losses.py (Custom Loss Function):


### Activity 2.3: Data Augmentation and Multi-Head CutMix Pipeline

Step 1: Spatial Augmentations: Spatial transformations include RandomRotation (+/-5 degrees), RandomTranslation (+/-5% height and width), and RandomZoom (+/-5%). All boundary padding explicitly utilizes BACKGROUND_FILL_VALUE from preprocessing.py.

Step 2: Multi-Head CutMix Blending: CutMix cuts a rectangular patch from a secondary image and pastes it onto the primary image. The area ratio lambda adjusts the one-hot target vectors across all three classification heads simultaneously, improving generalization.

Source Code: src/data/augmentation.py (CutMix Implementation):


## Milestone 3: Training Pipeline and Checkpointing

Milestone 3 implements the unified two-phase training protocol, runtime extrapolation safety checks, and atomic state checkpointing to manage GPU training sessions effectively.


### Activity 3.1: Two-Phase Training Protocol

Step 1: Phase 1 Warmup (Epochs 1 to 5): The ImageNet pretrained backbone remains frozen. A linear learning rate schedule warms up from 0 to 1e-4, allowing the randomly initialized dense classification heads to converge without corrupting pretrained backbone weights.

Step 2: Phase 2 Fine-Tuning (Epochs 6 to 50): All backbone layers are unfrozen. Training proceeds using AdamW with weight decay 1e-4, global gradient clipping at 1.0, Exponential Moving Average (EMA momentum 0.999), and Cosine Decay learning rate scheduling down to 1e-6.

*Figure: Two-Phase Training Protocol and Runtime Session Budget Safety*

Command to launch model training:

Configuration File: configs/multitask_effnetv2.yaml:


### Activity 3.2: Early Timing Extrapolation and Kaggle Session Guard

Step 1: Measure Epoch Duration: Cloud training environments like Kaggle and Google Colab enforce strict 12-hour session limits. TimingExtrapolationCallback tracks exact epoch durations during early training.

Step 2: Extrapolate Projected Session Duration: At the completion of Epoch 2, the callback calculates the rolling average seconds per epoch and projects the total runtime for remaining epochs. If the projected duration exceeds 11.5 hours, it raises an alert recommending batch size or epoch adjustments.

Source Code: src/train.py (Timing Extrapolation Callback):


### Activity 3.3: Atomic Full-State Checkpoint Management

Step 1: Multi-Asset State Serialization: Standard Keras model checkpoints save model weights only, discarding optimizer momentum, EMA variables, and iteration counts. FullStateCheckpointManager saves model weights (.weights.h5), optimizer state (.optimizer.pkl), EMA weights, and metadata JSON atomically.

Step 2: Safe Restoration: When resuming after session interruptions or during evaluation, restore_state loads the full training state, synchronizes iterations, and evaluates the best validation checkpoint.

Source Code: src/checkpointing.py (Full State Checkpoint Manager):


## Milestone 4: Comprehensive Model Evaluation and Error Analysis

Milestone 4 presents the quantitative evaluation of the multi-head EfficientNetV2 model on the held-out test split, comparing per-head performance, recombined 630-way accuracy, baseline benchmarks, and confused character pairs.


### Activity 4.1: Multi-Head and Recombined Test Set Evaluation

Step 1: Per-Head Metrics: On the held-out test set, the three classification heads achieved: Base Akshara Top-1 Accuracy: 96.57%, Vowel Modifier Top-1 Accuracy: 96.60%, and Conjunct Vattu Top-1 Accuracy: 99.34%.

Step 2: Recombined 630-Way Constrained Evaluation: Passing the three predicted probability distributions through the Constrained Maximum Likelihood Decoding engine yielded a Recombined Top-1 Accuracy of 94.72% and a Recombined Top-5 Accuracy of 99.29% across all 630 dataset class directories.

*Figure: Multi-Head and Recombined Test Set Accuracy Summary*

Command to run evaluation:

Source Code: src/evaluate.py (Evaluation Engine):


### Activity 4.2: Baseline Comparison and Confusion Analysis

Step 1: Baseline Comparisons: The multi-head EfficientNetV2 model substantially outperforms traditional single-head models. A monolithic MobileNetV2 baseline trained directly on 630 flat classes achieved 85.64% Top-1 accuracy, while an Always-None majority predictor achieves 0.0% accuracy on non-trivial classes.

Step 2: Confusion Analysis: The most common remaining prediction errors occur between characters with high visual overlap, such as 'th' (థ) vs 'tha' (థ), 'dh' (ధ) vs 'dha' (ధ), and subtle diacritic loops in vowel signs.


## Milestone 5: Streamlit Interactive Web Application

Milestone 5 covers the development of the interactive Streamlit user interface, the canvas stroke smoothing algorithm that resolves the digital-to-physical domain shift, and real-time Telugu Unicode glyph assembly.


### Activity 5.1: Two-Column Matte Technical Utility Interface

Step 1: Layout and Theme Styling: The interface uses a clean two-column layout constrained to 1100px max width. A warm matte palette (#fbf9f4 background, #edebe6 surface, #a03d00 terracotta accent) and Geist typography provide a distraction-free environment.

Step 2: Responsive Columns: The Left Column houses the drawing canvas (280x280), stroke width controls (4px to 24px), clear canvas button, and recognize button. The Right Column displays the predicted Telugu Unicode character, top-3 ranked alternatives, confidence bars, and linguistic breakdown badges.

*Figure: Streamlit Side-by-Side Matte Technical Utility Interface Layout*


### Activity 5.2: Canvas Preprocessing and Domain Shift Resolution

Step 1: Canvas Gaussian Blur Smoothing: Digital HTML5 canvas drawings produce sharp aliased binary edges with a Laplacian variance of ~5545, whereas real handwritten paper scans have ink dispersion and soft edges with a Laplacian variance of ~76.7. Applying cv2.GaussianBlur(..., (0, 0), sigmaX=3.0) smooths stroke edges and closes the distribution gap.

Step 2: Telugu Unicode Synthesis: get_display_glyph synthesizes authentic multi-part Telugu graphemes by combining the base character, virama marker (్), vattu consonant glyph, and vowel matra unicode string.

*Figure: Canvas Stroke Smoothing via Gaussian Blur for Domain Shift Resolution*


### Activity 5.3: Top-3 Prediction Cards and Detailed Probability Breakdown

Step 1: Primary Match Card: Displays the top predicted character in large Telugu font (44px), accompanied by the confidence percentage bar and folder class identifier.

Step 2: Component Breakdown Badges: Below the primary prediction, three badges display the individual confidence scores for Base Akshara, Vowel Modifier, and Conjunct Vattu.

Source Code: app.py (Streamlit Application Engine):


## Milestone 6: Deployment and Verification

Milestone 6 provides complete instructions for local application deployment, verification across drawing and image upload modes, and public deployment using Ngrok tunneling.


### Activity 6.1: Preparing the Application for Local Deployment

Step 1: Set Up Python Virtual Environment:

Step 2: Install Required Dependencies:

Complete requirements.txt listing:


### Activity 6.2: Local Testing and Verification

Step 1: Start the Streamlit Application Server:

Step 2: Verify Recognition Pipeline: Open http://localhost:8501 in a web browser. Test character recognition using both the interactive canvas drawing tool and uploaded test image files.


### Activity 6.3: Public Deployment via Ngrok

Step 1: Install and Configure Ngrok: Ngrok creates a secure HTTPS tunnel to the local Streamlit port, allowing public access without server configuration.

Step 2: Run the Public Deployment Script (run_public.py):

Source Code: run_public.py (Public Tunnel Deployment):


### Important Notes:

- Hardware Acceleration: Apple Silicon Acceleration: On macOS with Apple Silicon, TensorFlow utilizes Metal Performance Shaders (MPS). In automated test environments, CPU execution is enforced to prevent GPU kernel locks during rapid evaluation cycles.
- Drawing Technique: Canvas Stroke Thickness: For optimal recognition accuracy, stroke width should be set between 8px and 16px, matching the pen line thickness of standard dataset images.
- Tunnel Persistence: Ngrok Tunnel Expiration: On free Ngrok accounts, public tunnel URLs expire after session termination. Restarting run_public.py assigns a new active URL.

## Exploring the Web Application:


### Home / Recognition Page

The main application page features a top navigation bar with the brand title 'Telugu Akshara Recognizer', model status indicator (Online 94.7%), and active parameter badge (EfficientNetV2-B0 5.9M). The page is organized in a balanced two-column utility layout.


### Input Canvas and Drawing Controls

The left column provides an HTML5 drawing canvas with real-time mouse and stylus tracking. Users can select stroke widths from 4px to 24px, switch between drawing mode and image upload mode, and trigger recognition via the 'Recognize Character' action button.


### Recognition Results and Component Breakdown

The right column presents the recognition results. The primary prediction card highlights the recognized character in large Telugu typography, accompanied by the recombined confidence percentage. Three component badges display the breakdown for Base Akshara, Vowel Modifier, and Conjunct Vattu. Below the primary card, alternative Top-2 and Top-3 candidate cards provide secondary predictions with confidence bars.


## Conclusion

Telugu HCR v4 resolves the challenge of handwritten Telugu compound character recognition through linguistic decomposition and multi-task deep learning. By decomposing 630 character classes into 52 base aksharas, 16 vowel modifiers, and 37 conjunct vattus, the EfficientNetV2-B0 model achieves 94.72% Top-1 and 99.29% Top-5 recombined accuracy. The single source of truth preprocessing pipeline and Gaussian blur stroke smoothing effectively bridge the domain gap between digital canvas drawings and physical paper scans. Future work includes extending the pipeline to writer-independent grouping splits, sentence-level connected text recognition, and edge deployment via TensorFlow Lite.
