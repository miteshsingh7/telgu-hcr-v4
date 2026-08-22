"""Telugu Handwritten Character Recognizer — Streamlit Inference App.

Imports preprocessing directly from src.data.preprocessing (single source of truth).
Uses Constrained Maximum-Likelihood Decoding to display character breakdowns and top predictions.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Configure root path relative to this script
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.data.preprocessing import preprocess_image, IMAGE_SIZE
from src.data.decomposition import recombine_prediction
from src.models.multitask_effnetv2 import build_multitask_effnetv2, parse_model_prediction_outputs
from src.checkpointing import FullStateCheckpointManager

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


st.set_page_config(
    page_title="Telugu Handwritten Character Recognizer",
    page_icon="✍️",
    layout="wide"
)


@st.cache_resource(show_spinner="Loading Telugu HCR Model...")
def load_system_assets(checkpoint_dir: str = "checkpoints",
                       label_maps_path: str = "outputs/label_maps.json") -> Tuple[Optional[tf.keras.Model], Dict[str, Any]]:
    """Loads label maps and restores the best trained model checkpoint."""
    lmaps_p = APP_DIR / label_maps_path
    if not lmaps_p.exists():
        st.error(f"Label maps not found at: {lmaps_p}. Run split.py first.")
        st.stop()
        
    with open(lmaps_p, "r", encoding="utf-8") as f:
        label_maps = json.load(f)
        
    num_base = label_maps["num_base_classes"]
    num_mod = label_maps["num_modifier_classes"]
    num_vattu = label_maps["num_vattu_classes"]
    
    model = build_multitask_effnetv2(
        variant="B0",
        num_base=num_base,
        num_mod=num_mod,
        num_vattu=num_vattu,
        weights=None,
        backbone_trainable=False
    )
    
    ckpt_dir = APP_DIR / checkpoint_dir
    ckpt_manager = FullStateCheckpointManager(checkpoint_dir=ckpt_dir)
    
    try:
        dummy_opt = tf.keras.optimizers.AdamW()
        restored_epoch, meta = ckpt_manager.restore_state(model, dummy_opt, checkpoint_path_or_tag="best_model")
        st.sidebar.success(f"Loaded model checkpoint from epoch {restored_epoch} (val_loss: {meta.get('monitored_value', 'N/A')})")
    except Exception as e:
        st.sidebar.warning(f"No trained checkpoint found ({e}). Running in demo / uninitialized weights mode.")
        
    return model, label_maps


def run_inference(image_input: Any, 
                  model: tf.keras.Model, 
                  label_maps: Dict[str, Any]) -> Dict[str, Any]:
    """Runs preprocessing, forward pass, and constrained maximum-likelihood recombination."""
    # 1. Single Source of Truth Preprocessing
    preprocessed_tensor = preprocess_image(image_input, img_size=IMAGE_SIZE)
    input_batch = tf.expand_dims(preprocessed_tensor, axis=0)
    
    # 2. Forward pass
    raw_preds = model(input_batch, training=False)
    b_probs, m_probs, v_probs = parse_model_prediction_outputs(raw_preds)
    
    # 3. Constrained Recombination
    rec_result = recombine_prediction(
        base_probs=b_probs[0],
        mod_probs=m_probs[0],
        vattu_probs=v_probs[0],
        label_maps=label_maps
    )
    
    return rec_result, preprocessed_tensor.numpy()


def main():
    st.title("✍️ Telugu Handwritten Character Recognizer (v4)")
    st.markdown(
        "Powered by **Multi-Head EfficientNetV2** with independent structural decomposition "
        "(Base Akshara + Vowel Modifier + Conjunct Vattu) and Constrained Maximum-Likelihood Decoding."
    )
    
    model, label_maps = load_system_assets()
    
    col_input, col_results = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Input Character")
        input_mode = st.radio("Choose Input Method", ["Draw on Canvas", "Upload Image"], horizontal=True)
        
        image_to_process = None
        
        if input_mode == "Draw on Canvas":
            if CANVAS_AVAILABLE:
                st.write("Draw a Telugu character below (black strokes on white background):")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=12,
                    stroke_color="#000000",
                    background_color="#FFFFFF",
                    width=280,
                    height=280,
                    drawing_mode="freedraw",
                    key="telugu_canvas",
                )
                if canvas_result.image_data is not None:
                    # Check if user drew anything
                    img_array = canvas_result.image_data.astype(np.uint8)
                    # Look for non-white pixels
                    if np.mean(img_array[..., :3]) < 254.0:
                        image_to_process = img_array
            else:
                st.warning("streamlit-drawable-canvas not installed. Please use the Upload Image tab.")
                
        else:
            uploaded_file = st.file_uploader("Upload a scanned Telugu character image", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)
                st.image(image_to_process, caption="Uploaded Image", width=200)
                
    with col_results:
        st.subheader("Recognition Results")
        
        if image_to_process is not None and model is not None:
            with st.spinner("Analyzing character..."):
                rec_result, preprocessed_img = run_inference(image_to_process, model, label_maps)
                
            pred_class = rec_result["predicted_class"]
            confidence = rec_result["confidence"]
            base_let = rec_result["base_letter"]
            vow_mod = rec_result["vowel_modifier"]
            vattu = rec_result["vattu"]
            is_fallback = rec_result["is_fallback"]
            
            # Primary Card
            st.success(f"### Predicted Class: `{pred_class}`")
            
            metric_cols = st.columns(4)
            metric_cols[0].metric("Base Akshara", base_let)
            metric_cols[1].metric("Vowel Modifier", vow_mod)
            metric_cols[2].metric("Conjunct Vattu", vattu)
            metric_cols[3].metric("Confidence", f"{confidence:.1%}")
            
            if is_fallback:
                st.info("ℹ️ Constrained Maximum-Likelihood decoded from joint primitive likelihoods.")
                
            st.divider()
            
            # Top-5 Rankings
            st.subheader("Top-5 Joint Candidate Predictions")
            for rank, item in enumerate(rec_result["top_5"], 1):
                prob = item["probability"]
                cname = item["class_name"]
                b = item["base_letter"]
                m = item["vowel_modifier"]
                v = item["vattu"]
                
                st.write(f"**#{rank}** `{cname}` — **{prob:.2%}** (Base: `{b}`, Mod: `{m}`, Vattu: `{v}`)")
                st.progress(min(1.0, max(0.0, prob)))
                
            # Preprocessing Preview
            with st.expander("Show Normalized Model Input (128x128 3-ch)"):
                # Denormalize for display
                disp_img = (preprocessed_img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                st.image(disp_img, caption="Square-Padded, Bilinear-Resized 128x128 Input", width=128)
                
        else:
            st.info("Draw a Telugu character or upload an image on the left panel to see real-time recognition.")
            
    st.divider()
    with st.expander("📚 Telugu Akshara Structure Reference"):
        st.markdown("""
        **How Telugu Compound Characters are structured:**
        - **Achulu (Vowels)**: 16 standalone vowel glyphs (అ, ఆ, ఇ, ఈ, ఉ, ఊ, ఋ, ౠ, ఎ, ఏ, ఐ, ఒ, ఓ, ఔ, అం, అః)
        - **Hallulu (Consonants)**: 36 base consonant glyphs (క to క్ష)
        - **Guninthamulu (Consonant + Vowel sign)**: Consonant base + one of 16 vowel modifiers (తలకట్టు, దీర్ఘం, గుడి, కొమ్ము, ఎత్వం, etc.)
        - **Othulu (Conjuncts / Vattulu)**: Subscript consonant attached below a base character.
        """)


if __name__ == "__main__":
    main()
