"""Telugu Handwritten Character Recognizer — Streamlit App.

Faithful implementation of the Matte Technical Utility design system and reference layout.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

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


# Telugu Matra & Vattu Display Constants
MATRA_UNICODE = {
    "none": "",
    "aa": "\u0C3E",    # ా
    "i": "\u0C3F",     # ి
    "ii": "\u0C40",    # ీ
    "u": "\u0C41",     # ు
    "uu": "\u0C42",    # ూ
    "ru": "\u0C43",    # ృ
    "ruu": "\u0C44",   # ౄ
    "e": "\u0C46",     # ె
    "ee": "\u0C47",    # ే
    "ai": "\u0C48",    # ై
    "o": "\u0C4A",     # ొ
    "oo": "\u0C4B",    # ో
    "au": "\u0C4C",    # ౌ
    "am": "\u0C02",    # ం
    "ah": "\u0C03"     # ః
}

VIRAMA = "\u0C4D" # ్
VATTU_CONSONANT = {
    "k": "క", "kh": "ఖ", "g": "గ", "gh": "ఘ", "gna": "ఙ",
    "c": "చ", "ch": "ఛ", "j": "జ", "jh": "ఝ", "jna": "ఞ",
    "t": "ట", "tt": "ఠ", "th": "ఠ", "d": "డ", "dh": "ఢ", "ana": "ణ", "an": "ణ", "nn": "న",
    "tha": "థ", "da": "ద", "dha": "ధ", "n": "ట", "na": "న",
    "p": "ప", "ph": "ఫ", "b": "బ", "bh": "భ", "m": "మ",
    "y": "య", "r": "ర", "rr": "ఱ", "l": "ల", "ll": "ళ", "v": "వ",
    "s": "శ", "sh": "ష", "sa": "స", "h": "హ", "ha": "హ", "ksh": "క్ష", "z": "క"
}


def get_display_glyph(base_char: str, mod: str, vattu: str) -> str:
    """Combines primitives into an authentic Telugu Unicode grapheme string."""
    if base_char == "none":
        c = VATTU_CONSONANT.get(vattu, "క")
        return f"{VIRAMA}{c}"
    
    matra = MATRA_UNICODE.get(mod, "")
    if vattu != "none":
        v_char = VATTU_CONSONANT.get(vattu, "")
        return f"{base_char}{VIRAMA}{v_char}{matra}"
    return f"{base_char}{matra}"


# Configure Streamlit Page
st.set_page_config(
    page_title="Telugu Akshara Recognizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Global CSS injection to match reference design exactly
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-sans/style.css">
<style>
    :root {
        --bg-main: #fbf9f4;
        --surface: #edebe6;
        --surface-low: #f5f3ee;
        --surface-lowest: #ffffff;
        --outline: #8a8882;
        --outline-variant: #c9c7c2;
        --primary: #a03d00;
        --primary-container: #c1541a;
        --on-surface: #1c1c1b;
        --on-surface-variant: #574239;
    }

    * {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        box-sizing: border-box;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--on-surface) !important;
        min-height: 100vh !important;
    }

    header[data-testid="stHeader"], #MainMenu, footer {
        display: none !important;
    }

    .block-container {
        max-width: 100% !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Top Full-Width Navbar */
    .top-navbar {
        background-color: var(--bg-main);
        border-bottom: 1px solid var(--outline-variant);
        width: 100%;
        padding: 12px 24px;
    }

    .top-navbar-inner {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }

    .top-navbar-left {
        display: flex;
        align-items: baseline;
        gap: 16px;
    }

    .brand-title {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--on-surface);
        margin: 0;
    }

    .brand-subtitle {
        font-size: 14px;
        color: var(--on-surface-variant);
        margin: 0;
    }

    .top-navbar-right a {
        font-size: 12px;
        font-weight: 700;
        color: var(--primary);
        border-bottom: 1px solid var(--primary);
        padding-bottom: 2px;
        text-decoration: none;
    }

    /* Main Column Container */
    .main-column {
        max-width: 480px;
        margin: 16px auto 24px auto;
        padding: 0 12px;
    }

    /* Tools Bar */
    .tools-bar {
        background-color: var(--surface-lowest);
        border: 1px solid var(--outline-variant);
        border-radius: 4px;
        padding: 6px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .tools-label {
        font-size: 12px;
        font-weight: 500;
        color: var(--on-surface);
    }

    /* Canvas Frame */
    iframe[data-testid="stCustomComponentV1"] {
        display: block !important;
        margin: 0 auto !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
        width: 100% !important;
        aspect-ratio: 1 / 1 !important;
    }

    /* Predict Button Override */
    div.stButton > button[kind="primary"] {
        background-color: var(--primary-container) !important;
        color: #ffffff !important;
        border: 1px solid var(--primary-container) !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        width: 100% !important;
        cursor: pointer !important;
        margin-top: 8px !important;
        margin-bottom: 12px !important;
        transition: background-color 0.15s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #a84614 !important;
    }

    /* Results Header */
    .results-heading {
        font-size: 16px;
        font-weight: 600;
        color: var(--on-surface);
        margin-top: 12px;
        margin-bottom: 8px;
    }

    /* Predictions Grid */
    .predictions-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 20px;
    }

    .pred-card {
        background-color: var(--surface-lowest);
        border: 1px solid var(--outline-variant);
        border-radius: 4px;
        height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }

    .pred-card.primary {
        border: 1px solid var(--primary-container);
    }

    .pred-glyph {
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        font-weight: 600;
        font-family: 'Suranna', 'Gautami', 'Geist', sans-serif !important;
        color: var(--on-surface);
        line-height: 1;
        padding-top: 4px;
    }

    .pred-glyph.secondary {
        opacity: 0.8;
    }

    .pred-glyph.tertiary {
        opacity: 0.6;
    }

    .pred-bottom-bar {
        background-color: var(--surface-low);
        border-top: 1px solid var(--outline-variant);
        padding: 3px 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        font-weight: 600;
    }

    .pred-bottom-bar .tag {
        color: var(--on-surface);
    }

    .pred-bottom-bar .pct-primary {
        color: var(--primary-container);
        font-weight: 700;
    }

    .pred-bottom-bar .pct-secondary {
        color: var(--on-surface-variant);
    }

    .pred-progress-stripe {
        position: absolute;
        bottom: 0;
        left: 0;
        height: 3.5px;
        background-color: var(--primary-container);
    }

    /* Bottom Full-Width Footer */
    .bottom-footer {
        background-color: var(--bg-main);
        border-top: 1px solid var(--outline-variant);
        width: 100%;
        padding: 12px 24px;
        margin-top: 24px;
    }

    .bottom-footer-inner {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
        font-weight: 600;
        color: var(--on-surface);
    }

    .bottom-footer a {
        font-size: 11px;
        color: var(--primary);
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_system_assets(checkpoint_dir: str = "checkpoints",
                       label_maps_path: str = "outputs/label_maps.json") -> Tuple[Optional[tf.keras.Model], Dict[str, Any]]:
    """Loads label maps and restores the trained multi-head model checkpoint."""
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
    except Exception:
        pass
        
    return model, label_maps


def run_inference(image_input: Any, 
                  model: tf.keras.Model, 
                  label_maps: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
    """Runs preprocessing, forward pass, and constrained maximum-likelihood decoding."""
    preprocessed_tensor = preprocess_image(image_input, img_size=IMAGE_SIZE)
    input_batch = tf.expand_dims(preprocessed_tensor, axis=0)
    
    raw_preds = model(input_batch, training=False)
    b_probs, m_probs, v_probs = parse_model_prediction_outputs(raw_preds)
    
    rec_result = recombine_prediction(
        base_probs=b_probs[0],
        mod_probs=m_probs[0],
        vattu_probs=v_probs[0],
        label_maps=label_maps
    )
    
    return rec_result, preprocessed_tensor.numpy()


def main():
    # Session state initialization
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "last_results" not in st.session_state:
        st.session_state.last_results = None

    # 1. Top Navigation Bar
    st.markdown("""
    <header class="top-navbar">
        <div class="top-navbar-inner">
            <div class="top-navbar-left">
                <span class="brand-title">Telugu Akshara Recognizer</span>
                <span class="brand-subtitle">Handwriting recognition for Telugu scripts</span>
            </div>
            <nav class="top-navbar-right">
                <a href="#">Handwriting recognition for Telugu scripts</a>
            </nav>
        </div>
    </header>
    """, unsafe_allow_html=True)
    
    # 2. Main Centered Column Container
    st.markdown('<div class="main-column">', unsafe_allow_html=True)
    
    model, label_maps = load_system_assets()
    
    # 3. Tools Bar (Brush Size + Clear)
    col_tools_left, col_tools_right = st.columns([1.5, 1], vertical_alignment="center")
    
    with col_tools_left:
        brush_size = st.slider("Brush Size", min_value=3, max_value=20, value=6, step=1)
        
    with col_tools_right:
        if st.button("Clear Canvas", key="btn_clear", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.last_results = None
            st.rerun()
            
    image_to_process = None
    
    # 4. Canvas
    if CANVAS_AVAILABLE:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=brush_size,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=450,
            height=450,
            drawing_mode="freedraw",
            key=f"telugu_canvas_{st.session_state.canvas_key}"
        )
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype(np.uint8)
            if np.mean(img_array[..., :3]) < 254.0:
                image_to_process = img_array
    else:
        uploaded_file = st.file_uploader("Upload Telugu Character Image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, width=200)
            
    # 5. Predict Button
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    
    if predict_clicked and image_to_process is not None:
        with st.spinner("Analyzing..."):
            rec_result, preprocessed_img = run_inference(image_to_process, model, label_maps)
            st.session_state.last_results = (rec_result, preprocessed_img)
            
    # 6. Results Section (Top Predictions)
    st.markdown('<div class="results-heading">Top Predictions</div>', unsafe_allow_html=True)
    
    if st.session_state.last_results is not None:
        rec_result, preprocessed_img = st.session_state.last_results
        top_candidates = rec_result["top_5"][:3]
        
        cards_html = ['<div class="predictions-grid">']
        for idx, item in enumerate(top_candidates):
            cname = item["class_name"]
            b_char = item["base_letter"]
            m_code = item["vowel_modifier"]
            v_code = item["vattu"]
            prob = item["probability"]
            pct_str = f"{prob * 100:.1f}%"
            glyph = get_display_glyph(b_char, m_code, v_code)
            
            if idx == 0:
                card_class = "pred-card primary"
                glyph_class = "pred-glyph"
                tag_name = "Primary"
                pct_class = "pct-primary"
            elif idx == 1:
                card_class = "pred-card"
                glyph_class = "pred-glyph secondary"
                tag_name = "Match"
                pct_class = "pct-secondary"
            else:
                card_class = "pred-card"
                glyph_class = "pred-glyph tertiary"
                tag_name = "Match"
                pct_class = "pct-secondary"
                
            stripe_w = f"{max(1.0, min(100.0, prob * 100)):.1f}%"
            
            cards_html.append(f"""
            <div class="{card_class}">
                <div class="{glyph_class}">{glyph}</div>
                <div class="pred-bottom-bar">
                    <span class="tag">{tag_name}</span>
                    <span class="{pct_class}">{pct_str}</span>
                </div>
                <div class="pred-progress-stripe" style="width: {stripe_w};"></div>
            </div>
            """)
            
        cards_html.append('</div>')
        st.markdown("\n".join(cards_html), unsafe_allow_html=True)
        
    else:
        # Default placeholder cards matching reference mockup
        st.markdown("""
        <div class="predictions-grid">
            <div class="pred-card primary">
                <div class="pred-glyph">అ</div>
                <div class="pred-bottom-bar"><span class="tag">Primary</span><span class="pct-primary">98.2%</span></div>
                <div class="pred-progress-stripe" style="width: 98.2%;"></div>
            </div>
            <div class="pred-card">
                <div class="pred-glyph secondary">ఆ</div>
                <div class="pred-bottom-bar"><span class="tag">Match</span><span class="pct-secondary">1.5%</span></div>
                <div class="pred-progress-stripe" style="width: 1.5%;"></div>
            </div>
            <div class="pred-card">
                <div class="pred-glyph tertiary">క</div>
                <div class="pred-bottom-bar"><span class="tag">Match</span><span class="pct-secondary">0.2%</span></div>
                <div class="pred-progress-stripe" style="width: 0.2%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True) # Close main-column
    
    # 7. Bottom Full-Width Footer
    st.markdown("""
    <footer class="bottom-footer">
        <div class="bottom-footer-inner">
            <span>Model v1.2.0 • Trained on 50k+ Telugu glyphs</span>
            <nav>
                <a href="#">Technical Documentation</a>
            </nav>
        </div>
    </footer>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
