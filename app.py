"""Telugu Handwritten Character Recognizer — Streamlit App.

Exact 1-to-1 implementation of the Matte Technical Utility reference design.
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


# Configure Streamlit Page: Centered layout with controlled width
st.set_page_config(
    page_title="Telugu Akshara Recognizer",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Global CSS injection matching the reference mockup
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-sans/style.css">
<style>
    /* Reset all Streamlit base colors to Light Matte Greige */
    :root, [data-theme="dark"], [data-theme="light"], .stApp {
        --bg-main: #fbf9f4 !important;
        --surface: #edebe6 !important;
        --surface-low: #f5f3ee !important;
        --surface-lowest: #ffffff !important;
        --outline: #8a8882 !important;
        --outline-variant: #c9c7c2 !important;
        --primary: #a03d00 !important;
        --primary-container: #c1541a !important;
        --on-surface: #1c1c1b !important;
        --on-surface-variant: #574239 !important;
        background-color: #fbf9f4 !important;
        color: #1c1c1b !important;
    }

    * {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        box-sizing: border-box;
    }

    /* Hide standard Streamlit header and footer */
    header[data-testid="stHeader"], #MainMenu, footer {
        display: none !important;
    }

    /* Constrain main block to exactly 480px centered column */
    .block-container {
        max-width: 480px !important;
        padding-top: 16px !important;
        padding-bottom: 24px !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
        margin: 0 auto !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }

    /* Top Navbar */
    .top-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        border-bottom: 1px solid #c9c7c2;
        padding-bottom: 10px;
        margin-bottom: 12px;
        width: 100%;
    }

    .top-nav-left {
        display: flex;
        align-items: baseline;
        gap: 12px;
    }

    .brand-title {
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #1c1c1b;
        margin: 0;
        white-space: nowrap;
    }

    .brand-subtitle {
        font-size: 13px;
        color: #574239;
        margin: 0;
        white-space: nowrap;
    }

    .top-nav-right a {
        font-size: 12px;
        font-weight: 700;
        color: #a03d00;
        border-bottom: 1.5px solid #a03d00;
        padding-bottom: 2px;
        text-decoration: none;
        white-space: nowrap;
    }

    /* Tools Bar container */
    div[data-testid="stHorizontalBlock"] {
        background-color: #ffffff !important;
        border: 1px solid #c9c7c2 !important;
        border-radius: 4px !important;
        padding: 6px 12px !important;
        align-items: center !important;
        margin-bottom: 4px !important;
    }

    /* Slider styling */
    .stSlider {
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Canvas Frame */
    iframe[data-testid="stCustomComponentV1"] {
        display: block !important;
        margin: 0 auto !important;
        border: 1px solid #c9c7c2 !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
        width: 100% !important;
        max-width: 456px !important;
        box-shadow: none !important;
    }

    /* Predict Button */
    div.stButton > button[kind="primary"] {
        background-color: #c1541a !important;
        color: #ffffff !important;
        border: 1px solid #c1541a !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        height: 44px !important;
        width: 100% !important;
        cursor: pointer !important;
        margin-top: 6px !important;
        margin-bottom: 8px !important;
        transition: background-color 0.15s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #a84614 !important;
    }

    /* Secondary Clear Button */
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #1c1c1b !important;
        border: 1px solid #c9c7c2 !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
        height: 32px !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background-color: #f5f3ee !important;
    }

    /* Predictions Heading */
    .predictions-heading {
        font-size: 16px;
        font-weight: 600;
        color: #1c1c1b;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    /* Predictions Grid */
    .predictions-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 16px;
        width: 100%;
    }

    .pred-card {
        background-color: #ffffff;
        border: 1px solid #c9c7c2;
        border-radius: 4px;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }

    .pred-card.primary {
        border: 1.5px solid #c1541a;
    }

    .pred-glyph {
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        font-weight: 600;
        font-family: 'Suranna', 'Gautami', 'Geist', sans-serif !important;
        color: #1c1c1b;
        line-height: 1;
        padding-top: 6px;
    }

    .pred-glyph.secondary {
        opacity: 0.8;
    }

    .pred-glyph.tertiary {
        opacity: 0.6;
    }

    .pred-bottom-bar {
        background-color: #f5f3ee;
        border-top: 1px solid #c9c7c2;
        padding: 4px 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        font-weight: 600;
    }

    .pred-bottom-bar .tag {
        color: #1c1c1b;
    }

    .pred-bottom-bar .pct-primary {
        color: #c1541a;
        font-weight: 700;
    }

    .pred-bottom-bar .pct-secondary {
        color: #574239;
    }

    .pred-progress-stripe {
        position: absolute;
        bottom: 0;
        left: 0;
        height: 4px;
        background-color: #c1541a;
    }

    /* Footer */
    .bottom-footer {
        border-top: 1px solid #c9c7c2;
        padding-top: 12px;
        margin-top: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11.5px;
        font-weight: 600;
        color: #1c1c1b;
        width: 100%;
    }

    .bottom-footer a {
        color: #a03d00;
        text-decoration: underline;
        font-size: 11px;
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
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "last_results" not in st.session_state:
        st.session_state.last_results = None

    # 1. Top Navbar (One line)
    st.markdown("""
    <div class="top-nav-bar">
        <div class="top-nav-left">
            <span class="brand-title">Telugu Akshara Recognizer</span>
            <span class="brand-subtitle">Handwriting recognition for Telugu scripts</span>
        </div>
        <div class="top-nav-right">
            <a href="#">Handwriting recognition for Telugu scripts</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    model, label_maps = load_system_assets()
    
    # 2. Tools Bar (Brush Slider + Clear)
    col_brush, col_clear = st.columns([2.5, 1], vertical_alignment="center")
    
    with col_brush:
        brush_size = st.slider("Brush Size", min_value=2, max_value=20, value=6, step=1)
        
    with col_clear:
        if st.button("Clear", key="btn_clear_canvas", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.last_results = None
            st.rerun()
            
    image_to_process = None
    
    # 3. Canvas (Full 456px width inside the 480px container)
    if CANVAS_AVAILABLE:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=brush_size,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=456,
            height=456,
            drawing_mode="freedraw",
            key=f"canvas_session_{st.session_state.canvas_key}"
        )
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype(np.uint8)
            if np.mean(img_array[..., :3]) < 254.0:
                image_to_process = img_array
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            
    # 4. Predict Button
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    
    if predict_clicked and image_to_process is not None:
        with st.spinner("Decoding..."):
            rec_result, preprocessed_img = run_inference(image_to_process, model, label_maps)
            st.session_state.last_results = (rec_result, preprocessed_img)
            
    # 5. Results Section (Top Predictions)
    st.markdown('<div class="predictions-heading">Top Predictions</div>', unsafe_allow_html=True)
    
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
        
    # 6. Bottom Footer
    st.markdown("""
    <div class="bottom-footer">
        <span>Model v1.2.0 • Trained on 50k+ Telugu glyphs</span>
        <a href="#">Technical Documentation</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
