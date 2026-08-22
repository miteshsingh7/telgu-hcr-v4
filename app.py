"""Telugu Handwritten Character Recognizer — Streamlit App (Single-Page No-Scroll View).

Theme: Matte Technical Utility (Burnt Orange & Muted Stone Greige)
Layout: Precision Single-Viewport Container (100% visible without scrolling)
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
    "aa": "\u0C3E",    # ా (దీర్ఘం)
    "i": "\u0C3F",     # ి (గుడి)
    "ii": "\u0C40",    # ీ (గుడిదీర్ఘం)
    "u": "\u0C41",     # ు (కొమ్ము)
    "uu": "\u0C42",    # ూ (కొమ్ముదీర్ఘం)
    "ru": "\u0C43",    # ృ (వట్రుసుడి)
    "ruu": "\u0C44",   # ౄ (వట్రుసుడి దీర్ఘం)
    "e": "\u0C46",     # ె (ఎత్వం)
    "ee": "\u0C47",    # ే (ఏత్వం)
    "ai": "\u0C48",    # ై (ఐత్వం)
    "o": "\u0C4A",     # ొ (ఒత్వం)
    "oo": "\u0C4B",    # ో (ఓత్వం)
    "au": "\u0C4C",    # ౌ (ఔత్వం)
    "am": "\u0C02",    # ం (సున్నా)
    "ah": "\u0C03"     # ః (విసర్గ)
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
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Inject Matte Technical Utility Design System CSS (Precision Compact Fit)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Suranna&family=Gautami&display=swap');

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

    html, body, [class*="css"] {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--on-surface) !important;
    }

    /* Remove excessive top & bottom padding to ensure zero-scroll */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    #MainMenu, footer {
        display: none !important;
    }

    .block-container {
        max-width: 440px !important;
        padding-top: 0.8rem !important;
        padding-bottom: 0.8rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* Element spacing compaction */
    .element-container, div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.35rem !important;
    }

    /* Header */
    .app-header {
        border-bottom: 1px solid var(--outline-variant);
        padding-bottom: 6px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        width: 100%;
    }

    .app-title {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--on-surface);
        margin: 0;
        line-height: 1.1;
    }

    .app-subtitle {
        font-size: 11.5px;
        color: var(--on-surface-variant);
        margin: 0;
    }

    .app-header-badge {
        font-size: 11px;
        font-weight: 600;
        color: var(--primary-container);
        border-bottom: 1.5px solid var(--primary-container);
        padding-bottom: 1px;
        text-decoration: none;
    }

    /* Tools Bar */
    .tools-wrapper {
        background-color: var(--surface-lowest);
        border: 1px solid var(--outline-variant);
        border-radius: 4px;
        padding: 4px 8px;
        margin-bottom: 6px;
    }

    /* Canvas Wrapper */
    iframe[data-testid="stCustomComponentV1"] {
        display: block !important;
        margin: 0 auto !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
    }

    /* Predict Button */
    div.stButton > button[kind="primary"] {
        background-color: var(--primary-container) !important;
        color: #ffffff !important;
        border: 1px solid var(--primary-container) !important;
        border-radius: 4px !important;
        font-family: 'Geist', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        width: 100% !important;
        cursor: pointer !important;
        margin-top: 4px !important;
        margin-bottom: 6px !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #a84614 !important;
        border-color: #a84614 !important;
    }

    /* Top Predictions Grid */
    .predictions-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--on-surface);
        margin-top: 6px;
        margin-bottom: 6px;
    }

    .predictions-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        margin-bottom: 8px;
    }

    .pred-card {
        background-color: var(--surface-lowest);
        border: 1px solid var(--outline-variant);
        border-radius: 4px;
        height: 94px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }

    .pred-card.primary {
        border: 1.5px solid var(--primary-container);
    }

    .pred-glyph-container {
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        font-weight: 600;
        font-family: 'Suranna', 'Gautami', 'Geist', sans-serif;
        color: var(--on-surface);
        line-height: 1;
        padding-top: 4px;
    }

    .pred-glyph-container.secondary {
        opacity: 0.75;
    }

    .pred-glyph-container.tertiary {
        opacity: 0.55;
    }

    .pred-bottom-bar {
        background-color: var(--surface-low);
        border-top: 1px solid var(--outline-variant);
        padding: 2px 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 10px;
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
        height: 3px;
        background-color: var(--primary-container);
    }

    /* Breakdown Bar */
    .breakdown-bar {
        background-color: var(--surface-lowest);
        border: 1px solid var(--outline-variant);
        border-radius: 4px;
        padding: 4px 8px;
        display: flex;
        justify-content: space-around;
        font-size: 10.5px;
        color: var(--on-surface-variant);
        margin-top: 4px;
    }

    .breakdown-bar strong {
        color: var(--on-surface);
    }

    /* Footer */
    .app-footer {
        border-top: 1px solid var(--outline-variant);
        padding-top: 6px;
        margin-top: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 10px;
        font-weight: 500;
        color: var(--on-surface-variant);
    }

    .app-footer a {
        color: var(--primary-container);
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
    # 1. Compact Header
    st.markdown("""
    <div class="app-header">
        <div>
            <h1 class="app-title">Telugu Akshara Recognizer</h1>
            <p class="app-subtitle">Handwriting recognition for Telugu scripts</p>
        </div>
        <a class="app-header-badge" href="https://github.com/miteshsingh7/telgu-hcr-v4" target="_blank">v4.0 • 596 Compounds</a>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Load Assets
    model, label_maps = load_system_assets()
    
    # 3. Compact Tools Bar
    col_tools_left, col_tools_right = st.columns([1.1, 1], vertical_alignment="center")
    
    with col_tools_left:
        brush_size = st.slider("Brush Size", min_value=4, max_value=24, value=12, step=2, label_visibility="collapsed")
    
    with col_tools_right:
        input_mode = st.pills("Mode", ["Canvas", "Upload"], default="Canvas", label_visibility="collapsed")
        
    image_to_process = None
    
    # 4. Canvas or Upload (Exact 300x300 for zero-scroll fit)
    if input_mode == "Canvas":
        if CANVAS_AVAILABLE:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=brush_size,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=300,
                height=300,
                drawing_mode="freedraw",
                key="telugu_compact_canvas"
            )
            
            if canvas_result.image_data is not None:
                img_array = canvas_result.image_data.astype(np.uint8)
                if np.mean(img_array[..., :3]) < 254.0:
                    image_to_process = img_array
        else:
            st.warning("Canvas not installed. Use Upload mode.")
            uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, width=180)
            
    # 5. Predict Button
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
        
    if predict_clicked and image_to_process is not None:
        with st.spinner("Decoding..."):
            rec_result, preprocessed_img = run_inference(image_to_process, model, label_maps)
            st.session_state.last_results = (rec_result, preprocessed_img)
            
    # 6. Results Section
    st.markdown('<div class="predictions-title">Top Predictions</div>', unsafe_allow_html=True)
    
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
                card_type = "primary"
                glyph_class = ""
                tag_name = "Primary"
                pct_class = "pct-primary"
            elif idx == 1:
                card_type = "secondary"
                glyph_class = "secondary"
                tag_name = "Match"
                pct_class = "pct-secondary"
            else:
                card_type = "tertiary"
                glyph_class = "tertiary"
                tag_name = "Match"
                pct_class = "pct-secondary"
                
            stripe_width = f"{max(1.0, min(100.0, prob * 100)):.1f}%"
            
            cards_html.append(f"""
            <div class="pred-card {card_type}">
                <div class="pred-glyph-container {glyph_class}">{glyph}</div>
                <div class="pred-bottom-bar">
                    <span class="tag">{tag_name}</span>
                    <span class="{pct_class}">{pct_str}</span>
                </div>
                <div class="pred-progress-stripe" style="width: {stripe_width};"></div>
            </div>
            """)
        cards_html.append('</div>')
        st.markdown("\n".join(cards_html), unsafe_allow_html=True)
        
        # Compact single-row breakdown
        pred_item = rec_result["top_5"][0]
        st.markdown(f"""
        <div class="breakdown-bar">
            <span>Base: <strong>{pred_item['base_letter']}</strong></span>
            <span>Matra: <strong>{pred_item['vowel_modifier']}</strong></span>
            <span>Vattu: <strong>{pred_item['vattu']}</strong></span>
            <span>Class: <strong>{pred_item['class_name']}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Default placeholder cards
        st.markdown("""
        <div class="predictions-grid">
            <div class="pred-card primary">
                <div class="pred-glyph-container">అ</div>
                <div class="pred-bottom-bar"><span class="tag">Primary</span><span class="pct-primary">--%</span></div>
                <div class="pred-progress-stripe" style="width: 0%;"></div>
            </div>
            <div class="pred-card">
                <div class="pred-glyph-container secondary">ఆ</div>
                <div class="pred-bottom-bar"><span class="tag">Match</span><span class="pct-secondary">--%</span></div>
                <div class="pred-progress-stripe" style="width: 0%;"></div>
            </div>
            <div class="pred-card">
                <div class="pred-glyph-container tertiary">క</div>
                <div class="pred-bottom-bar"><span class="tag">Match</span><span class="pct-secondary">--%</span></div>
                <div class="pred-progress-stripe" style="width: 0%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 7. Compact Footer
    st.markdown("""
    <div class="app-footer">
        <span>Model v4.0 • 291k+ Telugu glyphs</span>
        <a href="https://github.com/miteshsingh7/telgu-hcr-v4" target="_blank">Technical Documentation</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
