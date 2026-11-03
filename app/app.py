import hashlib
import os
import sys
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import detector                      # noqa: E402
from ocr_module import ocr_image     # noqa: E402

st.set_page_config(page_title="LeakLens AI", page_icon="🛡️", layout="wide")

LABEL_NAMES = {"email": "Email", "phone": "Phone", "url_internal": "Internal URL",
               "credential": "Credential", "id_number": "ID number"}
STYLES = {"Blur": "blur", "Pixelate": "pixelate", "Black box": "black"}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 100%); color: #fff;
       padding: 1.6rem 1.9rem; border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p {margin: .35rem 0 0; opacity: .93; font-size: 1.02rem; color: #fff;}
.steps {display: flex; gap: .6rem; margin-top: .95rem; flex-wrap: wrap;}
.step {background: rgba(255,255,255,.2); padding: .3rem .85rem; border-radius: 999px; font-size: .85rem; color: #fff;}
.chip {display: inline-block; padding: .3rem .85rem; border-radius: 999px; font-weight: 600; font-size: .86rem;
       margin: 0 .45rem .45rem 0; color: #fff;}
.card-title {font-weight: 700; font-size: 1.05rem; margin-bottom: .35rem;}
.dot {display: inline-block; width: .8rem; height: .8rem; border-radius: 50%; margin-right: .55rem; vertical-align: middle;}
.empty {text-align: center; padding: 1.2rem .5rem;}
.empty h3 {margin-bottom: .2rem;}
.small {opacity: .75; font-size: .85rem;}
.risk {border: 3px solid; border-radius: 16px; padding: .7rem 1rem; text-align: center;}
.risk-label {font-size: .72rem; letter-spacing: .12em; opacity: .7;}
.risk-level {font-size: 2rem; font-weight: 800; line-height: 1.15;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def hex_color(label):
    b, g, r = detector.COLORS[label]
    return f"#{r:02x}{g:02x}{b:02x}"


def decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def mask(text, show):
    text = text.strip()
    return text if show else text[:2] + "•" * min(max(len(text) - 2, 0), 12)


def found_by(source):
    if source == "model":
        return "Model"
    if source == "rule+join":
        return "Rule · joined pieces"
    if source.startswith("rule:"):
        return "Rule · " + source[5:].replace("_", " ")
    return source


@st.cache_resource(show_spinner="Loading the model and OCR (first run only)...")
def load_models():
    import ocr_module
    detector.get_model()
    ocr_module.get_reader()
    return True


@st.cache_data(show_spinner="Reading the text in the image...")
def read_text(data: bytes, scale: float):
    return ocr_image(decode(data), scale=scale)      # slow, so it is cached per image


# ------------------------------------------------------------------ header
st.markdown(
    """<div class="hero"><h1>🛡️ LeakLens AI</h1>
    <p>Find and hide sensitive information in screenshots before you share them.</p>
    <div class="steps"><span class="step">1 · Upload</span><span class="step">2 · Review what was found</span>
    <span class="step">3 · Hide it and download</span></div></div>""",
    unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Sensitivity", 0.20, 0.90, 0.40, 0.05,
                          help="Lower catches more (more false alarms). Higher gives fewer false alarms.")
    scale = st.select_slider("OCR detail", options=[1.5, 2.0, 3.0], value=2.0,
                             help="Higher reads small text better but is slower and needs more memory.")
    st.subheader("Hiding")
    style_name = st.radio("Style", list(STYLES), horizontal=True,
                          help="A black box is the safest. Blur and pixelate keep the look of the page.")
    margin = st.slider("Extra margin (pixels)", 0, 20, 6, help="Grows each hidden area a little, in case a box is tight.")
    show_text = st.toggle("Show detected text in the table", value=False,
                          help="Off by default, so a secret is not displayed on your screen or in a screen share.")
    st.subheader("Colors")
    st.markdown("".join(f'<div><span class="dot" style="background:{hex_color(k)}"></span>{v}</div>'
                        for k, v in LABEL_NAMES.items()), unsafe_allow_html=True)
    st.caption("🔒 Everything runs on this computer. Images are not saved or uploaded.")

# ------------------------------------------------------------------ upload
file = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if file is None:
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown('<div class="empty"><h3>🔎 Finds</h3>Emails, phone numbers, internal URLs, passwords, '
                    'API keys, one-time codes and ID numbers.</div>', unsafe_allow_html=True)
    with c2, st.container(border=True):
        st.markdown('<div class="empty"><h3>🧠 How</h3>OCR reads the text, then a trained model and fixed '
                    'rules decide what is sensitive.</div>', unsafe_allow_html=True)
    with c3, st.container(border=True):
        st.markdown('<div class="empty"><h3>🛡️ Protects</h3>One click blurs the sensitive areas, and you '
                    'download a safe copy.</div>', unsafe_allow_html=True)
    st.stop()

data = file.getvalue()
img = decode(data)
if img is None:
    st.error("This file could not be read as an image.")
    st.stop()

load_models()
try:
    items = read_text(data, scale)                                  # cached, so moving a slider is instant
except RuntimeError as err:                                         # e.g. the computer ran out of memory
    st.error(str(err))
    st.stop()

out = detector.analyze(img, threshold, ocr_items=items)
flagged = out["flagged"]
key = (hashlib.md5(data).hexdigest(), threshold, scale, style_name, margin)

# ------------------------------------------------------------------ summary
risk = detector.risk_assessment(flagged)
RISK_COLORS = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a", "None": "#64748b"}
RISK_TEXT = {"High": "High", "Medium": "Medium", "Low": "Low", "None": "Clear"}
color = RISK_COLORS[risk["level"]]
counts = Counter(r["label"] for r in flagged)
c1, c2 = st.columns([1.2, 3])
c1.markdown(f'<div class="risk" style="border-color:{color}"><div class="risk-label">RISK LEVEL</div>'
            f'<div class="risk-level" style="color:{color}">{RISK_TEXT[risk["level"]]}</div>'
            f'<div class="small">Score {risk["score"]} / 100</div></div>', unsafe_allow_html=True)
with c2:
    if not items:
        st.warning("No text could be read in this image. Try a larger OCR detail or a clearer screenshot.")
    else:
        st.markdown(f"**{risk['headline']}**")
        if counts:
            st.markdown("".join(f'<span class="chip" style="background:{hex_color(k)}">{LABEL_NAMES[k]} · {n}</span>'
                                for k, n in counts.items()), unsafe_allow_html=True)
        st.caption(risk["advice"])

# ------------------------------------------------------------------ original and detected
left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown('<div class="card-title">Original</div>', unsafe_allow_html=True)
    st.image(to_rgb(img))
with right, st.container(border=True):
    st.markdown('<div class="card-title">Detected</div>', unsafe_allow_html=True)
    st.image(to_rgb(out["annotated"]))
    st.download_button("⬇️ Download image with boxes", png_bytes(out["annotated"]),
                       "leaklens_detected.png", "image/png", key="dl_detected")

# ------------------------------------------------------------------ hide button
if flagged:
    if st.button(f"🛡️ Hide sensitive data ({style_name.lower()})", type="primary"):
        protected = detector.redact(img, flagged, style=STYLES[style_name], margin=margin)
        st.session_state["protected"] = {"key": key, "img": protected, "png": png_bytes(protected), "n": len(flagged)}

    saved = st.session_state.get("protected")
    if saved and saved["key"] == key:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">✅ Protected image · {saved["n"]} area(s) hidden</div>',
                        unsafe_allow_html=True)
            st.image(to_rgb(saved["img"]))
            st.download_button("⬇️ Download protected image", saved["png"],
                               "leaklens_protected.png", "image/png", key="dl_protected", type="primary")
    elif saved:
        st.info("The settings or the image changed. Press the button again to update the protected image.")

    # ------------------------------------------------------------------ details
    with st.expander("What was found", expanded=True):
        rows = [{"Type": LABEL_NAMES[r["label"]], "Confidence": f'{r["prob"]:.0%}',
                 "Found by": found_by(r["source"]), "Text": mask(r["text"], show_text)} for r in flagged]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Boxes cover the whole text piece the OCR found, so a label next to a value (for example "
                   "'Phone:') is hidden too. That is deliberate: hiding a little too much is safer.")