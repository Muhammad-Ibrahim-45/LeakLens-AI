# import os
# import sys

# import cv2
# import numpy as np
# import pandas as pd
# import streamlit as st

# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, os.path.join(ROOT, "src"))
# import detector                      # noqa: E402
# from ocr_module import ocr_image     # noqa: E402

# st.set_page_config(page_title="LeakLens AI", layout="wide")


# def decode(data: bytes):
#     return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# @st.cache_resource(show_spinner="Loading the model and OCR (first run only)...")
# def load_models():
#     import ocr_module
#     detector.get_model()
#     ocr_module.get_reader()
#     return True


# @st.cache_data(show_spinner="Reading the text in the image...")
# def read_text(data: bytes, scale: float):
#     return ocr_image(decode(data), scale=scale)      # slow, so it is cached per image


# st.title("LeakLens AI")
# st.caption("Upload a screenshot. LeakLens finds emails, phone numbers, internal URLs and "
#            "credentials before you share it. Everything runs on this computer.")

# with st.sidebar:
#     st.header("Settings")
#     threshold = st.slider("Sensitivity threshold", 0.20, 0.90, 0.40, 0.05,
#                           help="Lower = catches more (more false alarms). Higher = fewer false alarms.")
#     scale = st.select_slider("OCR detail", options=[1.5, 2.0, 3.0], value=2.0,
#                              help="Higher reads small text better but is slower.")

# file = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"])

# if file is None:
#     st.info("Upload a screenshot to start.")
# else:
#     data = file.getvalue()
#     img = decode(data)
#     if img is None:
#         st.error("This file could not be read as an image.")
#         st.stop()
#     load_models()
#     items = read_text(data, scale)                              # cached, so moving the slider is instant
#     out = detector.analyze(img, threshold, ocr_items=items)     # fast

#     left, right = st.columns(2)
#     left.image(img[:, :, ::-1], caption="Original", use_container_width=True)
#     right.image(out["annotated"][:, :, ::-1],
#                 caption=f"{len(out['flagged'])} sensitive item(s) found", use_container_width=True)

#     if out["flagged"]:
#         table = pd.DataFrame(out["flagged"])[["label", "prob", "source", "text"]]
#         st.subheader("What was found")
#         st.dataframe(table, use_container_width=True)
#     else:
#         st.success("Nothing sensitive found.")

#     ok, png = cv2.imencode(".png", out["annotated"])
#     st.download_button("Download highlighted image", png.tobytes(), "leaklens_highlighted.png", "image/png")
import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import detector                      # noqa: E402
from ocr_module import ocr_image     # noqa: E402

st.set_page_config(page_title="LeakLens AI", layout="wide")


def decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@st.cache_resource(show_spinner="Loading the model and OCR (first run only)...")
def load_models():
    import ocr_module
    detector.get_model()
    ocr_module.get_reader()
    return True


@st.cache_data(show_spinner="Reading the text in the image...")
def read_text(data: bytes, scale: float):
    return ocr_image(decode(data), scale=scale)      # slow, so it is cached per image


st.title("LeakLens AI")
st.caption("Upload a screenshot. LeakLens finds emails, phone numbers, internal URLs and "
           "credentials before you share it. Everything runs on this computer.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Sensitivity threshold", 0.20, 0.90, 0.40, 0.05,
                          help="Lower = catches more (more false alarms). Higher = fewer false alarms.")
    scale = st.select_slider("OCR detail", options=[1.5, 2.0, 3.0], value=2.0,
                             help="Higher reads small text better but is slower.")

file = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"])

if file is None:
    st.info("Upload a screenshot to start.")
else:
    data = file.getvalue()
    img = decode(data)
    if img is None:
        st.error("This file could not be read as an image.")
        st.stop()
    load_models()
    try:
        items = read_text(data, scale)                          # cached, so moving the slider is instant
    except RuntimeError as err:                                 # e.g. the computer ran out of memory
        st.error(str(err))
        st.stop()
    out = detector.analyze(img, threshold, ocr_items=items)     # fast

    left, right = st.columns(2)
    left.image(img[:, :, ::-1], caption="Original", use_container_width=True)
    right.image(out["annotated"][:, :, ::-1],
                caption=f"{len(out['flagged'])} sensitive item(s) found", use_container_width=True)

    if out["flagged"]:
        table = pd.DataFrame(out["flagged"])[["label", "prob", "source", "text"]]
        st.subheader("What was found")
        st.dataframe(table, use_container_width=True)
    else:
        st.success("Nothing sensitive found.")

    ok, png = cv2.imencode(".png", out["annotated"])
    st.download_button("Download highlighted image", png.tobytes(), "leaklens_highlighted.png", "image/png")