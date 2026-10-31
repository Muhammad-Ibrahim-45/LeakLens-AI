# import joblib, cv2, numpy as np
# from ocr_module import ocr_image

# _model = None
# COLORS = {"email": (255, 100, 0), "phone": (0, 140, 255),
#           "url_internal": (200, 0, 160), "credential": (0, 0, 230)}

# def get_model(path="../src/leaklens_logreg.joblib"):
#     global _model
#     if _model is None:
#         _model = joblib.load(path)
#     return _model

# def detect(img, threshold=0.80, model_path="../src/leaklens_logreg.joblib"):
#     model = get_model(model_path)
#     classes = list(model.classes_)
#     safe_idx = classes.index("safe")
#     items = ocr_image(img)
#     if not items:
#         return items
#     probs = model.predict_proba([i["text"] for i in items])
#     for it, p in zip(items, probs):
#         p_sens = 1 - p[safe_idx]
#         others = p.copy(); others[safe_idx] = -1
#         if p_sens >= threshold:
#             it["label"], it["prob"] = classes[int(np.argmax(others))], round(float(p_sens), 3)
#         else:
#             it["label"], it["prob"] = "safe", round(float(1 - p_sens), 3)
#     return items

# def highlight(img, items):
#     out = img.copy()
#     for it in items:
#         if it["label"] == "safe":
#             continue
#         c = COLORS[it["label"]]
#         x, y, w, h = it["x"], it["y"], it["w"], it["h"]
#         cv2.rectangle(out, (x, y), (x + w, y + h), c, 3)
#         cv2.putText(out, it["label"], (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
#     return out

"""LeakLens detector: OCR pieces -> sensitive regions (ML model + safety-net rules)."""
import os
import re
import sys

import cv2
import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from ocr_module import ocr_image  # noqa: E402

MODEL_PATH = os.path.join(HERE, "leaklens_logreg.joblib")

COLORS = {  # BGR
    "email": (255, 100, 0),
    "phone": (0, 140, 255),
    "url_internal": (200, 0, 160),
    "credential": (0, 0, 230),
}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


# ---------------------------------------------------------------- 1. cleaning
def normalize(text):
    """Repair common OCR mistakes before classification."""
    t = text.strip()
    t = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", t)                       # "10 .0.4.21" -> "10.0.4.21"
    t = re.sub(r"(?i)\b(https?)\s*:\s*/\s*/\s*", r"\1://", t)           # "http ://" -> "http://"
    t = re.sub(r"^([A-Z][A-Z0-9]*_[A-Z0-9_]*)-(?=\S)", r"\1=", t)       # "DB_PASSWORD-x" -> "DB_PASSWORD=x"
    return t


# ----------------------------------------------------- 2. safety-net rules
_PRIVATE = (r"(?:localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)")
RULES = [
    ("credential", re.compile(r"AKIA[0-9A-Z]{12,20}")),
    ("credential", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("credential", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{16,}")),
    ("credential", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{15,}")),
    ("url_internal", re.compile(r"(?i)https?://" + _PRIVATE + r"[^\s'\"]*")),
    ("url_internal", re.compile(r"(?i)https?://[\w.-]+\.(?:local|corp|internal|lan)\b[^\s'\"]*")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
]

# ------------------------------------------------- 3. locating the sensitive part
_LOCATORS = {
    "email": [re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")],
    "phone": [re.compile(r"(?<![\w$-])\+?\(?\d[\d\s().-]{6,}\d(?![\w])")],
    "url_internal": [
        re.compile(r"(?i)https?://[^\s'\"]+"),
        re.compile(r"(?i)\b[\w.-]+\.(?:local|corp|internal|lan)\b[^\s'\"]*"),
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?[^\s'\"]*"),
    ],
    "credential": [
        re.compile(r"(?i)\b[\w.-]*(?:pass(?:word|wd)?|pwd|secret|token|api[_-]?key|access[_-]?key"
                   r"|private[_-]?key|auth\w*)[\w.-]*\s*[:=]\s*['\"]?([^\s'\"]{4,})"),
        re.compile(r"(?i)\b(?:password|passwd|pwd|pass|token|secret)\s+(?:is\s+)?['\"]?([^\s'\"]{6,})"),
    ],
}

_DATE_START = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")


def _secret_like(tok):
    has_d = any(c.isdigit() for c in tok)
    has_a = any(c.isalpha() for c in tok)
    mixed = any(c.islower() for c in tok) and any(c.isupper() for c in tok)
    return len(tok) >= 6 and ((has_d and has_a) or (mixed and len(tok) >= 10))


def plausible(label, span):
    """Cheap sanity checks that remove typical false alarms."""
    s = span.strip()
    if label == "phone":
        digits = sum(c.isdigit() for c in s)
        if "$" in s or re.search(r"[A-Za-z]", s) or not 7 <= digits <= 15:
            return False
        return not _DATE_START.match(s)
    if label == "email":
        return "@" in s and "." in s.split("@")[-1]
    if label == "url_internal":
        return "." in s or "localhost" in s.lower()
    if label == "credential":
        if re.search(r"\s", s):
            return len(s) >= 6
        return len(s) >= 6 and (_secret_like(s) or bool(re.match(r"(?i)(AKIA|ghp_|sk-)", s)))
    return False


def _overlap(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    return inter / max(1, min(a1 - a0, b1 - b0))


def _locate(norm, label, model, classes, threshold):
    """Return (start, end) of the sensitive part inside one OCR piece, or None."""
    for pat in _LOCATORS[label]:
        for m in pat.finditer(norm):
            g = 1 if pat.groups else 0
            s, e = m.start(g), m.end(g)
            if plausible(label, norm[s:e]):
                return s, e
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", norm)]
    if len(toks) > 1:
        probs = model.predict_proba([t for t, _, _ in toks])
        safe_idx = classes.index("safe")
        keep = []
        for (t, s, e), p in zip(toks, probs):
            others = p.copy()
            others[safe_idx] = -1
            if (1 - p[safe_idx]) >= threshold and classes[int(np.argmax(others))] == label and plausible(label, t):
                keep.append((s, e))
        if keep:
            return min(s for s, _ in keep), max(e for _, e in keep)
    return None


def _to_box(it, s, e, n, pad=1.0):
    cw = it["w"] / max(n, 1)
    x0 = max(it["x"], it["x"] + (s - pad) * cw)
    x1 = min(it["x"] + it["w"], it["x"] + (e + pad) * cw)
    return int(x0), int(it["y"]), max(int(x1 - x0), 1), int(it["h"])


# ------------------------------------------------------------ 4. main function
def detect(img=None, items=None, threshold=0.80, strong=0.95, precise=True, use_rules=True):
    """Return (detections, items).

    img        : OpenCV BGR image (used if items is None)
    items      : optional OCR output, so OCR is not repeated when you test thresholds
    threshold  : minimum P(sensitive) from the model to flag a piece
    strong     : above this, an unlocated piece is flagged as a whole
    precise    : True = box only the sensitive part, False = box the whole OCR piece
    use_rules  : add the safety-net rules (set False to see the ML model alone)
    """
    model = get_model()
    classes = list(model.classes_)
    safe_idx = classes.index("safe")
    items = [dict(i) for i in (items if items is not None else ocr_image(img))]
    for it in items:
        it["norm"] = normalize(it["text"])
    todo = [it for it in items if len(it["norm"]) >= 4]
    probs = model.predict_proba([it["norm"] for it in todo]) if todo else []

    detections = []
    for it, p in zip(todo, probs):
        norm, n = it["norm"], len(it["norm"])
        hits = []
        if use_rules:
            for label, pat in RULES:
                for m in pat.finditer(norm):
                    if plausible(label, m.group()):
                        hits.append((label, 1.0, "rule", m.start(), m.end()))
        p_sens = 1 - float(p[safe_idx])
        if p_sens >= threshold:
            others = p.copy()
            others[safe_idx] = -1
            label = classes[int(np.argmax(others))]
            span = _locate(norm, label, model, classes, threshold)
            if span is None and p_sens >= strong and plausible(label, norm):
                span = (0, n)
            if span and not any(_overlap(span[0], span[1], h[3], h[4]) >= 0.5 for h in hits):
                hits.append((label, round(p_sens, 3), "model", span[0], span[1]))
        for label, prob, source, s, e in hits:
            x, y, w, h = _to_box(it, s, e, n) if precise else (it["x"], it["y"], it["w"], it["h"])
            detections.append({"text": norm[s:e], "label": label, "prob": prob, "source": source,
                               "line": norm, "x": x, "y": y, "w": w, "h": h, "ocr_conf": it["conf"]})
    detections.sort(key=lambda d: (d["y"], d["x"]))
    return detections, items


def highlight(img, detections, thickness=3, fill_alpha=0.18):
    """Draw color-coded boxes with a label above each one."""
    overlay = img.copy()
    for d in detections:
        c = COLORS[d["label"]]
        cv2.rectangle(overlay, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]), c, -1)
    out = cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0)
    for d in detections:
        c = COLORS[d["label"]]
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, thickness)
        tag = f'{d["label"]} {d["prob"]:.0%}'
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y - 4 if y - th - 8 > 0 else y + h + th + 6
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 6, ty + 3), c, -1)
        cv2.putText(out, tag, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
