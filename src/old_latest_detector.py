
"""LeakLens detector.

One file, used by BOTH the notebook and the Streamlit app, so the logic lives in one place.
    classify(items, threshold)  -> labels each OCR piece (model + sanity checks + rules)
    draw_boxes(img, results)    -> copy of the image with color-coded boxes
    analyze(img, ...)           -> OCR + classify + draw in one call
"""

import re
from pathlib import Path

import cv2
import joblib
import numpy as np


# ------------------------------------------------------------------ model setup

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "leaklens_logreg.joblib"

_model = None

THRESHOLD = 0.40          # default sensitivity

# A private IP with no scheme/port/path (e.g. "192.168.1.132") is flagged as an internal URL only when the
# rest of the OCR piece has at most this many words of 3+ letters.  0 = only when the IP is alone,
# None = never.  "10.20.30.45 - Remote Desktop Connection" has 3 words, so it stays safe at 2.
BARE_IP_MAX_WORDS = 2

# Domains whose addresses must always be recognised, even when OCR turned the "@" into a letter.
KNOWN_EMAIL_DOMAINS = [
    "cloud.neduet.edu.pk",
    "neduet.edu.pk",
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com"
]

COLORS = {
    "email": (255, 100, 0),
    "phone": (0, 140, 255),
    "url_internal": (200, 0, 160),
    "credential": (0, 0, 230)
}  # BGR


def get_model():
    """Load the trained model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file missing: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


# ------------------------------------------------------------------ patterns

DATE = re.compile(
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
)

PHONE_PART = re.compile(
    r"(?<![\w$-])\+?\(?\d[\d\s().-]{6,}\d(?![\w])"
)

IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

ID_LIKE = re.compile(r"[A-Z]{2,6}-\d+(?:-\d+)*")

# The domain must end in letters, so "search-api@1.42.0" is not an email.
EMAIL = re.compile(
    r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)

_PRIV = (
    r"(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3})"
)

_NOT_MORE_DIGITS = r"(?!\.?\d)"

# What counts as an INTERNAL URL: scheme + private/internal host, internal host names,
# or a private IP followed by a port or a path. A bare IP alone is NOT a URL.
INTERNAL_URL = re.compile(
    r"(?i)(?:https?://(?:localhost|" + _PRIV + r")" + _NOT_MORE_DIGITS + r")"
    r"|(?:https?://[\w.-]+\.(?:local|corp|internal|lan)\b)"
    r"|(?:\b[\w-]+(?:\.[\w-]+)*\.(?:internal|corp|lan)\b)"
    r"|(?:\b[\w-]+(?:\.[\w-]+)*\.local(?=[:/]))"
    r"|(?:(?<![\d.])" + _PRIV + r"(?=:\d{2,5}|/\S))"
    r"|(?:\blocalhost(?=:\d))"
    r"|(?:https?://(?:staging|dev|uat|intranet|admin|jenkins|grafana|vpn)\.)"
    r"|(?:\bhttps?\s*:\s*[/\\Il|]{0,3}\s*(?:[\w-]+[. ]){1,4}(?:corp|internal|lan|local))"
    r"|(?:\bhttps?\s*:\s*[/\\Il|]{0,3}\s*(?:staging|dev|uat|intranet|admin|jenkins|grafana|vpn)[. ])"
)

# Local part directly followed by a known domain; the "@" may be missing, spaced, or read as a letter or symbol.
_KNOWN_EMAIL = re.compile(
    r"(?i)(?<![\w.+-])[\w+-][\w.+-]*[A-Za-z0-9](?:\s?[@\u00a9]\s?|\(a\))?(?:"
    + "|".join(
        re.escape(d)
        for d in sorted(KNOWN_EMAIL_DOMAINS, key=len, reverse=True)
    )
    + r")\b"
)

BARE_IP = re.compile(
    r"(?<![\d.])" + _PRIV + _NOT_MORE_DIGITS
)


RULES = [
    ("credential", re.compile(r"AKIA[0-9A-Z]{12,20}")),
    ("credential", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("credential", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")),
    ("credential", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{15,}")),
    (
        "credential",
        re.compile(
            r"(?i)[?&](?:sig|signature|token|access_token|api_?key|key|secret|password)="
            r"[A-Za-z0-9%._~+/-]{8,}"
        )
    ),
    (
        "credential",
        re.compile(
            r"(?i)(?:pass(?:word|wd)?|pwd|secret|token|api[_-]?key|access[_-]?key"
            r"|private[_-]?key)\w*\s*[:=]\s*['\"]?(?=\S*\d)\S{4,}"
        )
    ),
    (
        "credential",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd)\s+(?:is\s+)?['\"]?(?=\S*\d)\S{6,}"
        )
    ),
    ("url_internal", INTERNAL_URL),
    ("email", EMAIL),
    ("email", _KNOWN_EMAIL),
    (
        "email",
        re.compile(
            r"(?i)\b\w*e-?mail\w*\s*[:=]\s*['\"]?[\w.+-]{2,}\."
            r"(?:com|org|net|edu|gov|io|co|pk|uk|de|in|us|info|biz|me|app|dev|ai|ca|au|fr|it|es|nl|ru|cn|jp|br"
            r"|corp|internal|local|lan)\b"
        )
    ),
]


# ------------------------------------------------------------------ cleaning

def normalize(t):
    """Repair common OCR mistakes before classifying."""
    t = t.strip()

    t = re.sub(
        r"(?<=\d)\s*\.\s*(?=\d)",
        ".",
        t
    )  # 10 .0.4.21 -> 10.0.4.21

    t = re.sub(
        r"(?i)\b(https?)\s*:\s*/\s*/\s*",
        r"\1://",
        t
    )  # http: // -> http://

    t = re.sub(
        r"\b((?i:https?)):(?!//)(?:/|[Il|]{2}|/[Il|]|[Il|]/)(?=[A-Za-z0-9])",
        r"\1://",
        t
    )  # https:ll -> https://

    t = re.sub(
        r"(?i)\s*\.\s*(?=(?:internal|corp|lan|local)\b)",
        ".",
        t
    )  # host . internal -> host.internal

    t = re.sub(
        r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*)-(?=\S)",
        r"\1=",
        t
    )  # KEY-value -> KEY=value

    dom = r"(?=[A-Za-z0-9-]+\s*[.,]\s*[A-Za-z]{2,})"

    t = re.sub(
        r"(\w)\s*(?:[\u00a9@]|\(a\))\s*" + dom,
        r"\1@",
        t
    )  # john (c) gmail.com -> john@gmail.com

    for _ in range(3):
        t = re.sub(
            r"(@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)\s*[.,]\s*([A-Za-z]{2,})\b",
            r"\1.\2",
            t
        )  # "@gmail , com" -> "@gmail.com"

    return t


# ------------------------------------------------------------ sanity checks

def phone_ok(text):
    """7-15 digits, not a date, not an IP address."""
    for m in PHONE_PART.finditer(text):
        part = m.group().strip()
        digits = sum(c.isdigit() for c in part)

        if IPV4.search(part):
            continue

        if 7 <= digits <= 15 and not DATE.match(part):
            return True

    return False


def credential_ok(text):
    """A secret-looking token or a known key format."""
    if re.search(r"(?i)AKIA[0-9A-Z]{8,}|ghp_\w{10,}|sk-[\w-]{10,}", text):
        return True

    for t in re.findall(r"\S+", text):
        t = t.strip("'\".,;:()[]{}")

        if re.search(r"@v?\d+(?:\.\d+)+$", t) or re.fullmatch(r"v?\d+(?:\.\d+)+", t):
            continue  # version strings

        if (
            len(t) >= 6
            and not ID_LIKE.fullmatch(t)
            and any(c.isdigit() for c in t)
            and any(c.isalpha() for c in t)
        ):
            return True

    return False


# ------------------------------------------------------------------- main API

def rule_label(text):
    """Fixed-format rules. Returns a label or None. These are checked BEFORE the model."""
    for lab, pat in RULES:
        if pat.search(text):
            return lab

    if BARE_IP_MAX_WORDS is not None:
        m = BARE_IP.search(text)

        if m and len(
            re.findall(r"[A-Za-z]{3,}", text[:m.start()] + " " + text[m.end():])
        ) <= BARE_IP_MAX_WORDS:
            return "url_internal"

    return None


def classify(items, threshold=THRESHOLD):
    """Return a copy of the OCR items with label, prob and source ('model', 'rule' or 'rule+join').

    Order of decisions for every OCR piece:
      1. A fixed-format rule matches (email, internal URL, key formats) -> that label wins.
      2. Otherwise the model's sensitivity (1 - P(safe)) must reach the threshold,
         and its answer must pass the sanity check for its type.
    """
    items = [dict(i) for i in items]

    if not items:
        return items

    model = get_model()
    classes = [str(c) for c in model.classes_]
    safe = classes.index("safe")

    probs = model.predict_proba(
        [normalize(i["text"]) for i in items]
    )

    for it, p in zip(items, probs):
        t = normalize(it["text"])

        p_sens = 1 - p[safe]

        others = p.copy()
        others[safe] = -1

        label, prob, src = "safe", round(float(1 - p_sens), 3), "-"

        rl = rule_label(t)

        if rl:
            label, prob, src = rl, 1.0, "rule"

        elif p_sens >= threshold:
            lab = classes[int(np.argmax(others))]

            if lab == "phone":
                ok = phone_ok(t)

            elif lab == "credential":
                ok = credential_ok(t)

            elif lab == "email":
                ok = bool(EMAIL.search(t)) or (
                    p_sens >= 0.80 and bool(re.search(r"\w@[A-Za-z]", t))
                )

            elif lab == "url_internal":
                ok = bool(INTERNAL_URL.search(t))

            else:
                ok = True

            if ok:
                label, prob, src = lab, round(float(p_sens), 3), "model"

        it["label"] = label
        it["prob"] = prob
        it["source"] = src

    return _stitch(items, RULES)


def _stitch(results, rules):
    """Second pass: join neighbouring pieces on the same text line.

    Catches things OCR cut in two, e.g.
    'SUPPORT_EMAIL=ali.khan' + '@example.com'.
    """
    extra = []

    for a in results:
        best = None

        for b in results:
            if b is a:
                continue

            h = max(a["h"], b["h"], 1)

            same_line = abs(
                (a["y"] + a["h"] / 2) - (b["y"] + b["h"] / 2)
            ) < 0.6 * h

            gap = b["x"] - (a["x"] + a["w"])

            if (
                same_line
                and -0.3 * h <= gap < 4.0 * h
                and (best is None or gap < best[0])
            ):
                best = (gap, b)

        if best is None:
            continue

        gap, b = best

        if a["label"] != "safe" and b["label"] != "safe":
            continue

        h = max(a["h"], b["h"], 1)

        ta = normalize(a["text"])
        tb = normalize(b["text"])

        joined = normalize(
            a["text"] + ("" if gap < 0.3 * h else " ") + b["text"]
        )

        for lab, pat in rules:
            if pat.search(joined) and not (pat.search(ta) or pat.search(tb)):
                x0 = min(a["x"], b["x"])
                y0 = min(a["y"], b["y"])
                x1 = max(a["x"] + a["w"], b["x"] + b["w"])
                y1 = max(a["y"] + a["h"], b["y"] + b["h"])

                extra.append({
                    "text": joined,
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "conf": min(a["conf"], b["conf"]),
                    "label": lab,
                    "prob": 1.0,
                    "source": "rule+join"
                })

                break

    return results + extra


def draw_boxes(img, results):
    """Copy of the image with a color-coded box and a label for every flagged piece."""
    out = img.copy()

    for it in results:
        if it["label"] == "safe":
            continue

        c = COLORS[it["label"]]

        x, y, w, h = it["x"], it["y"], it["w"], it["h"]

        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            c,
            3
        )

        cv2.putText(
            out,
            f'{it["label"]} {it["prob"]:.0%}',
            (x, max(y - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            c,
            2
        )

    return out


def analyze(img, threshold=THRESHOLD, scale=2.0, ocr_items=None):
    """OCR + classify + draw. Pass ocr_items to skip the (slow) OCR step."""
    if ocr_items is None:
        from ocr_module import ocr_image
        ocr_items = ocr_image(img, scale=scale)

    results = classify(ocr_items, threshold)

    return {
        "ocr_items": ocr_items,
        "results": results,
        "flagged": [r for r in results if r["label"] != "safe"],
        "annotated": draw_boxes(img, results)
    }