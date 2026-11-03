"""LeakLens detector (version 2).

One file, used by BOTH the notebook and the Streamlit app, so the logic lives in one place.
    classify(items, threshold)  -> labels each OCR piece (rules first, then the model with sanity checks)
    draw_boxes(img, results)    -> copy of the image with color-coded boxes
    analyze(img, ...)           -> OCR + classify + draw in one call

Labels: email, phone, url_internal, credential   (+ id_number when DETECT_ID_NUMBERS is True)
The `source` of every detection says who decided:  "model", "rule:<rule name>", or "rule+join".
"""
import os
import re
import sys

import cv2
import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODEL_PATH = os.path.join(HERE, "leaklens_logreg.joblib")

# =====================================================================================
# 1. SETTINGS AND WORD LISTS  (the part you are expected to edit)
# =====================================================================================
THRESHOLD = 0.40            # default sensitivity of the model: lower catches more, higher gives fewer false alarms

# A private IP with no scheme/port/path (e.g. "192.168.1.132") is flagged as an internal URL only when the
# rest of the OCR piece has at most this many words of 3+ letters.  0 = only when the IP is alone,
# None = never.  "10.20.30.45 - Remote Desktop Connection" has 3 words, so it stays safe at 2.
BARE_IP_MAX_WORDS = 2

DETECT_ID_NUMBERS = True    # card numbers (Luhn check), Pakistani CNIC, IBAN (checksum), US SSN  -> label "id_number"
DETECT_OTP_PIN = True       # "OTP: 482913", "verification code 123456", "PIN: 4821"             -> label "credential"

# Regexes for text that must NEVER be flagged, e.g. a public support address:  [r"support@mycompany\.com"]
IGNORE_TEXT = []

# Domains whose addresses must always be recognised, even when OCR turned the "@" into a letter.
# If OCR read the "@" as a LETTER, only domains whose main name has 5+ letters can be recognised (neduet, gmail, yahoo...);
# short ones (att.net, nu.edu.pk) still work when the "@" shows up as (c), (a) or with spaces.
# Do not list a domain that is only the END of another listed domain (uet.edu.pk inside neduet.edu.pk):
# the guard further down drops such entries automatically.
KNOWN_EMAIL_DOMAINS = [
    # --- NED University ---
    "cloud.neduet.edu.pk", "neduet.edu.pk",
    # --- other Pakistani universities ---
    "khi.iba.edu.pk", "iba.edu.pk", "nu.edu.pk", "ssuet.edu.pk", "szabist.edu.pk", "uok.edu.pk", "aku.edu",
    "habib.edu.pk", "duhs.edu.pk", "dsu.edu.pk", "pafkiet.edu.pk", "iqra.edu.pk", "hamdard.edu.pk",
    "bahria.edu.pk", "comsats.edu.pk", "nust.edu.pk", "giki.edu.pk", "pu.edu.pk", "lums.edu.pk",
    "itu.edu.pk", "numl.edu.pk", "qau.edu.pk", "iiu.edu.pk", "riphah.edu.pk", "umt.edu.pk",
    "bzu.edu.pk", "muet.edu.pk",
    # --- Pakistani internet providers ---
    "ptcl.net.pk", "nayatel.com",
    # --- Google, Microsoft, Apple ---
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "hotmail.co.uk", "icloud.com",
    # --- Yahoo and other big providers ---
    "yahoo.com", "yahoo.co.uk", "yahoo.co.in", "ymail.com", "rocketmail.com", "aol.com",
    "zoho.com", "gmx.com", "gmx.net", "yandex.com", "yandex.ru", "rediffmail.com", "fastmail.com",
    # --- privacy-focused providers ---
    "protonmail.com", "proton.me", "tutanota.com", "tuta.com",
    # --- internet providers abroad ---
    "comcast.net", "verizon.net", "att.net", "sbcglobal.net", "btinternet.com", "web.de", "t-online.de", "orange.fr",
    # --- add your own company or university domains here, for example: "mycompany.com"
]

# Host endings that only exist inside a private network ("/" or ":" is needed after .local, because of files like .env.local)
INTERNAL_TLDS = ["internal", "corp", "lan", "intranet", "localdomain", "home.arpa", "cluster.local"]
# Host names that start with one of these words are treated as internal even on a public domain (https://staging.acme.com)
INTERNAL_HOST_PREFIXES = ["staging", "stage", "dev", "uat", "qa", "preprod", "intranet", "admin", "jenkins",
                          "grafana", "kibana", "prometheus", "argocd", "vpn", "bastion", "nexus", "artifactory",
                          "sonarqube"]
# Address schemes that can point at an internal machine
URL_SCHEMES = ["https?", "ftps?", "sftp", "ssh", "smb", "rdp", "vnc", "ldaps?", "mysql", "postgres(?:ql)?",
               "mongodb(?:\\+srv)?", "redis", "amqps?"]
# Words that, followed by ":" or "=" and a value, mark the value as a secret
CREDENTIAL_WORDS = ["pass(?:word|wd|code|phrase)?", "pwd", "secret", "token", "api[_-]?key", "access[_-]?key",
                    "private[_-]?key", "signing[_-]?key", "encryption[_-]?key", "master[_-]?key",
                    "session[_-]?(?:id|token|key)", "(?:php|j)?sessid"]

COLORS = {
    "email": (255, 100, 0), 
    "phone": (0, 140, 255), 
    "url_internal": (200, 0, 160),
    "credential": (0, 0, 230), 
    "id_number": (0, 170, 0)
    # Add any missing labels from your LABEL_NAMES here:
   # "your_missing_label_name": (100, 100, 100) ############################################################################
}
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


# =====================================================================================
# 2. PATTERNS
# =====================================================================================
DATE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")
PHONE_PART = re.compile(r"(?<![\w$-])\+?\(?\d[\d\s().-]{6,}\d(?![\w])")
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
ID_LIKE = re.compile(r"[A-Z]{2,6}-\d+(?:-\d+)*")
# the domain must end in letters, so "search-api@1.42.0" is not an email
EMAIL = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")

# private IPv4 ranges: 10.x.x.x, 192.168.x.x, 172.16-31.x.x, 127.x.x.x   (ONE group of four alternatives)
_PRIV = (r"(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
         r"|192\.168\.\d{1,3}\.\d{1,3}"
         r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
         r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3})")
_NOT_MORE_DIGITS = r"(?!\.?\d)"
_SCHEME = r"(?:" + "|".join(URL_SCHEMES) + r")"
_TLDS = "|".join(re.escape(t) for t in INTERNAL_TLDS)
_PREFIXES = "|".join(INTERNAL_HOST_PREFIXES)

# what counts as an INTERNAL URL: scheme + private/internal host, internal host names,
# a private IP followed by a port or a path, or a Windows network share. A bare IP alone is handled separately.
INTERNAL_URL = re.compile(
    r"(?i)(?:" + _SCHEME + r"://(?:localhost|" + _PRIV + r")" + _NOT_MORE_DIGITS + r")"
    r"|(?:" + _SCHEME + r"://[\w.-]+\.(?:local|" + _TLDS + r")\b)"
    r"|(?:\b[\w-]+(?:\.[\w-]+)*\.(?:" + _TLDS + r")\b)"
    r"|(?:\b[\w-]+(?:\.[\w-]+)*\.local(?=[:/]))"
    r"|(?:(?<![\d.])" + _PRIV + r"(?=:\d{2,5}|/\S))"
    r"|(?:\blocalhost(?=:\d))"
    r"|(?:" + _SCHEME + r"://(?:" + _PREFIXES + r")\.)"
    r"|(?:\bhttps?\s*:\s*[/\\Il|]{0,3}\s*(?:[\w-]+[. ]){1,4}(?:corp|internal|lan|local))"
    r"|(?:\bhttps?\s*:\s*[/\\Il|]{0,3}\s*(?:" + _PREFIXES + r")[. ])"
    r"|(?:\\\\[\w.-]+\\[\w$.-]+)")

BARE_IP = re.compile(r"(?<![\d.])" + _PRIV + _NOT_MORE_DIGITS)


def _usable_domains(domains):
    """Drop a domain that is only the END of another listed domain (e.g. uet.edu.pk inside neduet.edu.pk):
    without an '@' the two cannot be told apart, so the shorter one would cause false alarms."""
    ds = list(dict.fromkeys(d.lower() for d in domains))
    return [d for d in ds
            if not any(e != d and e.endswith(d) and not e[:-len(d)].endswith(".") for e in ds)]


def _core_length(domain):
    """Length of the main name of a domain: neduet.edu.pk -> 6 (neduet), gmail.com -> 5, att.net -> 3."""
    d = re.sub(r"\.(?:edu|com|net|org|co|gov)\.(?:pk|uk|in|au)$|\.[a-z]+$", "", domain.lower())
    return max(len(label) for label in d.split("."))


def _alt(domains):
    return "|".join(re.escape(d) for d in sorted(domains, key=len, reverse=True))


_LOCAL = r"(?i)(?<![\w.+-])[\w+-][\w.+-]*[A-Za-z0-9]"
# (a) the "@" was read as (c), (a) or spaced: every listed domain is safe to use, even short ones
_KNOWN_EMAIL_MARKED = re.compile(_LOCAL + r"(?:\s?[@\u00a9]\s?|\(a\))(?:" + _alt(KNOWN_EMAIL_DOMAINS) + r")\b")
# (b) the "@" was read as a LETTER, so the local part touches the domain: only domains with a main name of
#     5+ letters, otherwise "wyatt.net" would look like an address at att.net
_KNOWN_EMAIL_BARE = re.compile(_LOCAL + r"(?:" + _alt([d for d in _usable_domains(KNOWN_EMAIL_DOMAINS)
                                                       if _core_length(d) >= 5]) + r")\b")

# the value after "password:" must contain a digit, or a letter plus a symbol ("********" is a mask, not a secret)
_VALUE = r"['\"]?(?:(?=[^\s'\"]*\d)|(?=[^\s'\"]*[A-Za-z])(?=[^\s'\"]*[!@#$%^&*]))\S{4,}"
_CRED_WORDS = "|".join(CREDENTIAL_WORDS)

# (label, name, pattern).  The name appears in the table as  source = "rule:<name>"
RULES = [
    # ---- secrets with a fixed format -------------------------------------------------
    ("credential", "aws_access_key", re.compile(r"AKIA[0-9A-Z]{12,20}")),
    ("credential", "github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("credential", "gitlab_token", re.compile(r"glpat-[A-Za-z0-9_-]{16,}")),
    ("credential", "sk_key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")),
    ("credential", "stripe_key", re.compile(r"(?<![A-Za-z0-9])(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{12,}")),
    ("credential", "slack_token", re.compile(r"xox[abeprs]-[A-Za-z0-9-]{10,}")),
    ("credential", "google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{30,}")),
    ("credential", "sendgrid_key", re.compile(r"SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}")),
    ("credential", "service_token", re.compile(r"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{30,}|npm_[A-Za-z0-9]{30,}"
                                               r"|shp(?:at|ca|pa|ss)_[a-f0-9]{24,})")),
    ("credential", "telegram_bot", re.compile(r"(?<!\d)\d{8,10}:[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])")),
    ("credential", "jwt", re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("credential", "private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("credential", "bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{15,}")),
    ("credential", "basic_auth_header", re.compile(r"(?i)authorization\W{0,3}basic\s+[A-Za-z0-9+/=]{12,}")),
    ("credential", "password_in_url", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]{3,}@[^\s/]+")),
    ("credential", "azure_account_key", re.compile(r"AccountKey=[A-Za-z0-9+/=]{20,}")),
    ("credential", "secret_in_url", re.compile(r"(?i)[?&](?:sig|signature|token|access_token|api_?key|key|secret|password|auth)="
                                               r"[A-Za-z0-9%._~+/-]{8,}")),
    # ---- "name = value" lines ----------------------------------------------------------
    ("credential", "keyword_value", re.compile(r"(?i)(?:" + _CRED_WORDS + r")\w*\s*[:=]\s*" + _VALUE)),
    ("credential", "password_is", re.compile(r"(?i)\b(?:password|passwd|pwd)\s+(?:is\s+)?['\"]?(?=\S*\d)\S{6,}")),
    # ---- internal addresses -------------------------------------------------------------
    ("url_internal", "internal_url", INTERNAL_URL),
    # ---- emails -----------------------------------------------------------------------
    ("email", "email", EMAIL),
    ("email", "known_domain_email", _KNOWN_EMAIL_MARKED),
    ("email", "known_domain_email", _KNOWN_EMAIL_BARE),
    ("email", "email_after_label", re.compile(
        r"(?i)\b\w*e-?mail\w*\s*[:=]\s*['\"]?[\w.+-]{2,}\."
        r"(?:com|org|net|edu|gov|io|co|pk|uk|de|in|us|info|biz|me|app|dev|ai|ca|au|fr|it|es|nl|ru|cn|jp|br"
        r"|corp|internal|local|lan)\b")),
]
if DETECT_OTP_PIN:
    RULES += [
        ("credential", "otp_code", re.compile(
            r"(?i)\b(?:otp|one[- ]time (?:pass(?:word|code)|code)|(?:verification|security|confirmation|login|"
            r"authentication|auth|2fa) code)\b\D{0,25}?(?<![\d-])\d{4,8}(?![\d-])")),
        ("credential", "otp_code_first", re.compile(
            r"(?i)(?<![\d-])\d{4,8}(?![\d-])\s*(?:is|as)\s+(?:your|the)\s+(?:otp|(?:one[- ]time\s+)?(?:pass(?:word|code)|code)|"
            r"(?:verification|security|confirmation|login|authentication|auth|2fa)\s+code)\b")),
        ("credential", "pin_code", re.compile(r"(?i)\bpin\b\s*[:=#]\s*\d{4,6}(?!\d)")),
    ]

# ---- ID numbers (checked with real checksums, so random digits rarely pass) --------------
CARD_CAND = re.compile(r"(?<![\d-])(?:\d[ -]?){14,15}\d(?![\d-])")
# Visa 16, Mastercard 16, Amex 15, Discover 16, JCB 16 (other lengths are mostly barcodes and order numbers)
CARD_PREFIX = re.compile(r"4\d{15}|5[1-5]\d{14}|2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)\d{12}|3[47]\d{13}"
                         r"|6011\d{12}|64[4-9]\d{13}|65\d{14}|35(?:2[89]|[3-8]\d)\d{12}")
# a long number written after one of these words is an order / tracking / product number, not a card
_NOT_A_CARD = re.compile(r"(?i)\b(?:ref|no|number|order|invoice|inv|tracking|track|id|sku|ticket|case|txn|transaction|"
                         r"serial|barcode|ean|isbn|upc|awb|consignment|shipment)\b[.#:\s-]*$")
CNIC = re.compile(r"(?<!\d)[1-7]\d{4}\s?[-\u2013\u2014]\s?\d{7}\s?[-\u2013\u2014]\s?\d(?!\d)")
SSN = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
IBAN_CAND = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b")


def _luhn(digits):
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(text):
    s = re.sub(r"\s", "", text)
    if not 15 <= len(s) <= 34:
        return False
    return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def _id_number(text):
    """Name of the ID-number rule that matches, or None."""
    for m in CARD_CAND.finditer(text):
        d = re.sub(r"\D", "", m.group())
        if CARD_PREFIX.fullmatch(d) and _luhn(d) and not _NOT_A_CARD.search(text[:m.start()]):
            return "card_number"
    if CNIC.search(text):
        return "cnic"
    if SSN.search(text):
        return "ssn"
    for m in IBAN_CAND.finditer(text):
        if _iban_ok(m.group()):
            return "iban"
    return None


# ---- phone numbers with a fixed or labelled format ---------------------------------------
PK_MOBILE = re.compile(r"(?<![\d+])(?:\+92|0092|92|0)[-\s]?3(?:[0-4]\d|55)[-\s]?\d{7}(?!\d)")
INTL_PHONE = re.compile(r"(?<![\w+])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)")
LABELLED_PHONE = re.compile(r"(?i)\b(?:phone|tel|telephone|mobile|mob|cell|whatsapp|contact|call|fax)\b"
                            r"[^0-9+(\n]{0,12}(\+?\(?\d[\d\s().-]{6,}\d)")


def _phone_rule(text):
    for name, pat in (("phone_pk_mobile", PK_MOBILE), ("phone_international", INTL_PHONE)):
        for m in pat.finditer(text):
            if phone_ok(m.group()):
                return name
    for m in LABELLED_PHONE.finditer(text):
        part = m.group(1).strip()
        if re.fullmatch(r"\d+", part) and len(part) < 10:      # a plain 8-9 digit number after "contact" is not enough
            continue
        if phone_ok(part):
            return "phone_label"
    return None


# =====================================================================================
# 3. CLEANING AND SANITY CHECKS
# =====================================================================================
def normalize(t):
    """Repair common OCR mistakes and spelled-out addresses before classifying."""
    t = t.strip()
    t = re.sub(r"(?i)\s*[\[(]\s*at\s*[\])]\s*", "@", t)                             # john [at] gmail.com -> john@gmail.com
    t = re.sub(r"(?i)\s*[\[(]\s*dot\s*[\])]\s*", ".", t)                            # gmail [dot] com -> gmail.com
    t = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", t)                                    # 10 .0.4.21 -> 10.0.4.21
    t = re.sub(r"(?i)\b(https?)\s*:\s*/\s*/\s*", r"\1://", t)                       # http: // -> http://
    t = re.sub(r"\b((?i:https?)):(?!//)(?:/|[Il|]{2}|/[Il|]|[Il|]/)(?=[A-Za-z0-9])", r"\1://", t)   # https:ll -> https://
    t = re.sub(r"(?i)\s*\.\s*(?=(?:internal|corp|lan|local)\b)", ".", t)            # host . internal -> host.internal
    t = re.sub(r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*)-(?=\S)", r"\1=", t)                  # KEY-value -> KEY=value
    dom = r"(?=[A-Za-z0-9-]+\s*[.,]\s*[A-Za-z]{2,})"
    t = re.sub(r"(\w)\s*(?:[\u00a9@]|\(a\))\s*" + dom, r"\1@", t)                   # john (c) gmail.com -> john@gmail.com
    for _ in range(3):                                                              # "@gmail , com" -> "@gmail.com"
        t = re.sub(r"(@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)\s*[.,]\s*([A-Za-z]{2,})\b", r"\1.\2", t)
    return t


def phone_ok(text):
    """7-15 digits, not a date, not an IP address."""
    for m in PHONE_PART.finditer(text):
        part = m.group().strip()
        digits = sum(c.isdigit() for c in part)
        if IPV4.search(part):
            continue
        if re.fullmatch(r"\d+", part) and not 10 <= digits <= 13:      # plain 7-9 or 14-15 digits: an ID, not a phone
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
        if re.search(r"@v?\d+(?:\.\d+)+$", t) or re.fullmatch(r"v?\d+(?:\.\d+)+", t):   # version strings
            continue
        if (len(t) >= 6 and not ID_LIKE.fullmatch(t)
                and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)):
            return True
    return False


def _ignored(text):
    return any(re.search(p, text, re.I) for p in IGNORE_TEXT)


# =====================================================================================
# 4. MAIN API
# =====================================================================================
def rule_match(text):
    """Fixed-format rules, checked BEFORE the model. Returns (label, rule name) or None."""
    for label, name, pat in RULES:
        if pat.search(text):
            return label, name
    if DETECT_ID_NUMBERS:
        name = _id_number(text)
        if name:
            return "id_number", name
    name = _phone_rule(text)
    if name:
        return "phone", name
    if BARE_IP_MAX_WORDS is not None:
        m = BARE_IP.search(text)
        if m and len(re.findall(r"[A-Za-z]{3,}", text[:m.start()] + " " + text[m.end():])) <= BARE_IP_MAX_WORDS:
            return "url_internal", "bare_ip"
    return None


def rule_label(text):
    """Same as rule_match but returns only the label (kept for older code)."""
    hit = rule_match(text)
    return hit[0] if hit else None


def classify(items, threshold=THRESHOLD):
    """Return a copy of the OCR items with label, prob and source ('model', 'rule:<name>' or 'rule+join').

    Order of decisions for every OCR piece:
      1. a fixed-format rule matches (email, internal URL, key formats, ID numbers, labelled phones) -> that label wins
      2. otherwise the model's sensitivity (1 - P(safe)) must reach the threshold,
         and its answer must pass the sanity check for its type
      3. text matching IGNORE_TEXT is never flagged
    """
    items = [dict(i) for i in items]
    if not items:
        return items
    model = get_model()
    classes = [str(c) for c in model.classes_]
    safe = classes.index("safe")
    probs = model.predict_proba([normalize(i["text"]) for i in items])
    for it, p in zip(items, probs):
        t = normalize(it["text"])
        p_sens = 1 - p[safe]
        others = p.copy()
        others[safe] = -1
        label, prob, src = "safe", round(float(1 - p_sens), 3), "-"
        hit = rule_match(t)
        if hit:
            label, prob, src = hit[0], 1.0, "rule:" + hit[1]
        elif p_sens >= threshold:
            lab = classes[int(np.argmax(others))]
            if lab == "phone":
                ok = phone_ok(t)
            elif lab == "credential":
                ok = credential_ok(t)
            elif lab == "email":
                ok = bool(EMAIL.search(t)) or (p_sens >= 0.80 and bool(re.search(r"\w@[A-Za-z]", t)))
            elif lab == "url_internal":
                ok = bool(INTERNAL_URL.search(t))
            else:
                ok = True
            if ok:
                label, prob, src = lab, round(float(p_sens), 3), "model"
        if label != "safe" and _ignored(t):
            label, src = "safe", "ignored"
        it["label"], it["prob"], it["source"] = label, prob, src
    return _stitch(items)


def _stitch(results):
    """Second pass, rules only: join neighbouring pieces on the same text line.
    Catches things OCR cut in two, e.g. 'SUPPORT_EMAIL=ali.khan' + '@example.com'."""
    n = len(results)
    if n < 2:
        return results
    cy = [r["y"] + r["h"] / 2 for r in results]
    order = sorted(range(n), key=lambda i: cy[i])          # top to bottom, so only nearby rows are compared
    max_h = max(max(r["h"], 1) for r in results)
    extra = []
    for pos, i in enumerate(order):
        a = results[i]
        best = None
        for step in (-1, 1):                               # walk up and down the sorted list
            k = pos + step
            while 0 <= k < n and abs(cy[order[k]] - cy[i]) < 0.6 * max_h:
                b = results[order[k]]
                h = max(a["h"], b["h"], 1)
                gap = b["x"] - (a["x"] + a["w"])
                if abs(cy[order[k]] - cy[i]) < 0.6 * h and -0.3 * h <= gap < 4.0 * h and (best is None or gap < best[0]):
                    best = (gap, b)
                k += step
        if best is None:
            continue
        gap, b = best
        if a["label"] != "safe" and b["label"] != "safe":
            continue
        h = max(a["h"], b["h"], 1)
        ta, tb = normalize(a["text"]), normalize(b["text"])
        joined = normalize(a["text"] + ("" if gap < 0.3 * h else " ") + b["text"])
        hit = rule_match(joined)
        if hit and not rule_match(ta) and not rule_match(tb) and not _ignored(joined):
            x0, y0 = min(a["x"], b["x"]), min(a["y"], b["y"])
            x1 = max(a["x"] + a["w"], b["x"] + b["w"])
            y1 = max(a["y"] + a["h"], b["y"] + b["h"])
            extra.append({"text": joined, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                          "conf": min(a["conf"], b["conf"]), "label": hit[0], "prob": 1.0,
                          "source": "rule+join"})
    return results + extra


def draw_boxes(img, results):
    """Copy of the image with a color-coded box and a label for every flagged piece."""
    out = img.copy()
    for it in results:
        if it["label"] == "safe":
            continue
        c = COLORS[it["label"]]
        x, y, w, h = it["x"], it["y"], it["w"], it["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 3)
        cv2.putText(out, f'{it["label"]} {it["prob"]:.0%}', (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    return out


def analyze(img, threshold=THRESHOLD, scale=2.0, ocr_items=None):
    """OCR + classify + draw. Pass ocr_items to skip the (slow) OCR step."""
    if ocr_items is None:
        from ocr_module import ocr_image
        ocr_items = ocr_image(img, scale=scale)
    results = classify(ocr_items, threshold)
    return {"ocr_items": ocr_items,
            "results": results,
            "flagged": [r for r in results if r["label"] != "safe"],
            "annotated": draw_boxes(img, results)}


# =====================================================================================
# 5. HIDING THE SENSITIVE AREAS
# =====================================================================================
def redact(img, results, style="blur", margin=6):
    """Copy of the image with every flagged region hidden.

    style : "blur" (strong Gaussian blur), "pixelate" (big coarse blocks) or "black" (solid box, the safest)
    margin: extra pixels around each box, because OCR boxes are never perfectly tight
    """
    out = img.copy()
    height, width = out.shape[:2]
    for r in results:
        if r["label"] == "safe":
            continue
        x0, y0 = max(int(r["x"]) - margin, 0), max(int(r["y"]) - margin, 0)
        x1, y1 = min(int(r["x"] + r["w"]) + margin, width), min(int(r["y"] + r["h"]) + margin, height)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]                               # a view: changing it changes `out`
        h, w = roi.shape[:2]
        if style == "black":
            roi[:] = 0
        elif style == "pixelate":
            block = max(10, h // 2)
            small = cv2.resize(roi, (max(w // block, 1), max(h // block, 1)), interpolation=cv2.INTER_AREA)
            roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            k = max(31, h | 1)                                # kernel at least as tall as the text, and odd
            k = k if k % 2 else k + 1
            for _ in range(2):                                # two passes make the text unrecoverable
                roi[:] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


# =====================================================================================
# 6. RISK SCORE
# =====================================================================================
# Every type has a base risk. The worst finding sets the level, and each extra finding adds a little (up to 25).
RISK_BASE = {"credential": 70, "id_number": 70, "url_internal": 40, "phone": 35, "email": 15}
RISK_WORDS = {"credential": ("credential", "credentials"), "id_number": ("ID number", "ID numbers"),
              "url_internal": ("internal URL", "internal URLs"), "phone": ("phone number", "phone numbers"),
              "email": ("email", "emails")}
RISK_ADVICE = {"High": "Do not share this screenshot until the sensitive areas are hidden.",
               "Medium": "Check every item before sharing, or hide them.",
               "Low": "Low risk, but consider hiding these items.",
               "None": "Nothing sensitive was found. Look over the image once before sharing."}


def risk_assessment(flagged):
    """Turn the flagged items into one verdict: level (High/Medium/Low/None), score 0-100, headline, advice, counts."""
    flagged = [r for r in flagged if r["label"] in RISK_BASE]
    if not flagged:
        return {"level": "None", "score": 0, "headline": "No sensitive data found",
                "advice": RISK_ADVICE["None"], "counts": {}}
    counts = {}
    for r in flagged:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    score = min(100, max(RISK_BASE[k] for k in counts) + min(25, 3 * (len(flagged) - 1)))
    level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    parts = [f"{n} {RISK_WORDS[k][0 if n == 1 else 1]}"
             for k, n in sorted(counts.items(), key=lambda kv: -RISK_BASE[kv[0]])]
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return {"level": level, "score": score, "headline": f"{level} risk: {listed} found",
            "advice": RISK_ADVICE[level], "counts": counts}