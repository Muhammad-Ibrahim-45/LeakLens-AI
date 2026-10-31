import cv2
import easyocr

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader

def merge_same_line(items, gap_factor=2.0):
    """Join OCR pieces that sit on the same text line and are close together."""
    items = sorted(items, key=lambda i: i["x"])          # left to right
    lines = []
    for it in items:
        cy = it["y"] + it["h"] / 2
        for ln in lines:
            lcy = ln["y"] + ln["h"] / 2
            hmax = max(it["h"], ln["h"])
            gap = it["x"] - (ln["x"] + ln["w"])
            if abs(cy - lcy) < 0.6 * hmax and -0.5 * hmax < gap < gap_factor * hmax:
                x2 = max(ln["x"] + ln["w"], it["x"] + it["w"])
                y1 = min(ln["y"], it["y"])
                y2 = max(ln["y"] + ln["h"], it["y"] + it["h"])
                ln["text"] += " " + it["text"]
                ln["x"], ln["y"] = min(ln["x"], it["x"]), y1
                ln["w"], ln["h"] = x2 - ln["x"], y2 - y1
                ln["conf"] = min(ln["conf"], it["conf"])
                break
        else:
            lines.append(dict(it))
    return sorted(lines, key=lambda i: (i["y"], i["x"]))

def ocr_image(img, scale=2.0, merge_lines=True):
    """img: OpenCV BGR image. Returns a list of {text, x, y, w, h, conf}."""
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = get_reader().readtext(big, width_ths=1.0)
    items = []
    for box, text, conf in results:
        xs = [p[0] / scale for p in box]
        ys = [p[1] / scale for p in box]
        x, y = int(min(xs)), int(min(ys))
        items.append({"text": text, "x": x, "y": y,
                      "w": int(max(xs) - x), "h": int(max(ys) - y),
                      "conf": round(float(conf), 3)})
    return merge_same_line(items) if merge_lines else items
