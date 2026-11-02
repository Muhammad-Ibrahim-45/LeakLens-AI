# import cv2
# import easyocr

# _reader = None

# def get_reader():
#     global _reader
#     if _reader is None:
#         _reader = easyocr.Reader(["en"], gpu=False)
#     return _reader

# def merge_same_line(items, gap_factor=2.0):
#     """Join OCR pieces that sit on the same text line and are close together."""
#     items = sorted(items, key=lambda i: i["x"])          # left to right
#     lines = []
#     for it in items:
#         cy = it["y"] + it["h"] / 2
#         for ln in lines:
#             lcy = ln["y"] + ln["h"] / 2
#             hmax = max(it["h"], ln["h"])
#             gap = it["x"] - (ln["x"] + ln["w"])
#             if abs(cy - lcy) < 0.6 * hmax and -0.5 * hmax < gap < gap_factor * hmax:
#                 x2 = max(ln["x"] + ln["w"], it["x"] + it["w"])
#                 y1 = min(ln["y"], it["y"])
#                 y2 = max(ln["y"] + ln["h"], it["y"] + it["h"])
#                 ln["text"] += " " + it["text"]
#                 ln["x"], ln["y"] = min(ln["x"], it["x"]), y1
#                 ln["w"], ln["h"] = x2 - ln["x"], y2 - y1
#                 ln["conf"] = min(ln["conf"], it["conf"])
#                 break
#         else:
#             lines.append(dict(it))
#     return sorted(lines, key=lambda i: (i["y"], i["x"]))

# def ocr_image(img, scale=2.0, merge_lines=True):
#     """img: OpenCV BGR image. Returns a list of {text, x, y, w, h, conf}."""
#     big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#     results = get_reader().readtext(big, width_ths=1.0)
#     items = []
#     for box, text, conf in results:
#         xs = [p[0] / scale for p in box]
#         ys = [p[1] / scale for p in box]
#         x, y = int(min(xs)), int(min(ys))
#         items.append({"text": text, "x": x, "y": y,
#                       "w": int(max(xs) - x), "h": int(max(ys) - y),
#                       "conf": round(float(conf), 3)})
#     return merge_same_line(items) if merge_lines else items

"""LeakLens OCR: reads text and its position from a screenshot with EasyOCR.

    ocr_image(img, scale=2.0)  -> list of {text, x, y, w, h, conf}   (x, y, w, h in ORIGINAL image pixels)

Memory safety: EasyOCR's text finder needs gigabytes of RAM on a big image.  The image sent to it is
therefore capped at MAX_SIDE pixels on its longest side, and if the computer still runs out of memory the
image is retried at smaller sizes (FALLBACK_SIDES) before giving up with a readable message.
"""
import gc

import cv2
import easyocr

MAX_SIDE = 2200                      # longest side (pixels) sent to OCR; lower = less memory, less detail
FALLBACK_SIDES = (1600, 1200, 900)   # tried in this order if memory runs out

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


def _out_of_memory(err):
    msg = str(err).lower()
    return isinstance(err, MemoryError) or "not enough memory" in msg or "out of memory" in msg or "alloc" in msg


def ocr_image(img, scale=2.0, merge_lines=True, max_side=MAX_SIDE):
    """img: OpenCV BGR image. Returns a list of {text, x, y, w, h, conf} in the original image's pixels."""
    h0, w0 = img.shape[:2]
    longest = max(h0, w0)
    sides = [max_side] + [s for s in FALLBACK_SIDES if s < max_side]

    results, eff, last_err = None, scale, None
    for side in sides:
        eff = min(scale, side / longest)                  # never send more than `side` pixels on the long edge
        interp = cv2.INTER_CUBIC if eff >= 1 else cv2.INTER_AREA
        big = cv2.resize(img, None, fx=eff, fy=eff, interpolation=interp)
        try:
            results = get_reader().readtext(big, width_ths=1.0, canvas_size=side)
            break
        except (RuntimeError, MemoryError) as err:
            if not _out_of_memory(err):
                raise
            last_err = err
            del big
            gc.collect()
    if results is None:
        raise RuntimeError("Not enough memory to read this image. Close other programs "
                           "(including notebooks that have EasyOCR loaded) or use a smaller screenshot.") from last_err

    items = []
    for box, text, conf in results:
        xs = [p[0] / eff for p in box]
        ys = [p[1] / eff for p in box]
        x, y = int(min(xs)), int(min(ys))
        items.append({"text": text, "x": x, "y": y,
                      "w": int(max(xs) - x), "h": int(max(ys) - y),
                      "conf": round(float(conf), 3)})
    return merge_same_line(items) if merge_lines else items