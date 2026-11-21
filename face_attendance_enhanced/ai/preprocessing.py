"""Image decoding, face cropping, normalisation and quality checks."""
import base64
import re

import cv2
import numpy as np

_DATA_URL = re.compile(r"^data:image/(jpeg|jpg|png);base64,", re.I)


def decode_image_bytes(data, max_dim=6000):
    """Bytes -> BGR image, or None when the data is not a valid image."""
    if not data:
        return None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        return None
    return img


def decode_data_url(value, max_chars=8_000_000, max_dim=6000):
    """Browser canvas data URL -> BGR image, or None."""
    if not isinstance(value, str) or len(value) > max_chars:
        return None
    m = _DATA_URL.match(value)
    if not m:
        return None
    try:
        raw = base64.b64decode(value[m.end():], validate=True)
    except Exception:
        return None
    return decode_image_bytes(raw, max_dim)


def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_face(gray, box):
    x, y, w, h = [int(v) for v in box]
    H, W = gray.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    return gray[y0:y1, x0:x1]


def normalize_face(gray_crop, size=(200, 200), equalize=False):
    """Resize to a fixed size; optionally apply histogram equalisation."""
    if gray_crop is None or gray_crop.size == 0:
        raise ValueError("Empty face crop")
    face = cv2.resize(gray_crop, tuple(size), interpolation=cv2.INTER_AREA)
    if equalize:
        face = cv2.equalizeHist(face)
    return face


def check_quality(gray, box, min_face=90, min_sharpness=40.0, min_brightness=50.0, max_brightness=210.0):
    """Returns (ok, reasons, metrics) for a detected face region."""
    crop = crop_face(gray, box)
    reasons = []
    if crop.size == 0:
        return False, ["Face region is empty."], {}
    h, w = crop.shape[:2]
    sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    bright = float(crop.mean())
    if min(h, w) < min_face:
        reasons.append(f"Face too small ({min(h, w)}px; need at least {min_face}px). Move closer.")
    if sharp < min_sharpness:
        reasons.append("Image looks blurry. Hold still and improve focus.")
    if bright < min_brightness:
        reasons.append("Face is too dark. Add light in front of you.")
    if bright > max_brightness:
        reasons.append("Face is overexposed. Reduce direct light.")
    return (not reasons), reasons, {"sharpness": round(sharp, 1), "brightness": round(bright, 1), "size": min(h, w)}
