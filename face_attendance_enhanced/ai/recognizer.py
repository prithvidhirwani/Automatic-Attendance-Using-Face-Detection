"""LBPH recognition with an explicit 'Unknown' decision."""
import json
import threading
from pathlib import Path

import cv2

from ai.preprocessing import crop_face, normalize_face, to_gray

UNKNOWN = "Unknown"


class ModelNotReady(RuntimeError):
    """Raised when recognition is requested but no usable model is loaded."""


class Recognizer:
    def __init__(self, model_path, label_map_path, threshold=70.0, face_size=(200, 200)):
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.threshold = float(threshold)
        self.face_size = tuple(face_size)
        self.equalize = False
        self.meta = {}
        self.error = None
        self._model = None
        self._labels = set()
        self._lock = threading.Lock()

    @property
    def ready(self):
        return self._model is not None

    def load(self):
        """Load model + label map from disk. Never raises; sets self.error instead."""
        with self._lock:
            self._model, self._labels, self.meta, self.error = None, set(), {}, None
            if not self.model_path.exists() or not self.label_map_path.exists():
                self.error = "No trained model found. Register students, capture images, then train."
                return False
            try:
                if not hasattr(cv2, "face"):
                    raise RuntimeError("cv2.face missing - install opencv-contrib-python, not opencv-python.")
                meta = json.loads(self.label_map_path.read_text(encoding="utf-8"))
                labels = {int(x) for x in meta["labels"]}
                if not labels:
                    raise ValueError("label map has no identities")
                model = cv2.face.LBPHFaceRecognizer_create()
                model.read(str(self.model_path))
            except Exception as exc:  # corrupted / incompatible files
                self.error = f"Model files could not be loaded ({exc}). Retrain the model."
                return False
            self._model, self._labels, self.meta = model, labels, meta
            self.equalize = bool(meta.get("equalize", False))
            return True

    def predict_face(self, face_gray):
        """face_gray: already-normalised 2-D image. Returns (student_id | None, distance)."""
        if not self.ready:
            raise ModelNotReady(self.error or "Model not loaded.")
        with self._lock:
            label, distance = self._model.predict(face_gray)
        distance = float(distance)
        if label not in self._labels or distance != distance:  # unknown label or NaN
            return None, distance
        if distance <= self.threshold:
            return int(label), distance
        return None, distance

    def recognize(self, image, boxes):
        """Recognise each box. Returns one dict per box: box, student_id (None = Unknown), distance."""
        gray = to_gray(image)
        results = []
        for box in boxes:
            crop = crop_face(gray, box)
            if crop.size == 0:
                results.append({"box": tuple(box), "student_id": None, "distance": None})
                continue
            face = normalize_face(crop, self.face_size, self.equalize)
            sid, dist = self.predict_face(face)
            results.append({"box": tuple(int(v) for v in box), "student_id": sid, "distance": dist})
        return results
