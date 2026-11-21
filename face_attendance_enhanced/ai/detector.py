"""Haar-cascade face detection. Detection only - no identity here."""
import cv2

from ai.preprocessing import to_gray


class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=60):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image):
        """Returns [(x, y, w, h), ...] sorted left to right. Empty list = no faces."""
        if image is None or image.size == 0:
            return []
        gray = to_gray(image)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        out = [tuple(int(v) for v in b) for b in boxes if min(b[2], b[3]) >= self.min_size]
        return sorted(out, key=lambda b: b[0])
