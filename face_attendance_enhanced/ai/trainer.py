"""Builds the LBPH model from dataset/train/<student_id>/*.png|jpg."""
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ai.preprocessing import normalize_face

IMG_EXT = {".png", ".jpg", ".jpeg"}


class TrainingError(RuntimeError):
    pass


def list_student_images(train_dir, student_id):
    folder = Path(train_dir) / str(student_id)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def train_model(db, train_dir, model_path, label_map_path, face_size=(200, 200), equalize=True):
    """Train on every registered student's images. Returns a report dict."""
    if not hasattr(cv2, "face"):
        raise TrainingError("cv2.face is missing. Install opencv-contrib-python (not opencv-python).")
    train_dir = Path(train_dir)
    students = db.list_students()
    if not students:
        raise TrainingError("No students are registered yet.")

    images, labels, counts, warnings = [], [], {}, []
    for s in students:
        usable = 0
        for p in list_student_images(train_dir, s["id"]):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                warnings.append(f"Skipped unreadable file {p.name} ({s['roll_no']}).")
                continue
            images.append(normalize_face(img, face_size, equalize))
            labels.append(s["id"])
            usable += 1
        counts[str(s["id"])] = usable
        if usable == 0:
            warnings.append(f"{s['roll_no']} ({s['full_name']}) has no usable images and was not trained.")

    known = {str(s["id"]) for s in students}
    if train_dir.is_dir():
        for d in train_dir.iterdir():
            if d.is_dir() and d.name not in known:
                warnings.append(f"Ignored folder '{d.name}': it does not match a registered student.")

    if not images:
        raise TrainingError("No usable training images were found. Capture images first.")
    identities = sorted(set(labels))
    if len(identities) < 2:
        warnings.append("Only one identity is trained. The model can then only separate 'this person' from "
                        "'unknown' via the distance threshold - register more students for meaningful results.")

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.array(labels, dtype=np.int32))

    model_path, label_map_path = Path(model_path), Path(label_map_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp.yml")
    model.write(str(tmp))
    os.replace(tmp, model_path)  # atomic: never leaves a half-written model

    meta = {
        "labels": identities,
        "counts": counts,
        "equalize": bool(equalize),
        "face_size": list(face_size),
        "num_images": len(images),
        "num_identities": len(identities),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    label_map_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {**meta, "warnings": warnings}
