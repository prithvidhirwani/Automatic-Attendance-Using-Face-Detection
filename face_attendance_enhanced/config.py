"""Central configuration. Every value can be overridden with an environment
variable (see README) so nothing needs to be edited in code."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Config:
    # ---- paths -------------------------------------------------------
    BASE_DIR = BASE_DIR
    INSTANCE_DIR = BASE_DIR / "instance"
    DATABASE_PATH = INSTANCE_DIR / "attendance.db"
    SECRET_KEY_FILE = INSTANCE_DIR / "secret_key.txt"
    TRAIN_DIR = BASE_DIR / "dataset" / "train"
    TEST_DIR = BASE_DIR / "test_dataset"
    GROUND_TRUTH_CSV = TEST_DIR / "ground_truth.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "lbph_model.yml"
    LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
    RESULTS_DIR = BASE_DIR / "results"

    # ---- face detection (Haar cascade) ------------------------------
    DETECT_SCALE_FACTOR = _float("FACE_DETECT_SCALE", 1.1)
    DETECT_MIN_NEIGHBORS = _int("FACE_DETECT_NEIGHBORS", 5)
    DETECT_MIN_SIZE = _int("FACE_DETECT_MIN_SIZE", 60)  # pixels; smaller boxes are dropped

    # ---- preprocessing ----------------------------------------------
    FACE_SIZE = (200, 200)
    # Histogram equalisation is optional. Train + evaluate with it on and off to
    # find out whether it helps with YOUR data.
    USE_EQUALIZATION = _bool("FACE_EQUALIZE", True)

    # ---- capture quality gates --------------------------------------
    MIN_CAPTURE_FACE = _int("FACE_MIN_CAPTURE_SIZE", 90)
    MIN_BLUR_VARIANCE = _float("FACE_MIN_SHARPNESS", 40.0)   # Laplacian variance
    MIN_BRIGHTNESS = _float("FACE_MIN_BRIGHTNESS", 50.0)
    MAX_BRIGHTNESS = _float("FACE_MAX_BRIGHTNESS", 210.0)
    TARGET_TRAIN_IMAGES = 20
    MIN_TRAIN_IMAGES = 10
    MAX_TRAIN_IMAGES = 60

    # ---- recognition -------------------------------------------------
    # LBPH returns a DISTANCE (lower = more similar). It is NOT a probability
    # or a percentage. A face is accepted as a registered student only when the
    # distance is <= this threshold; otherwise it is labelled "Unknown".
    # There is no universal value: choose it using a validation set that is
    # separate from your final test set.
    LBPH_THRESHOLD = _float("FACE_LBPH_THRESHOLD", 70.0)
    CONFIRM_FRAMES = _int("FACE_CONFIRM_FRAMES", 3)  # consecutive frames before marking attendance

    # ---- evaluation --------------------------------------------------
    TARGET_IDENTITIES = 34
    IOU_THRESHOLD = 0.5

    # ---- web / security ----------------------------------------------
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB per request
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
    MAX_IMAGE_DIMENSION = 6000
