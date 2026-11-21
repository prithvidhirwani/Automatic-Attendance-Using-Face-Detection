import base64
import json

import cv2
import numpy as np
import pytest

from ai.detector import FaceDetector
from ai.recognizer import ModelNotReady, Recognizer
from ai.trainer import TrainingError, train_model
from database import Database


def data_url(img):
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def test_pages_load_empty(client):
    for path in ["/", "/students", "/students/register", "/attendance", "/attendance/live", "/evaluation"]:
        assert client.get(path).status_code == 200, path
    assert b"not been performed" in client.get("/evaluation").data


def test_dashboard_numbers_come_from_database(client, app):
    db = app.extensions["face"].db
    a = db.add_student("R1", "Asha")
    db.add_student("R2", "Ben")
    db.mark_attendance(a, 30.0)
    html = client.get("/").get_data(as_text=True)
    assert "<b>2</b>Registered" in html and "<b>1</b>Present" in html and "50.0%" in html


def test_register_student_and_duplicate(client):
    r = client.post("/students/register", data={"roll_no": "R1", "full_name": "Asha", "department": "IT"})
    assert r.status_code == 302
    r2 = client.post("/students/register", data={"roll_no": "r1", "full_name": "Other"})
    assert r2.status_code == 400 and b"already registered" in r2.data
    assert b"Asha" in client.get("/students?q=ash").data
    assert b"Asha" not in client.get("/students?q=zzz").data


def test_delete_student_removes_images(client, app):
    sid = app.extensions["face"].db.add_student("R1", "Asha")
    folder = app.config["TRAIN_DIR"] / str(sid)
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "a.png"), np.zeros((20, 20), np.uint8))
    assert client.post(f"/students/{sid}/delete").status_code == 302
    assert not folder.exists() and client.get(f"/students/{sid}").status_code == 404


def test_capture_rejects_invalid_and_no_face(client, app):
    sid = app.extensions["face"].db.add_student("R1", "Asha")
    assert client.post(f"/api/students/{sid}/capture", json={"image": "garbage"}).status_code == 400
    blank = np.full((480, 640, 3), 128, np.uint8)
    r = client.post(f"/api/students/{sid}/capture", json={"image": data_url(blank)})
    assert r.status_code == 422 and "No face" in r.get_json()["error"]


def test_recognize_without_model_is_409(client):
    blank = np.zeros((100, 100, 3), np.uint8)
    r = client.post("/api/recognize", json={"image": data_url(blank)})
    assert r.status_code == 409 and not r.get_json()["ok"]


def test_train_without_students_or_images(client, app):
    assert client.post("/api/train", json={}).status_code == 422
    app.extensions["face"].db.add_student("R1", "Asha")
    r = client.post("/api/train", json={})
    assert r.status_code == 422 and "No usable training images" in r.get_json()["error"]


def test_csv_exports_and_whitelist(client, app):
    db = app.extensions["face"].db
    sid = db.add_student("R1", "=cmd|' /C calc'!A0")
    db.mark_attendance(sid, 1.0)
    body = client.get("/attendance/export.csv").get_data(as_text=True)
    assert body.startswith("date,roll_no") and "'=cmd" in body        # formula injection neutralised
    assert client.get("/evaluation/export/../../config.csv").status_code == 404
    assert client.get("/evaluation/export/secrets.csv").status_code == 404
    assert client.get("/instance/attendance.db").status_code == 404
    assert client.get("/models/m.yml").status_code == 404


def test_upload_rejects_non_image_and_bad_ground_truth(client, app):
    db = app.extensions["face"].db
    db.add_student("R1", "Asha")
    import io
    bad = client.post("/evaluation/test-image", data={"type": "individual", "people": "R1",
                      "image": (io.BytesIO(b"<?php echo 1; ?>"), "x.jpg")}, content_type="multipart/form-data")
    assert bad.status_code == 302
    assert not list((app.config["TEST_DIR"] / "individual").glob("*"))
    ok, buf = cv2.imencode(".png", np.zeros((50, 50, 3), np.uint8))
    for people in ("NOPE", "R1;R1"):
        client.post("/evaluation/test-image", data={"type": "individual", "people": people,
                    "image": (io.BytesIO(buf.tobytes()), "x.png")}, content_type="multipart/form-data")
    assert not list((app.config["TEST_DIR"] / "individual").glob("*"))
    client.post("/evaluation/test-image", data={"type": "individual", "people": "r1",
                "image": (io.BytesIO(buf.tobytes()), "x.png")}, content_type="multipart/form-data")
    assert len(list((app.config["TEST_DIR"] / "individual").glob("*.jpg"))) == 1
    assert "R1" in app.config["GROUND_TRUTH_CSV"].read_text()


# ------------------------------------------------------------------ model / detector units
def test_detector_no_faces_in_blank_image():
    det = FaceDetector()
    assert det.detect(np.full((300, 300, 3), 200, np.uint8)) == []
    assert det.detect(None) == []


def test_recognizer_missing_and_corrupt_model(tmp_path):
    rec = Recognizer(tmp_path / "m.yml", tmp_path / "m.json")
    assert rec.load() is False and not rec.ready and "No trained model" in rec.error
    with pytest.raises(ModelNotReady):
        rec.predict_face(np.zeros((200, 200), np.uint8))
    (tmp_path / "m.yml").write_text("not a model")
    (tmp_path / "m.json").write_text("{broken json")
    assert rec.load() is False and "could not be loaded" in rec.error
    (tmp_path / "m.json").write_text(json.dumps({"labels": [1]}))
    assert rec.load() is False


def synthetic_face(seed):
    """Non-biometric stand-in: a fixed random texture per 'identity' plus a little noise."""
    base = np.random.RandomState(seed).randint(0, 255, (200, 200)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    return base


def test_train_load_predict_and_unknown_threshold(tmp_path):
    db = Database(tmp_path / "d.db")
    db.init_db()
    ids = [db.add_student(f"R{i}", f"S{i}") for i in range(2)]
    train = tmp_path / "train"
    rng = np.random.RandomState(0)
    for sid in ids:
        (train / str(sid)).mkdir(parents=True)
        for k in range(6):
            img = np.clip(synthetic_face(sid) + rng.normal(0, 4, (200, 200)), 0, 255).astype(np.uint8)
            cv2.imwrite(str(train / str(sid) / f"{k}.png"), img)
    (train / "999").mkdir()  # orphan folder must be ignored, not crash
    rep = train_model(db, train, tmp_path / "m.yml", tmp_path / "m.json", equalize=False)
    assert rep["num_identities"] == 2 and rep["num_images"] == 12
    assert any("999" in w for w in rep["warnings"])

    rec = Recognizer(tmp_path / "m.yml", tmp_path / "m.json", threshold=80)
    assert rec.load() and rec.ready
    probe = np.clip(synthetic_face(ids[1]) + rng.normal(0, 4, (200, 200)), 0, 255).astype(np.uint8)
    sid, dist = rec.predict_face(probe)
    assert sid == ids[1] and dist < 80
    stranger = np.clip(synthetic_face(777), 0, 255).astype(np.uint8)
    rec.threshold = 0.5      # absurdly strict -> everything is Unknown
    assert rec.predict_face(probe)[0] is None
    out = rec.recognize(cv2.cvtColor(probe, cv2.COLOR_GRAY2BGR), [(0, 0, 200, 200)])
    assert out[0]["student_id"] is None and out[0]["distance"] is not None


def test_training_with_empty_dataset_raises(tmp_path):
    db = Database(tmp_path / "d.db")
    db.init_db()
    with pytest.raises(TrainingError):
        train_model(db, tmp_path / "train", tmp_path / "m.yml", tmp_path / "m.json")
    db.add_student("R1", "A")
    with pytest.raises(TrainingError):
        train_model(db, tmp_path / "train", tmp_path / "m.yml", tmp_path / "m.json")
