"""End-to-end evaluation test using synthetic, non-biometric textures and a stub detector.
It checks the plumbing (ground truth -> predictions -> CSVs), NOT real-world accuracy."""
import csv

import cv2
import numpy as np

from ai.recognizer import Recognizer
from ai.trainer import train_model
from evaluation.common import append_ground_truth, run_evaluation


class WholeImageDetector:
    def detect(self, img):
        h, w = img.shape[:2]
        return [(0, 0, w, h)]


class NoFaceDetector:
    def detect(self, img):
        return []


def texture(seed, rng):
    base = cv2.GaussianBlur(np.random.RandomState(seed).randint(0, 255, (200, 200)).astype(np.float32), (0, 0), 3)
    return np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)


def setup(app):
    st = app.extensions["face"]
    cfg = app.config
    rng = np.random.RandomState(1)
    ids = [st.db.add_student(f"R{i}", f"S{i}") for i in range(3)]
    for sid in ids:
        d = cfg["TRAIN_DIR"] / str(sid)
        d.mkdir(parents=True)
        for k in range(6):
            cv2.imwrite(str(d / f"{k}.png"), texture(sid, rng))
    train_model(st.db, cfg["TRAIN_DIR"], cfg["MODEL_PATH"], cfg["LABEL_MAP_PATH"], equalize=False)
    st.recognizer.load()
    st.recognizer.threshold = 90
    for i, sid in enumerate(ids):  # independent test images (fresh noise)
        name = f"individual/t{i}.png"
        cv2.imwrite(str(cfg["TEST_DIR"] / name), texture(sid, rng))
        append_ground_truth(cfg["GROUND_TRUTH_CSV"], {"image": name, "type": "individual", "people": f"R{i}"})
    return st, ids


class Cfg:  # attribute view
    def __init__(self, d): self.__dict__.update(d)


def test_individual_and_summary_files(app):
    st, _ = setup(app)
    cfg = Cfg(app.config)
    info = run_evaluation("individual", cfg, db=st.db, detector=WholeImageDetector(), recognizer=st.recognizer)
    assert info["images_evaluated"] == 3 and info["images_skipped"] == 0
    rows = list(csv.DictReader((app.config["RESULTS_DIR"] / "individual_results.csv").open()))
    assert [r["actual_identity"] for r in rows] == ["R0", "R1", "R2"]
    assert all(r["predicted_identity"] == r["actual_identity"] for r in rows)
    summ = {(r["scope"], r["metric"]): r["value"] for r in csv.DictReader((app.config["RESULTS_DIR"] / "summary_metrics.csv").open())}
    assert summ[("individual", "identity_accuracy")] == "100.0"
    assert summ[("coverage", "identities_tested_individually")] == "3"
    assert summ[("coverage", "test_complete")] == "False"      # target is 34
    assert summ[("group", "identity_accuracy")] == ""            # no group data -> not measurable, not 0
    assert (app.config["RESULTS_DIR"] / "group_results.csv").exists() is False


def test_no_face_and_unregistered_rows(app):
    st, _ = setup(app)
    append_ground_truth(app.config["GROUND_TRUTH_CSV"], {"image": "individual/t0.png", "type": "individual", "people": "R99"})
    info = run_evaluation("individual", Cfg(app.config), db=st.db, detector=NoFaceDetector(), recognizer=st.recognizer)
    assert info["images_skipped"] == 1 and "not registered" in info["warnings"][0]
    rows = list(csv.DictReader((app.config["RESULTS_DIR"] / "individual_results.csv").open()))
    assert all(r["predicted_identity"] == "(no face detected)" and r["identity_accuracy"] == "0.0" for r in rows)


def test_route_evaluation_flow(client, app):
    assert client.post("/api/evaluation/run", json={"kind": "both"}).status_code == 409   # no model yet
    setup(app)
    assert client.post("/api/evaluation/run", json={"kind": "nope"}).status_code == 400
    assert client.post("/api/evaluation/run", json={"kind": "group", "threshold": "abc"}).status_code == 400
    r = client.post("/api/evaluation/run", json={"kind": "group"})   # real Haar detector; no group rows
    assert r.status_code == 200 and r.get_json()["group"]["evaluated"] == 0
    assert app.extensions["face"].recognizer.threshold == 90          # one-off threshold must not leak
    assert client.get("/evaluation").status_code == 200
    assert client.get("/evaluation/export/summary_metrics.csv").status_code == 200
