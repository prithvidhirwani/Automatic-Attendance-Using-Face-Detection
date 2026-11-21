"""Shared evaluation pipeline: ground truth -> detect -> recognise -> score -> CSV."""
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2

from ai.detector import FaceDetector
from ai.recognizer import Recognizer, ModelNotReady, UNKNOWN
from config import Config
from database import Database
from evaluation import metrics as M

GT_FIELDS = ["image", "type", "people", "boxes", "notes"]
RESULT_FIELDS = [
    "image", "type", "mode", "box_verified", "actual_identity", "predicted_identity", "is_correct",
    "actual_faces", "faces_detected", "tp_faces", "missed_faces", "false_detections",
    "registered_actual", "unknown_actual", "correct_identities", "wrong_identities", "false_unknown",
    "correct_rejections", "false_accepts", "identity_accuracy", "unknown_rejection_rate", "pairs",
]
RESULT_FILES = {"individual": "individual_results.csv", "group": "group_results.csv"}
SUMMARY_FILE = "summary_metrics.csv"


def ensure_ground_truth(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(GT_FIELDS)


def read_ground_truth(path):
    ensure_ground_truth(path)
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_ground_truth(path, row):
    ensure_ground_truth(path)
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=GT_FIELDS).writerow({k: row.get(k, "") for k in GT_FIELDS})


def safe_cell(v):
    """Neutralise spreadsheet formula injection in exported CSVs."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") and not _looks_numeric(v):
        return "'" + v
    return v


def _looks_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_context(cfg=None, threshold=None):
    cfg = cfg or Config
    db = Database(cfg.DATABASE_PATH)
    db.init_db()
    det = FaceDetector(cfg.DETECT_SCALE_FACTOR, cfg.DETECT_MIN_NEIGHBORS, cfg.DETECT_MIN_SIZE)
    rec = Recognizer(cfg.MODEL_PATH, cfg.LABEL_MAP_PATH,
                     cfg.LBPH_THRESHOLD if threshold is None else threshold, cfg.FACE_SIZE)
    rec.load()
    return db, det, rec


def run_evaluation(kind, cfg=None, threshold=None, db=None, detector=None, recognizer=None):
    """kind: 'individual' | 'group'. Writes results/<kind>_results.csv and refreshes summary_metrics.csv."""
    cfg = cfg or Config
    if db is None or detector is None or recognizer is None:
        db, detector, recognizer = build_context(cfg, threshold)
    elif threshold is not None:
        recognizer.threshold = float(threshold)
    if not recognizer.ready:
        raise ModelNotReady(recognizer.error or "Model not loaded.")

    roll_of = {s["id"]: s["roll_no"] for s in db.list_students()}
    registered = {r.lower(): r for r in roll_of.values()}
    rows, warnings = [], []
    gt_rows = [r for r in read_ground_truth(cfg.GROUND_TRUTH_CSV) if (r.get("type") or "").strip().lower() == kind]

    for gt in gt_rows:
        name = (gt.get("image") or "").strip()
        try:
            people = M.parse_people(gt.get("people"))
            boxes = M.parse_boxes(gt.get("boxes"))
        except ValueError as exc:
            warnings.append(f"{name}: skipped - {exc}")
            continue
        if kind == "individual" and len(people) != 1:
            warnings.append(f"{name}: skipped - an individual image must list exactly one person.")
            continue
        bad = [p for p in people if p != UNKNOWN and p.lower() not in registered]
        if bad:
            warnings.append(f"{name}: skipped - not registered: {', '.join(bad)} (write UNKNOWN for unregistered people).")
            continue
        people = [registered.get(p.lower(), p) for p in people]
        path = (Path(cfg.TEST_DIR) / name).resolve()
        if Path(cfg.TEST_DIR).resolve() not in path.parents or not path.is_file():
            warnings.append(f"{name}: skipped - file not found inside test_dataset.")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            warnings.append(f"{name}: skipped - unreadable image.")
            continue
        if boxes is not None and len(boxes) != len(people):
            warnings.append(f"{name}: boxes ignored - {len(boxes)} boxes for {len(people)} people.")
            boxes = None

        found = detector.detect(img)
        recs = recognizer.recognize(img, found)
        dets = [{"box": r["box"], "label": roll_of.get(r["student_id"], UNKNOWN) if r["student_id"] is not None else UNKNOWN}
                for r in recs]
        res = M.score_image(people, dets, boxes, cfg.IOU_THRESHOLD)
        pair_preds = [p for _, p in M.decode_pairs(res["pairs"])]
        res.update(
            image=name, type=kind,
            actual_identity=";".join(people),
            predicted_identity=";".join(d["label"] for d in dets) if dets else "(no face detected)",
            is_correct=(res["correct_identities"] + res["correct_rejections"]) == res["actual_faces"],
        )
        rows.append(res)

    out_dir = Path(cfg.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / RESULT_FILES[kind]).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else safe_cell(r.get(k))) for k in RESULT_FIELDS})

    info = {
        "kind": kind, "run_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": recognizer.threshold, "equalize": recognizer.equalize,
        "model_trained_at": recognizer.meta.get("trained_at"),
        "model_identities": recognizer.meta.get("num_identities"),
        "images_evaluated": len(rows), "images_skipped": len(warnings), "warnings": warnings,
        "ground_truth_rows": len(gt_rows),
    }
    (out_dir / f"run_info_{kind}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    try:
        db.log_evaluation_run(kind, recognizer.threshold, recognizer.equalize, len(rows))
    except Exception:
        pass
    build_summary(cfg, db)
    return info


# ------------------------------------------------------------------ summary
def load_results(results_dir, kind):
    p = Path(results_dir) / RESULT_FILES[kind]
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_info(results_dir, kind):
    p = Path(results_dir) / f"run_info_{kind}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_summary(ind_rows, grp_rows, registered_rolls, target=34):
    """Returns list of (scope, metric, value). Values are None when not measurable."""
    out = []

    def add(scope, rep, rows):
        agg = M.aggregate(rows)
        for k in ["images", "actual_faces", "faces_detected", "registered_actual", "unknown_actual",
                  "correct_identities", "wrong_identities", "false_unknown", "correct_rejections",
                  "false_accepts", "identity_accuracy", "unknown_rejection_rate", "box_verified_images",
                  "detection_tp", "detection_fp", "detection_fn", "detection_precision", "detection_recall"]:
            out.append((scope, k, agg[k]))
        pairs = [pr for r in rows for pr in M.decode_pairs(r.get("pairs"))]
        mc = M.multiclass_report(pairs)
        if mc:
            for kind in ("macro", "weighted"):
                for m in ("precision", "recall", "f1"):
                    out.append((scope, f"{kind}_{m}", mc[kind][m]))
            if scope == "overall":
                for lab, v in mc["per_class"].items():
                    for m in ("precision", "recall", "f1", "support"):
                        out.append(("per_identity", f"{lab}:{m}", v[m]))
        else:
            for kind in ("macro", "weighted"):
                for m in ("precision", "recall", "f1"):
                    out.append((scope, f"{kind}_{m}", None))

    add("individual", None, ind_rows)
    add("group", None, grp_rows)
    add("overall", None, ind_rows + grp_rows)

    tested = {p.split(";")[0] for p in (r.get("actual_identity", "") for r in ind_rows) if p and p != UNKNOWN}
    tested = {t for t in tested if t in set(registered_rolls)}
    out.append(("coverage", "registered_identities", len(registered_rolls)))
    out.append(("coverage", "identities_tested_individually", len(tested)))
    out.append(("coverage", "target_identities", target))
    out.append(("coverage", "test_complete", len(tested) >= target))
    return out


def build_summary(cfg, db):
    ind, grp = load_results(cfg.RESULTS_DIR, "individual"), load_results(cfg.RESULTS_DIR, "group")
    rolls = [s["roll_no"] for s in db.list_students()]
    summary = compute_summary(ind, grp, rolls, cfg.TARGET_IDENTITIES)
    info = [load_info(cfg.RESULTS_DIR, k) for k in ("individual", "group")]
    for k, i in zip(("individual", "group"), info):
        if i:
            summary += [("settings", f"{k}_threshold", i["threshold"]), ("settings", f"{k}_equalize", i["equalize"]),
                        ("settings", f"{k}_run_at", i["run_at"]), ("settings", f"{k}_images_skipped", i["images_skipped"])]
    out = Path(cfg.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with (out / SUMMARY_FILE).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "metric", "value"])
        for scope, metric, value in summary:
            w.writerow([scope, safe_cell(metric), "" if value is None else value])
    return summary
