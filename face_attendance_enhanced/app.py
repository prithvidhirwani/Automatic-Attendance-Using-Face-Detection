"""Flask application: AI-based face recognition attendance system."""
import csv
import io
import re
import secrets
import shutil
import threading
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)

from ai.detector import FaceDetector
from ai.preprocessing import (check_quality, crop_face, decode_data_url, decode_image_bytes,
                              normalize_face, to_gray)
from ai.recognizer import ModelNotReady, Recognizer
from ai.trainer import TrainingError, list_student_images, train_model
from config import Config
from database import Database, DuplicateRollError
from evaluation import metrics as M
from evaluation.common import (RESULT_FILES, SUMMARY_FILE, append_ground_truth, ensure_ground_truth,
                               load_info, load_results, safe_cell)
from evaluation.evaluate_groups import run as run_groups
from evaluation.evaluate_individual import run as run_individual

IMG_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}\.(png|jpg|jpeg)$", re.I)
EXPORTS = {"individual_results": RESULT_FILES["individual"], "group_results": RESULT_FILES["group"],
           "summary_metrics": SUMMARY_FILE}


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    cfg = app.config
    for key in ("INSTANCE_DIR", "TRAIN_DIR", "MODEL_DIR", "RESULTS_DIR"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    for sub in ("individual", "groups"):
        (Path(cfg["TEST_DIR"]) / sub).mkdir(parents=True, exist_ok=True)
    ensure_ground_truth(cfg["GROUND_TRUTH_CSV"])

    key_file = Path(cfg["SECRET_KEY_FILE"])
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
    app.secret_key = key_file.read_text().strip()

    db = Database(cfg["DATABASE_PATH"])
    db.init_db()
    detector = FaceDetector(cfg["DETECT_SCALE_FACTOR"], cfg["DETECT_MIN_NEIGHBORS"], cfg["DETECT_MIN_SIZE"])
    recognizer = Recognizer(cfg["MODEL_PATH"], cfg["LABEL_MAP_PATH"], cfg["LBPH_THRESHOLD"], cfg["FACE_SIZE"])
    recognizer.load()
    state = SimpleNamespace(db=db, detector=detector, recognizer=recognizer, streaks={}, lock=threading.Lock())
    app.extensions["face"] = state

    # ------------------------------------------------------------ security
    @app.before_request
    def csrf_protect():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        if request.method == "POST" and not cfg.get("TESTING"):
            sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
            if not sent or not secrets.compare_digest(sent, session["csrf"]):
                abort(400, "Missing or invalid CSRF token. Reload the page and try again.")

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def inject():
        return {"csrf_token": session.get("csrf", ""), "cfg": cfg,
                "model_ready": recognizer.ready, "model_error": recognizer.error}

    @app.errorhandler(413)
    def too_big(_):
        return jsonify(ok=False, error="File too large (limit 8 MB)."), 413

    # ------------------------------------------------------------ helpers
    def train_dir(sid):
        return Path(cfg["TRAIN_DIR"]) / str(int(sid))

    def student_row(s):
        n = len(list_student_images(cfg["TRAIN_DIR"], s["id"]))
        meta = recognizer.meta if recognizer.ready else {}
        trained = meta.get("counts", {}).get(str(s["id"]))
        if n == 0:
            status = "No images"
        elif n < cfg["MIN_TRAIN_IMAGES"]:
            status = "Needs more images"
        elif trained is None:
            status = "Not trained"
        elif trained != n:
            status = "Retrain needed"
        else:
            status = "Trained"
        return {**s, "image_count": n, "status": status}

    def get_or_404(sid):
        s = db.get_student(sid)
        if not s:
            abort(404)
        return s

    # ------------------------------------------------------------ pages
    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", stats=db.dashboard_stats(), recent=db.recent_attendance(10),
                               meta=recognizer.meta if recognizer.ready else None, threshold=recognizer.threshold)

    @app.get("/students")
    def students():
        q = request.args.get("q", "")
        return render_template("students.html", students=[student_row(s) for s in db.list_students(q)], q=q)

    @app.route("/students/register", methods=["GET", "POST"])
    def register_student():
        if request.method == "POST":
            try:
                sid = db.add_student(request.form.get("roll_no"), request.form.get("full_name"),
                                     request.form.get("department"))
            except (DuplicateRollError, ValueError) as exc:
                flash(str(exc), "error")
                return render_template("register_student.html", form=request.form), 400
            flash("Student registered. Now capture face images.", "ok")
            return redirect(url_for("student_detail", sid=sid))
        return render_template("register_student.html", form={})

    @app.get("/students/<int:sid>")
    def student_detail(sid):
        s = student_row(get_or_404(sid))
        files = [p.name for p in list_student_images(cfg["TRAIN_DIR"], sid)]
        return render_template("student_detail.html", s=s, files=files)

    @app.get("/students/<int:sid>/images/<name>")
    def student_image(sid, name):
        if not IMG_NAME.match(name):
            abort(404)
        return send_from_directory(train_dir(sid), name)

    @app.post("/students/<int:sid>/delete")
    def delete_student(sid):
        s = get_or_404(sid)
        folder = train_dir(sid).resolve()
        root = Path(cfg["TRAIN_DIR"]).resolve()
        if folder.parent == root and folder.is_dir():
            shutil.rmtree(folder)
        db.delete_student(sid)
        flash(f"Deleted {s['full_name']} ({s['roll_no']}), their attendance and their images. Retrain the model.", "ok")
        return redirect(url_for("students"))

    @app.post("/students/<int:sid>/images/<name>/delete")
    def delete_image(sid, name):
        get_or_404(sid)
        if not IMG_NAME.match(name):
            abort(404)
        p = train_dir(sid) / name
        if p.is_file():
            p.unlink()
        return redirect(url_for("student_detail", sid=sid))

    # ------------------------------------------------------------ capture / train API
    @app.post("/api/students/<int:sid>/capture")
    def api_capture(sid):
        get_or_404(sid)
        img = decode_data_url((request.get_json(silent=True) or {}).get("image"), max_dim=cfg["MAX_IMAGE_DIMENSION"])
        if img is None:
            return jsonify(ok=False, error="Invalid image."), 400
        gray = to_gray(img)
        boxes = detector.detect(gray)
        if not boxes:
            return jsonify(ok=False, error="No face detected. Face the camera in good light."), 422
        if len(boxes) > 1:
            return jsonify(ok=False, error="More than one face in view. Only the student should be visible."), 422
        ok, reasons, qm = check_quality(gray, boxes[0], cfg["MIN_CAPTURE_FACE"], cfg["MIN_BLUR_VARIANCE"],
                                        cfg["MIN_BRIGHTNESS"], cfg["MAX_BRIGHTNESS"])
        if not ok:
            return jsonify(ok=False, error=" ".join(reasons), quality=qm), 422
        folder = train_dir(sid)
        folder.mkdir(parents=True, exist_ok=True)
        count = len(list_student_images(cfg["TRAIN_DIR"], sid))
        if count >= cfg["MAX_TRAIN_IMAGES"]:
            return jsonify(ok=False, error=f"Limit of {cfg['MAX_TRAIN_IMAGES']} images reached."), 422
        face = normalize_face(crop_face(gray, boxes[0]), cfg["FACE_SIZE"], equalize=False)
        name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.png"
        cv2.imwrite(str(folder / name), face)
        return jsonify(ok=True, count=count + 1, name=name, quality=qm,
                       url=url_for("student_image", sid=sid, name=name))

    @app.post("/api/train")
    def api_train():
        equalize = (request.get_json(silent=True) or {}).get("equalize", cfg["USE_EQUALIZATION"])
        try:
            report = train_model(db, cfg["TRAIN_DIR"], cfg["MODEL_PATH"], cfg["LABEL_MAP_PATH"],
                                 cfg["FACE_SIZE"], bool(equalize))
        except TrainingError as exc:
            return jsonify(ok=False, error=str(exc)), 422
        recognizer.load()
        with state.lock:
            state.streaks.clear()
        return jsonify(ok=True, **{k: report[k] for k in ("num_images", "num_identities", "equalize", "trained_at", "warnings")})

    # ------------------------------------------------------------ live attendance
    @app.get("/attendance/live")
    def live_attendance():
        return render_template("live_attendance.html", threshold=recognizer.threshold)

    @app.post("/api/recognize")
    def api_recognize():
        data = request.get_json(silent=True) or {}
        img = decode_data_url(data.get("image"), max_dim=cfg["MAX_IMAGE_DIMENSION"])
        if img is None:
            return jsonify(ok=False, error="Invalid frame."), 400
        if not recognizer.ready:
            return jsonify(ok=False, error=recognizer.error or "Model not trained."), 409
        session_id = str(data.get("session_id", ""))[:40] or "default"
        mark = bool(data.get("mark", True))
        boxes = detector.detect(img)
        try:
            results = recognizer.recognize(img, boxes)
        except ModelNotReady as exc:
            return jsonify(ok=False, error=str(exc)), 409
        names = {s["id"]: s for s in db.list_students()}
        faces, marked = [], []
        with state.lock:
            if len(state.streaks) > 50:
                state.streaks.clear()
            seen = {}
            streak = state.streaks.setdefault(session_id, {})
            for r in results:
                sid = r["student_id"]
                f = {"box": list(r["box"]), "distance": None if r["distance"] is None else round(r["distance"], 1),
                     "name": "Unknown", "roll": None, "status": "unknown"}
                if sid is not None and sid in names:
                    seen[sid] = streak.get(sid, 0) + 1
                    f.update(name=names[sid]["full_name"], roll=names[sid]["roll_no"], status="confirming")
                    if mark and seen[sid] >= cfg["CONFIRM_FRAMES"]:
                        new = db.mark_attendance(sid, r["distance"])
                        f["status"] = "marked" if new else "already_marked"
                        if new:
                            marked.append(names[sid]["full_name"])
                faces.append(f)
            state.streaks[session_id] = seen  # students not seen this frame lose their streak
        h, w = img.shape[:2]
        return jsonify(ok=True, faces=faces, marked=marked, frame=[w, h])

    # ------------------------------------------------------------ records
    @app.get("/attendance")
    def attendance_records():
        f = {k: request.args.get(k, "") for k in ("date_from", "date_to", "q")}
        return render_template("attendance_records.html", records=db.attendance_records(**f), f=f,
                               export_args={k: v for k, v in f.items() if v})

    @app.get("/attendance/export.csv")
    def attendance_export():
        f = {k: request.args.get(k, "") for k in ("date_from", "date_to", "q")}
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "roll_no", "name", "department", "timestamp", "status", "lbph_distance"])
        for r in db.attendance_records(**f):
            w.writerow([r["att_date"], safe_cell(r["roll_no"]), safe_cell(r["full_name"]), safe_cell(r["department"]),
                        r["marked_at"], r["status"], "" if r["distance"] is None else round(r["distance"], 2)])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=attendance.csv"})

    # ------------------------------------------------------------ evaluation
    def read_summary():
        p = Path(cfg["RESULTS_DIR"]) / SUMMARY_FILE
        if not p.exists():
            return None
        with p.open(newline="", encoding="utf-8") as fh:
            return [r for r in csv.DictReader(fh)]

    @app.get("/evaluation")
    def evaluation():
        summary = read_summary()
        s = {}
        for r in summary or []:
            s.setdefault(r["scope"], {})[r["metric"]] = r["value"]
        per_identity = {}
        for k, v in s.get("per_identity", {}).items():
            lab, m = k.rsplit(":", 1)
            per_identity.setdefault(lab, {})[m] = v
        gt = []
        try:
            with Path(cfg["GROUND_TRUTH_CSV"]).open(newline="", encoding="utf-8-sig") as fh:
                gt = list(csv.DictReader(fh))
        except OSError:
            pass
        return render_template(
            "evaluation.html", s=s, per_identity=per_identity, has_summary=summary is not None,
            ind=load_results(cfg["RESULTS_DIR"], "individual"), grp=load_results(cfg["RESULTS_DIR"], "group"),
            info={k: load_info(cfg["RESULTS_DIR"], k) for k in ("individual", "group")},
            gt_count=len(gt), gt_ind=sum(1 for r in gt if r.get("type") == "individual"),
            gt_grp=sum(1 for r in gt if r.get("type") == "group"),
            students=db.list_students(), threshold=recognizer.threshold)

    @app.post("/api/evaluation/run")
    def api_eval_run():
        data = request.get_json(silent=True) or {}
        kind = data.get("kind")
        if kind not in ("individual", "group", "both"):
            return jsonify(ok=False, error="kind must be individual, group or both"), 400
        thr = data.get("threshold")
        try:
            thr = float(thr) if thr not in (None, "") else None
        except (TypeError, ValueError):
            return jsonify(ok=False, error="Threshold must be a number."), 400
        if thr is not None and not (0 < thr < 1000):
            return jsonify(ok=False, error="Threshold must be between 0 and 1000."), 400
        if not recognizer.ready:
            return jsonify(ok=False, error=recognizer.error or "Train the model first."), 409
        original = recognizer.threshold
        try:
            kw = dict(cfg=_Cfg(cfg), threshold=thr, db=db, detector=detector, recognizer=recognizer)
            out = {}
            for k, fn in (("individual", run_individual), ("group", run_groups)):
                if kind in (k, "both"):
                    out[k] = fn(**kw)
        finally:
            recognizer.threshold = original  # a one-off evaluation threshold must not leak into live attendance
        return jsonify(ok=True, **{k: {"evaluated": v["images_evaluated"], "skipped": v["images_skipped"]}
                                    for k, v in out.items()})

    @app.get("/evaluation/export/<name>.csv")
    def eval_export(name):
        if name not in EXPORTS:
            abort(404)
        p = Path(cfg["RESULTS_DIR"]) / EXPORTS[name]
        if not p.exists():
            flash("That results file has not been generated yet. Run the evaluation first.", "error")
            return redirect(url_for("evaluation"))
        return send_from_directory(cfg["RESULTS_DIR"], EXPORTS[name], as_attachment=True)

    @app.post("/evaluation/test-image")
    def upload_test_image():
        kind = request.form.get("type")
        if kind not in ("individual", "group"):
            flash("Choose the image type.", "error")
            return redirect(url_for("evaluation"))
        f = request.files.get("image")
        if not f or not f.filename:
            flash("Choose an image file.", "error")
            return redirect(url_for("evaluation"))
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in cfg["ALLOWED_IMAGE_EXTENSIONS"]:
            flash("Only JPG or PNG images are allowed.", "error")
            return redirect(url_for("evaluation"))
        img = decode_image_bytes(f.read(), cfg["MAX_IMAGE_DIMENSION"])
        if img is None:
            flash("That file is not a valid image.", "error")
            return redirect(url_for("evaluation"))
        try:
            people = M.parse_people(request.form.get("people"))
            boxes = M.parse_boxes(request.form.get("boxes"))
        except ValueError as exc:
            flash(f"Ground truth problem: {exc}", "error")
            return redirect(url_for("evaluation"))
        known = {s["roll_no"].lower() for s in db.list_students()}
        bad = [p for p in people if p != M.UNKNOWN and p.lower() not in known]
        if bad:
            flash(f"Not a registered roll number: {', '.join(bad)}. Use UNKNOWN for people who are not registered.", "error")
            return redirect(url_for("evaluation"))
        if kind == "individual" and len(people) != 1:
            flash("An individual image must contain exactly one person.", "error")
            return redirect(url_for("evaluation"))
        if boxes is not None and len(boxes) != len(people):
            flash("Number of boxes must equal number of people (same order).", "error")
            return redirect(url_for("evaluation"))
        sub = "individual" if kind == "individual" else "groups"
        name = f"{uuid.uuid4().hex[:12]}.jpg"
        cv2.imwrite(str(Path(cfg["TEST_DIR"]) / sub / name), img)  # re-encoded: strips metadata / payloads
        append_ground_truth(cfg["GROUND_TRUTH_CSV"], {
            "image": f"{sub}/{name}", "type": kind,
            "people": ";".join(known_case(db, p) for p in people),
            "boxes": request.form.get("boxes", "").strip(), "notes": request.form.get("notes", "").strip()[:200]})
        flash(f"Saved test image {sub}/{name} with its ground truth.", "ok")
        return redirect(url_for("evaluation"))

    return app


def known_case(db, p):
    if p == M.UNKNOWN:
        return M.UNKNOWN
    s = db.get_by_roll(p)
    return s["roll_no"] if s else p


class _Cfg:
    """Attribute view over Flask config so evaluation code can use cfg.NAME."""
    def __init__(self, d):
        self._d = d

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(k)


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False, threaded=True)
