# Face Attendance Enhanced

Flask + OpenCV (Haar cascade detection, LBPH recognition) + SQLite attendance system with an evaluation module for
individual and group images. Vanilla HTML/CSS/JS, no React.

> **Privacy notice.** Face images are biometric data. Before capturing, tell each student what is stored, why, and for
> how long, and get their consent. Keep this computer's `dataset/` and `instance/` folders private.

## Status (read this first)
* Unit/integration tests (40) pass: database, routes, metrics, model-file errors, empty datasets, unknown identities,
  images with no faces, and a synthetic end-to-end evaluation run.
* **Not yet validated:** live webcam recognition and real-world accuracy. No student photos are included and no accuracy
  figure is shipped. Every number in the UI comes from *your* labelled test images; with none, the page says the
  evaluation has not been performed.

## Windows PowerShell quick start (Python 3.11.9)
```powershell
cd path\to\face_attendance_enhanced
powershell -ExecutionPolicy Bypass -File .\run.ps1      # creates .venv, installs, inits DB, opens browser
```
`-ExecutionPolicy Bypass` applies to that one command only. Or step by step:
```powershell
cd path\to\face_attendance_enhanced
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from database import Database; from config import Config; Database(Config.DATABASE_PATH).init_db()"
.\.venv\Scripts\python.exe app.py
start http://127.0.0.1:5000
```
Optional activation: `.\.venv\Scripts\Activate.ps1` (if blocked: `Set-ExecutionPolicy -Scope Process Bypass`, which lasts only for that window).
Run tests: `.\run.ps1 -Test`. Use Chrome or Edge; the browser asks for camera permission (works on `127.0.0.1`).

## Workflow
1. **Register students** (Register student) - name, unique roll number, optional class.
2. **Capture training images** on the student page: 20+ varied images each (min 10). One face, sharp, well lit.
3. **Train** (Students -> Train model). The model is saved to `models/` and loaded without retraining.
4. **Live attendance**: attendance is marked only after the same student is recognised in 3 consecutive frames with
   LBPH distance <= threshold. One record per student per day (SQLite `UNIQUE(student_id, att_date)`).
5. **Capture independent test images** (Model evaluation -> add labelled test image, by file or webcam). Use different
   sessions/days from training. Test images go to `test_dataset/` and are never used for training.
6. **Run evaluation** (individual, group, or both). Results are written to
   `results/individual_results.csv`, `results/group_results.csv`, `results/summary_metrics.csv` and shown in the UI.
7. **Export**: download the CSVs from the evaluation page or copy them from `results/`.
CLI alternative: `.\.venv\Scripts\python.exe -m evaluation.evaluate_individual` and `-m evaluation.evaluate_groups`
(`--threshold 70` optional).

## Ground truth (`test_dataset/ground_truth.csv`)
```
image,type,people,boxes,notes
individual/a1b2.jpg,individual,21IT001,,
groups/c3d4.jpg,group,21IT001;21IT007;UNKNOWN,"40,60,120,120;210,55,118,118;380,70,115,115",front row
```
* `image` is relative to `test_dataset/`. `type` is `individual` or `group`.
* `people` lists a roll number for each actual face, `;` separated; `UNKNOWN` = not registered. Faces = number of entries.
* `boxes` (optional) = `x,y,w,h` in pixels per person, **same order as `people`**. Needed for detection precision/recall.
* The 34-identity test: one individual image for each of 34 *registered* students. The page reports
  "test incomplete" until 34 distinct registered identities have individual test images.

## How metrics are computed
* **Image identity accuracy** = correctly identified registered faces / registered actual faces x 100
  (4 of 6 = 66.67%). Missed faces count as not identified. Detected-face count is never used as "correct".
* **Aggregate** accuracy = total correct / total actual faces (counts are summed; percentages are not averaged).
* **Unknown rejection rate** = unknown people labelled Unknown / unknown people. Reported separately.
* **Detection precision/recall**: only for images with `boxes`; detection matches an annotation at IoU >= 0.5.
  Otherwise shown as n/a. (Count-based "missed / false detections" in no-box images are only count differences.)
* **Matching modes**: `box` (judged per face), `single` (individual image: largest detected face), `set` (group without
  boxes: names are matched as a multiset, position unknown - approximate).
* **Precision/recall/F1** per identity from (actual, predicted) pairs (scikit-learn), macro and weighted averages.
  Missed faces are predicted `MISSED`: they lower recall of that identity and are nobody's false positive.
  Unmatched false detections are counted in detection precision only.
* Empty denominators give `n/a` (blank in CSV), never 0 or a placeholder.

## Why accuracy may fall in group photos (things to test)
Smaller faces (lower resolution after Haar scaling), non-frontal poses and occlusion, uneven lighting across the group,
more false detections, and the LBPH grid being sensitive to crop alignment. Compare individual vs group results at the
same threshold; vary one factor at a time (e.g. photo distance, equalisation on/off) and report with the settings
recorded in `summary_metrics.csv`.

## Settings and the threshold
LBPH returns a **distance** (lower = more similar), *not* a probability. A face is "Unknown" when distance > threshold
(`FACE_LBPH_THRESHOLD`, default 70, also editable per evaluation run). Choose it on a separate validation set, then evaluate
once on the test set; do not keep tuning on the test images. Other env vars: `FACE_EQUALIZE` (1/0), `FACE_DETECT_SCALE`,
`FACE_DETECT_NEIGHBORS`, `FACE_DETECT_MIN_SIZE`, `FACE_CONFIRM_FRAMES`, `FACE_MIN_CAPTURE_SIZE`, `FACE_MIN_SHARPNESS`.
Equalisation is not guaranteed to help: train with it on, evaluate; train with it off, evaluate; compare. Example:
`$env:FACE_LBPH_THRESHOLD = "65"; .\run.ps1`.

## Security and data removal
* Uploads: JPG/PNG only, 8 MB max, decoded and re-encoded (never executed); stored only under `test_dataset/`/`dataset/train/`.
* `instance/` (database, secret key) and `models/` are not served by any route. CSRF token on all POST requests.
  CSV exports neutralise spreadsheet formulas. The secret key is generated into `instance/secret_key.txt`, not hard-coded.
* The app binds to `127.0.0.1` only. It has no login: do not expose it to a network without adding authentication.
* **Delete a student**: Students -> student -> Delete (removes the record, attendance and `dataset/train/<id>/`), then retrain.
  Their old test images / ground-truth rows are not deleted automatically: remove the files and rows by hand.
  **Delete everything**: stop the app, then delete `instance/`, `dataset/train/*`, `models/*`, `test_dataset/*`, `results/*`.

## Limitations
LBPH + Haar are classical methods: sensitive to lighting, pose and expression, and weak for small or angled faces. Photos
and screens can spoof it (no liveness check). One camera frame is processed at a time (about 2 per second). Group boxes
must be annotated by hand. Not tested here: live webcam, Windows, real faces.
