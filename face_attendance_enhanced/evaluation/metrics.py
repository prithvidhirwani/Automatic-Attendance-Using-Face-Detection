"""Pure metric functions (no OpenCV / Flask) so they can be unit-tested.

Vocabulary
  actual faces      people really present in the image (from ground_truth.csv)
  registered face   an actual face of a registered student
  unknown face      an actual face of someone NOT registered (written UNKNOWN)
"""
from sklearn.metrics import precision_recall_fscore_support

UNKNOWN = "Unknown"
MISSED = "MISSED"          # actual face that no detection was matched to
SUM_FIELDS = [
    "actual_faces", "faces_detected", "registered_actual", "unknown_actual",
    "correct_identities", "wrong_identities", "false_unknown",
    "correct_rejections", "false_accepts", "missed_faces", "false_detections",
]


# ---------------------------------------------------------------- parsing
def norm_person(p):
    p = (p or "").strip()
    return UNKNOWN if p.lower() == "unknown" else p


def parse_people(text):
    people = [norm_person(p) for p in (text or "").replace(",", ";").split(";") if p.strip()]
    if not people:
        raise ValueError("people column is empty")
    return people


def parse_boxes(text):
    """'x,y,w,h;x,y,w,h' -> [(x,y,w,h), ...]; blank -> None."""
    if not text or not text.strip():
        return None
    boxes = []
    for part in text.split(";"):
        nums = part.strip().split(",")
        if len(nums) != 4:
            raise ValueError(f"bad box '{part}' (need x,y,w,h)")
        x, y, w, h = (int(float(n)) for n in nums)
        if w <= 0 or h <= 0:
            raise ValueError(f"box '{part}' has non-positive size")
        boxes.append((x, y, w, h))
    return boxes


def encode_pairs(pairs):
    return ";".join(f"{t}>{p}" for t, p in pairs)


def decode_pairs(text):
    out = []
    for item in (text or "").split(";"):
        if ">" in item:
            t, p = item.split(">", 1)
            out.append((t, p))
    return out


# ---------------------------------------------------------------- detection
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, det_boxes, thr=0.5):
    """Greedy one-to-one matching by descending IoU. Returns (matches, unmatched_gt, unmatched_det)."""
    cands = sorted(
        ((iou(g, d), gi, di) for gi, g in enumerate(gt_boxes) for di, d in enumerate(det_boxes)),
        reverse=True,
    )
    used_g, used_d, matches = set(), set(), []
    for score, gi, di in cands:
        if score < thr:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        matches.append((gi, di))
    return (
        matches,
        [i for i in range(len(gt_boxes)) if i not in used_g],
        [i for i in range(len(det_boxes)) if i not in used_d],
    )


# ---------------------------------------------------------------- one image
def score_image(gt_people, detections, gt_boxes=None, iou_thr=0.5):
    """Score one image.

    gt_people   list of roll numbers / 'Unknown' (one per actual face)
    detections  list of {"box": (x,y,w,h), "label": roll_no | 'Unknown'}
    gt_boxes    optional list of boxes aligned with gt_people

    Matching modes
      box    gt_boxes given: detections matched to annotations by IoU. Detection
             precision/recall are measured; identity is judged per face.
      single exactly one actual face, no boxes: the LARGEST detection is the
             prediction (detection P/R are NOT measured).
      set    several faces, no boxes: predicted labels are matched to the
             people list as a multiset (position is unknown). Detection P/R are
             NOT measured; identity numbers are an approximation.
    """
    gt_people = [norm_person(p) for p in gt_people]
    n = len(gt_people)
    reg_idx = [i for i, p in enumerate(gt_people) if p != UNKNOWN]
    unk_idx = [i for i, p in enumerate(gt_people) if p == UNKNOWN]

    r = {k: 0 for k in SUM_FIELDS}
    r.update(actual_faces=n, faces_detected=len(detections),
             registered_actual=len(reg_idx), unknown_actual=len(unk_idx))
    pairs = []  # (true_label, predicted_label) per actual face

    def judge(true, pred):
        if true != UNKNOWN:
            if pred == true:
                r["correct_identities"] += 1
            elif pred == UNKNOWN:
                r["false_unknown"] += 1
            elif pred == MISSED:
                r["missed_faces_id"] = r.get("missed_faces_id", 0) + 1
            else:
                r["wrong_identities"] += 1
        else:
            if pred == UNKNOWN:
                r["correct_rejections"] += 1
            elif pred != MISSED:
                r["false_accepts"] += 1
        pairs.append((true, pred))

    if gt_boxes is not None and len(gt_boxes) == n:
        mode = "box"
        matches, un_g, un_d = match_boxes(gt_boxes, [d["box"] for d in detections], iou_thr)
        r["tp_faces"], r["false_detections"], r["missed_faces"] = len(matches), len(un_d), len(un_g)
        pred_for = {gi: detections[di]["label"] for gi, di in matches}
        for i in range(n):
            judge(gt_people[i], pred_for.get(i, MISSED))
    elif n == 1:
        mode = "single"
        r["tp_faces"] = None
        r["missed_faces"] = 1 if not detections else 0
        r["false_detections"] = max(len(detections) - 1, 0)
        if detections:
            best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
            judge(gt_people[0], best["label"])
        else:
            judge(gt_people[0], MISSED)
    else:
        mode = "set"
        r["tp_faces"] = None
        r["missed_faces"] = max(n - len(detections), 0)
        r["false_detections"] = max(len(detections) - n, 0)
        preds = [d["label"] for d in detections]
        outcome = {}
        for i in reg_idx:                       # 1) exact identity matches
            if gt_people[i] in preds:
                preds.remove(gt_people[i])
                outcome[i] = gt_people[i]
        for i in unk_idx:                       # 2) unknown people take 'Unknown' predictions
            if UNKNOWN in preds:
                preds.remove(UNKNOWN)
                outcome[i] = UNKNOWN
        for i in range(n):                      # 3) leftovers: wrong label, else missed
            if i in outcome:
                continue
            if gt_people[i] != UNKNOWN and UNKNOWN in preds:
                preds.remove(UNKNOWN)
                outcome[i] = UNKNOWN
            elif preds:
                outcome[i] = preds.pop(0)
            else:
                outcome[i] = MISSED
        for i in range(n):
            judge(gt_people[i], outcome[i])

    # A missed registered face is not a 'wrong identity' but it is not correct either.
    r.pop("missed_faces_id", None)
    r["mode"] = mode
    r["box_verified"] = mode == "box"
    r["pairs"] = encode_pairs(pairs)
    r["identity_accuracy"] = pct(r["correct_identities"], r["registered_actual"])
    r["unknown_rejection_rate"] = pct(r["correct_rejections"], r["unknown_actual"])
    return r


def pct(num, den):
    return round(num / den * 100, 2) if den else None


# ---------------------------------------------------------------- aggregate
def _num(v):
    if v in (None, ""):
        return None
    return float(v)


def aggregate(rows):
    """Aggregate from TOTAL COUNTS (never an average of per-image percentages)."""
    t = {k: 0 for k in SUM_FIELDS}
    tp = fp = fn = 0
    verified = 0
    for row in rows:
        for k in SUM_FIELDS:
            t[k] += int(_num(row.get(k)) or 0)
        if str(row.get("box_verified")).lower() in ("true", "1"):
            verified += 1
            tp += int(_num(row.get("tp_faces")) or 0)
            fp += int(_num(row.get("false_detections")) or 0)
            fn += int(_num(row.get("missed_faces")) or 0)
    out = dict(t)
    out["images"] = len(rows)
    out["box_verified_images"] = verified
    out["identity_accuracy"] = pct(t["correct_identities"], t["registered_actual"])
    out["unknown_rejection_rate"] = pct(t["correct_rejections"], t["unknown_actual"])
    out["detection_precision"] = pct(tp, tp + fp) if verified else None
    out["detection_recall"] = pct(tp, tp + fn) if verified else None
    out["detection_tp"], out["detection_fp"], out["detection_fn"] = (tp, fp, fn) if verified else (None,) * 3
    return out


def multiclass_report(pairs):
    """Precision/recall/F1 per identity + macro/weighted averages.

    pairs: [(true, predicted)]. Classes are the labels present in the ground
    truth (registered identities and, if present, Unknown). A missed face is
    predicted as MISSED: it lowers recall of its true class and counts as no
    one's false positive. Predictions of people absent from the ground truth
    lower recall of the true class but cannot lower any listed precision."""
    if not pairs:
        return None
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    labels = sorted(set(y_true))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per = {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                 "f1": round(float(f[i]), 4), "support": int(s[i])} for i, lab in enumerate(labels)}
    avg = {}
    for kind in ("macro", "weighted"):
        ap, ar, af, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=kind, zero_division=0)
        avg[kind] = {"precision": round(float(ap), 4), "recall": round(float(ar), 4), "f1": round(float(af), 4)}
    return {"per_class": per, **avg, "total_faces": len(pairs)}
