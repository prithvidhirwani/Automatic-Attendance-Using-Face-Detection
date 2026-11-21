import pytest

from evaluation import metrics as M
from evaluation.common import compute_summary


def D(label, box=(0, 0, 100, 100)):
    return {"box": box, "label": label}


def test_iou_and_matching():
    assert M.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert M.iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    m, ug, ud = M.match_boxes([(0, 0, 10, 10), (50, 50, 10, 10)], [(1, 1, 10, 10)])
    assert m == [(0, 0)] and ug == [1] and ud == []


def test_spec_example_four_of_six_is_66_67():
    people = [f"R{i}" for i in range(6)]
    boxes = [(i * 100, 0, 90, 90) for i in range(6)]
    dets = [D(people[i], boxes[i]) for i in range(4)] + [D("Unknown", boxes[4]), D("R9", boxes[5])]
    r = M.score_image(people, dets, boxes)
    assert r["correct_identities"] == 4 and r["registered_actual"] == 6
    assert r["identity_accuracy"] == 66.67
    assert r["false_unknown"] == 1 and r["wrong_identities"] == 1 and r["box_verified"]


def test_detected_count_is_not_correct_count():
    people = ["R1", "R2", "R3"]
    boxes = [(0, 0, 50, 50), (100, 0, 50, 50), (200, 0, 50, 50)]
    dets = [D("R9", b) for b in boxes]  # all detected, all wrong
    r = M.score_image(people, dets, boxes)
    assert r["faces_detected"] == 3 and r["correct_identities"] == 0 and r["identity_accuracy"] == 0.0


def test_missed_and_false_detections_with_boxes():
    people = ["R1", "R2"]
    boxes = [(0, 0, 50, 50), (100, 0, 50, 50)]
    dets = [D("R1", (0, 0, 50, 50)), D("R5", (400, 400, 50, 50))]  # R2 missed, one false detection
    r = M.score_image(people, dets, boxes)
    assert (r["tp_faces"], r["missed_faces"], r["false_detections"]) == (1, 1, 1)
    assert r["correct_identities"] == 1 and r["identity_accuracy"] == 50.0


def test_no_faces_detected():
    r = M.score_image(["R1"], [])
    assert r["faces_detected"] == 0 and r["missed_faces"] == 1 and r["identity_accuracy"] == 0.0
    assert M.decode_pairs(r["pairs"]) == [("R1", M.MISSED)]


def test_unknown_person_rejection_and_false_accept():
    ok = M.score_image(["Unknown"], [D("Unknown")])
    assert ok["correct_rejections"] == 1 and ok["unknown_rejection_rate"] == 100.0 and ok["identity_accuracy"] is None
    bad = M.score_image(["unknown"], [D("R3")])
    assert bad["false_accepts"] == 1 and bad["unknown_rejection_rate"] == 0.0


def test_single_mode_uses_largest_face():
    r = M.score_image(["R1"], [D("R9", (0, 0, 20, 20)), D("R1", (0, 0, 90, 90))])
    assert r["mode"] == "single" and r["correct_identities"] == 1 and r["false_detections"] == 1
    assert r["box_verified"] is False


def test_set_mode_group_without_boxes():
    r = M.score_image(["R1", "R2", "R3"], [D("R1"), D("R2")])
    assert r["mode"] == "set" and r["correct_identities"] == 2 and r["missed_faces"] == 1
    assert r["identity_accuracy"] == 66.67


def test_aggregate_uses_counts_not_mean_of_percentages():
    a = M.score_image(["R1"], [D("R1")])                       # 1/1 = 100 %
    big = M.score_image(["R1", "R2", "R3", "R4"], [D("R1")])   # 1/4 = 25 %
    agg = M.aggregate([a, big])
    assert agg["identity_accuracy"] == 40.0                    # 2/5, NOT (100+25)/2 = 62.5


def test_detection_pr_only_from_box_verified_images():
    boxed = M.score_image(["R1"], [D("R1", (0, 0, 10, 10)), D("R1", (50, 50, 10, 10))], [(0, 0, 10, 10)])
    plain = M.score_image(["R1"], [D("R1")])
    agg = M.aggregate([boxed, plain])
    assert agg["box_verified_images"] == 1
    assert agg["detection_precision"] == 50.0 and agg["detection_recall"] == 100.0
    none = M.aggregate([plain])
    assert none["detection_precision"] is None and none["detection_recall"] is None


def test_empty_dataset_is_not_measurable():
    agg = M.aggregate([])
    assert agg["images"] == 0 and agg["identity_accuracy"] is None and agg["detection_recall"] is None
    assert M.multiclass_report([]) is None
    summary = dict(((s, m), v) for s, m, v in compute_summary([], [], [], 34))
    assert summary[("overall", "identity_accuracy")] is None
    assert summary[("coverage", "test_complete")] is False


def test_multiclass_report_with_missed():
    pairs = [("A", "A"), ("A", "B"), ("B", "B"), ("B", M.MISSED)]
    rep = M.multiclass_report(pairs)
    assert rep["per_class"]["A"]["precision"] == 1.0 and rep["per_class"]["A"]["recall"] == 0.5
    assert rep["per_class"]["B"]["precision"] == 0.5 and rep["per_class"]["B"]["recall"] == 0.5
    assert rep["macro"]["recall"] == 0.5 and rep["weighted"]["f1"] == pytest.approx(0.5833, abs=1e-3)


def test_parse_helpers():
    assert M.parse_people("R1; unknown ,R2") == ["R1", "Unknown", "R2"]
    assert M.parse_boxes("1,2,3,4;5,6,7,8") == [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert M.parse_boxes("") is None
    with pytest.raises(ValueError):
        M.parse_boxes("1,2,3")
    with pytest.raises(ValueError):
        M.parse_people("  ")
