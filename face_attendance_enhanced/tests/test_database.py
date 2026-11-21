import sqlite3
from datetime import datetime

import pytest

from database import Database, DuplicateRollError


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "x.db")
    d.init_db()
    return d


def test_init_creates_tables_and_is_idempotent(db):
    db.init_db()
    with sqlite3.connect(db.path) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"students", "attendance", "evaluation_runs"} <= names


def test_duplicate_roll_rejected_case_insensitive(db):
    db.add_student("21IT001", "Asha Rao")
    with pytest.raises(DuplicateRollError):
        db.add_student("21it001", "Someone Else")
    assert db.count_students() == 1


@pytest.mark.parametrize("roll,name", [("", "A"), ("bad roll!", "A"), ("R1", ""), ("unknown", "A"), ("a;b", "A")])
def test_invalid_students_rejected(db, roll, name):
    with pytest.raises(ValueError):
        db.add_student(roll, name)


def test_sql_injection_text_is_stored_literally(db):
    db.add_student("R1", "Robert'); DROP TABLE students;--")
    assert db.count_students() == 1
    assert "DROP TABLE" in db.list_students()[0]["full_name"]
    assert db.list_students("DROP")  # search term is parameterised, no crash


def test_attendance_unique_per_day_and_stats(db):
    a = db.add_student("R1", "A")
    db.add_student("R2", "B")
    t = datetime(2026, 1, 5, 9, 0)
    assert db.mark_attendance(a, 40.0, t) is True
    assert db.mark_attendance(a, 41.0, t.replace(hour=10)) is False       # same day
    assert db.mark_attendance(a, 40.0, datetime(2026, 1, 6, 9)) is True   # next day
    s = db.dashboard_stats(t.date())
    assert (s["total"], s["present"], s["absent"], s["percentage"]) == (2, 1, 1, 50.0)


def test_empty_database_stats_have_no_percentage(db):
    s = db.dashboard_stats()
    assert s["total"] == 0 and s["percentage"] is None


def test_delete_cascades_attendance(db):
    a = db.add_student("R1", "A")
    db.mark_attendance(a, 1.0)
    assert db.delete_student(a) is True
    assert db.attendance_records() == []
    assert db.delete_student(a) is False
