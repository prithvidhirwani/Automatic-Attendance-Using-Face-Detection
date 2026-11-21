"""SQLite access layer. All queries are parameterised."""
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date as date_cls
from pathlib import Path

ROLL_RE = re.compile(r"^[A-Za-z0-9._/-]{1,30}$")
RESERVED_ROLLS = {"unknown", "missed"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no     TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name   TEXT NOT NULL,
    department  TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    att_date    TEXT NOT NULL,          -- YYYY-MM-DD
    marked_at   TEXT NOT NULL,          -- ISO timestamp
    status      TEXT NOT NULL DEFAULT 'Present',
    distance    REAL,                   -- LBPH distance at the time (lower = closer)
    UNIQUE (student_id, att_date)       -- policy: one record per student per day
);
CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(att_date);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    threshold   REAL,
    equalize    INTEGER,
    images      INTEGER
);
"""


class DuplicateRollError(ValueError):
    pass


class Database:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        with self.connect() as c:
            c.executescript(SCHEMA)

    # ---- students ------------------------------------------------------
    @staticmethod
    def validate_student(roll_no, full_name, department):
        roll_no = (roll_no or "").strip()
        full_name = " ".join((full_name or "").split())
        department = " ".join((department or "").split())
        if not full_name or len(full_name) > 100:
            raise ValueError("Full name is required (max 100 characters).")
        if not ROLL_RE.match(roll_no):
            raise ValueError("Roll number must be 1-30 characters: letters, digits, '.', '_', '-', '/'.")
        if roll_no.lower() in RESERVED_ROLLS:
            raise ValueError("That roll number is reserved.")
        if len(department) > 60:
            raise ValueError("Class / department is too long (max 60 characters).")
        return roll_no, full_name, department

    def add_student(self, roll_no, full_name, department=""):
        roll_no, full_name, department = self.validate_student(roll_no, full_name, department)
        try:
            with self.connect() as c:
                cur = c.execute(
                    "INSERT INTO students (roll_no, full_name, department, created_at) VALUES (?,?,?,?)",
                    (roll_no, full_name, department, datetime.now().isoformat(timespec="seconds")),
                )
                return cur.lastrowid
        except sqlite3.IntegrityError:
            raise DuplicateRollError(f"Roll number '{roll_no}' is already registered.")

    def list_students(self, q=""):
        q = (q or "").strip()
        with self.connect() as c:
            if q:
                like = f"%{q}%"
                rows = c.execute(
                    "SELECT * FROM students WHERE full_name LIKE ? OR roll_no LIKE ? OR department LIKE ? ORDER BY roll_no",
                    (like, like, like),
                ).fetchall()
            else:
                rows = c.execute("SELECT * FROM students ORDER BY roll_no").fetchall()
        return [dict(r) for r in rows]

    def get_student(self, student_id):
        with self.connect() as c:
            r = c.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        return dict(r) if r else None

    def get_by_roll(self, roll_no):
        with self.connect() as c:
            r = c.execute("SELECT * FROM students WHERE roll_no = ?", ((roll_no or "").strip(),)).fetchone()
        return dict(r) if r else None

    def delete_student(self, student_id):
        """Deletes the student and (via ON DELETE CASCADE) their attendance rows."""
        with self.connect() as c:
            cur = c.execute("DELETE FROM students WHERE id = ?", (student_id,))
            return cur.rowcount > 0

    def count_students(self):
        with self.connect() as c:
            return c.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    # ---- attendance ----------------------------------------------------
    def mark_attendance(self, student_id, distance=None, when=None):
        """Returns True if a new record was created, False if already marked that day."""
        when = when or datetime.now()
        with self.connect() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO attendance (student_id, att_date, marked_at, status, distance) VALUES (?,?,?,?,?)",
                (student_id, when.date().isoformat(), when.isoformat(timespec="seconds"), "Present", distance),
            )
            return cur.rowcount == 1

    def dashboard_stats(self, day=None):
        day = (day or date_cls.today()).isoformat()
        with self.connect() as c:
            total = c.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            present = c.execute(
                "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE att_date = ?", (day,)
            ).fetchone()[0]
        absent = max(total - present, 0)
        pct = round(present / total * 100, 1) if total else None
        return {"total": total, "present": present, "absent": absent, "percentage": pct, "date": day}

    def recent_attendance(self, limit=10):
        with self.connect() as c:
            rows = c.execute(
                """SELECT a.marked_at, a.status, a.distance, s.full_name, s.roll_no
                   FROM attendance a JOIN students s ON s.id = a.student_id
                   ORDER BY a.marked_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def attendance_records(self, date_from=None, date_to=None, q=""):
        sql = """SELECT a.att_date, a.marked_at, a.status, a.distance, s.full_name, s.roll_no, s.department
                 FROM attendance a JOIN students s ON s.id = a.student_id WHERE 1=1"""
        params = []
        if date_from:
            sql += " AND a.att_date >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND a.att_date <= ?"
            params.append(date_to)
        if q and q.strip():
            sql += " AND (s.full_name LIKE ? OR s.roll_no LIKE ?)"
            params += [f"%{q.strip()}%"] * 2
        sql += " ORDER BY a.marked_at DESC"
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def log_evaluation_run(self, kind, threshold, equalize, images):
        with self.connect() as c:
            c.execute(
                "INSERT INTO evaluation_runs (kind, created_at, threshold, equalize, images) VALUES (?,?,?,?,?)",
                (kind, datetime.now().isoformat(timespec="seconds"), threshold, int(bool(equalize)), images),
            )
