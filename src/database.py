"""SQLite persistence layer.

Two tables:
    students   -> one row per enrolled person
    attendance -> one row per attendance event (timestamped)

Using a real relational store (rather than flat CSV files) gives us
transactional writes, duplicate-prevention constraints, and fast
date-range queries for reporting - see docs/er_diagram.md for the schema.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional

from src.exceptions import StudentAlreadyExistsError, StudentNotFoundError
from src.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    label_id    INTEGER UNIQUE NOT NULL,
    registered_on TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    date        TEXT NOT NULL,
    time        TEXT NOT NULL,
    confidence  REAL NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    UNIQUE(student_id, date)
);
"""


@dataclass
class Student:
    student_id: str
    name: str
    label_id: int
    registered_on: str


@dataclass
class AttendanceRecord:
    student_id: str
    name: str
    date: str
    time: str
    confidence: float


class Database:
    """Thin, explicit wrapper around sqlite3 (no ORM, by design - keeps the
    course's data-structure/SQL concepts visible rather than hidden)."""

    def __init__(self, db_file: str):
        self.db_file = db_file
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.info("Database ready at %s", db_file)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Students
    # ------------------------------------------------------------------
    def add_student(self, student_id: str, name: str) -> int:
        if self.get_student(student_id) is not None:
            raise StudentAlreadyExistsError(f"Student '{student_id}' is already registered.")

        with self._connect() as conn:
            next_label = conn.execute(
                "SELECT COALESCE(MAX(label_id), -1) + 1 AS next_id FROM students"
            ).fetchone()["next_id"]
            conn.execute(
                "INSERT INTO students (student_id, name, label_id, registered_on) "
                "VALUES (?, ?, ?, ?)",
                (student_id, name, next_label, datetime.now().isoformat(timespec="seconds")),
            )
        logger.info("Registered student %s (%s) with label_id=%s", student_id, name, next_label)
        return next_label

    def get_student(self, student_id: str) -> Optional[Student]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM students WHERE student_id = ?", (student_id,)
            ).fetchone()
        return Student(row["student_id"], row["name"], row["label_id"], row["registered_on"]) if row else None

    def get_student_by_label(self, label_id: int) -> Optional[Student]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM students WHERE label_id = ?", (label_id,)
            ).fetchone()
        return Student(row["student_id"], row["name"], row["label_id"], row["registered_on"]) if row else None

    def list_students(self) -> List[Student]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
        return [Student(r["student_id"], r["name"], r["label_id"], r["registered_on"]) for r in rows]

    # ------------------------------------------------------------------
    # Attendance
    # ------------------------------------------------------------------
    def mark_attendance(self, student_id: str, confidence: float, once_per_day: bool = True) -> bool:
        """Insert an attendance row. Returns True if a new row was inserted,
        False if the student was already marked present today (idempotent)."""
        student = self.get_student(student_id)
        if student is None:
            raise StudentNotFoundError(f"No such student: {student_id}")

        now = datetime.now()
        date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO attendance (student_id, name, date, time, confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (student_id, student.name, date_str, time_str, confidence),
                )
            logger.info("Marked attendance: %s at %s %s", student_id, date_str, time_str)
            return True
        except sqlite3.IntegrityError:
            if once_per_day:
                logger.debug("Attendance already marked today for %s", student_id)
                return False
            raise

    def get_attendance(self, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> List[AttendanceRecord]:
        query = "SELECT * FROM attendance"
        params: list = []
        if start_date and end_date:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start_date, end_date]
        elif start_date:
            query += " WHERE date = ?"
            params = [start_date]
        query += " ORDER BY date, time"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AttendanceRecord(r["student_id"], r["name"], r["date"], r["time"], r["confidence"]) for r in rows]
