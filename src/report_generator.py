"""Reporting & analytics module.

Turns raw attendance rows from the database into:
  * a CSV export
  * a printed console summary
  * a bar chart (PNG) of attendance count per student

This is the "Reporting / analytics" functional module required by the
course's project rubric.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import date, timedelta
from typing import List, Optional

from src.config import AppConfig
from src.database import AttendanceRecord, Database
from src.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db

    # ------------------------------------------------------------------
    def _records(self, start: Optional[str], end: Optional[str]) -> List[AttendanceRecord]:
        return self.db.get_attendance(start, end)

    def export_csv(self, records: List[AttendanceRecord], filename: str) -> str:
        path = os.path.join(self.config.reports_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["student_id", "name", "date", "time", "confidence"])
            for r in records:
                writer.writerow([r.student_id, r.name, r.date, r.time, f"{r.confidence:.2f}"])
        logger.info("Wrote %d attendance rows to %s", len(records), path)
        return path

    def daily_report(self, day: Optional[str] = None):
        day = day or date.today().strftime("%Y-%m-%d")
        records = self._records(day, day)
        csv_path = self.export_csv(records, f"attendance_{day}.csv")
        summary = {
            "date": day,
            "total_present": len(records),
            "students": [f"{r.name} ({r.student_id}) @ {r.time}" for r in records],
        }
        return summary, csv_path

    def range_report(self, start: str, end: str, chart: bool = True):
        records = self._records(start, end)
        csv_path = self.export_csv(records, f"attendance_{start}_to_{end}.csv")

        counts = Counter(r.name for r in records)
        chart_path = None
        if chart and counts:
            chart_path = self._bar_chart(counts, start, end)

        summary = {
            "start": start,
            "end": end,
            "total_records": len(records),
            "attendance_by_student": dict(counts),
        }
        return summary, csv_path, chart_path

    def _bar_chart(self, counts: Counter, start: str, end: str) -> str:
        import matplotlib
        matplotlib.use("Agg")  # headless/CLI-safe backend
        import matplotlib.pyplot as plt

        names = list(counts.keys())
        values = [counts[n] for n in names]

        plt.figure(figsize=(8, 4.5))
        plt.bar(names, values, color="#3A7CA5")
        plt.title(f"Attendance count ({start} to {end})")
        plt.ylabel("Days present")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        chart_path = os.path.join(self.config.reports_dir, f"attendance_chart_{start}_to_{end}.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info("Saved attendance chart to %s", chart_path)
        return chart_path

    @staticmethod
    def last_n_days_range(n: int = 7):
        end = date.today()
        start = end - timedelta(days=n - 1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
