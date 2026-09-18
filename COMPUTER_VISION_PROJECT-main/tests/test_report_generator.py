import csv
import os
from datetime import date

from src.report_generator import ReportGenerator


def test_daily_report_empty(tmp_config, tmp_db):
    gen = ReportGenerator(tmp_config, tmp_db)
    summary, csv_path = gen.daily_report()
    assert summary["total_present"] == 0
    assert os.path.isfile(csv_path)


def test_daily_report_with_records(tmp_config, tmp_db):
    tmp_db.add_student("S001", "Amit")
    tmp_db.add_student("S002", "Bela")
    tmp_db.mark_attendance("S001", confidence=30.0)
    tmp_db.mark_attendance("S002", confidence=40.0)

    gen = ReportGenerator(tmp_config, tmp_db)
    today = date.today().strftime("%Y-%m-%d")
    summary, csv_path = gen.daily_report(today)

    assert summary["total_present"] == 2
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["student_id", "name", "date", "time", "confidence"]
    assert len(rows) == 3  # header + 2 records


def test_range_report_generates_chart(tmp_config, tmp_db):
    tmp_db.add_student("S001", "Amit")
    tmp_db.mark_attendance("S001", confidence=30.0)

    gen = ReportGenerator(tmp_config, tmp_db)
    start, end = ReportGenerator.last_n_days_range(7)
    summary, csv_path, chart_path = gen.range_report(start, end)

    assert summary["total_records"] == 1
    assert os.path.isfile(csv_path)
    assert chart_path is not None
    assert os.path.isfile(chart_path)


def test_last_n_days_range_length():
    start, end = ReportGenerator.last_n_days_range(7)
    assert start <= end
