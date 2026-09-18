# Screenshots

All images below are captured from **actual runs** of this codebase
against the bundled synthetic demo dataset
(`scripts/generate_synthetic_faces.py`), not mockups.

1. **01_enroll_and_train.png** — enrolling two synthetic identities from
   pre-cropped image folders and training the LBPH model.
2. **02_recognize_and_list.png** — running recognition on a held-out
   sample image for each identity (attendance gets marked), then
   listing enrolled students.
3. **03_report_range.png** — a 7-day range attendance report (CSV +
   chart generation).
4. **04_pytest_suite.png** — the full automated test suite (21 tests)
   passing.
5. **05_attendance_chart.png** — the matplotlib bar chart produced by
   the reporting module for a date-range report.
