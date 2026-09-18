#!/usr/bin/env python3
"""
Face Recognition Attendance System - Command Line Interface
=============================================================

A fully CLI-driven computer-vision project: enroll students (from a
webcam or from a folder of photos), train an LBPH face-recognition
model, run recognition against a webcam / video / image, and generate
attendance reports - no GUI required.

Run `python main.py -h` for full usage, or `python main.py <command> -h`
for help on a specific subcommand.
"""

from __future__ import annotations

import argparse
import sys

from src.attendance_manager import AttendanceManager
from src.config import load_config
from src.database import Database
from src.exceptions import AttendanceSystemError
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_trainer import FaceTrainer
from src.logger import configure_logging, get_logger
from src.report_generator import ReportGenerator

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Command handlers
# ----------------------------------------------------------------------
def cmd_enroll(args, config):
    db = Database(config.db_file)
    detector = FaceDetector(config)
    trainer = FaceTrainer(config, db, detector)

    if args.source == "webcam":
        n = trainer.capture_from_webcam(args.id, args.name, args.samples, args.camera_index)
    else:
        n = trainer.capture_from_images(args.id, args.name, args.path, assume_cropped=args.cropped)

    print(f"[OK] Enrolled '{args.name}' ({args.id}) with {n} face sample(s).")
    print("     Run `python main.py train` next to (re)build the recognition model.")


def cmd_train(args, config):
    db = Database(config.db_file)
    detector = FaceDetector(config)
    trainer = FaceTrainer(config, db, detector)
    n_students = trainer.train_model()
    print(f"[OK] Model trained on {n_students} student(s). Saved to {config.model_file}")


def cmd_recognize(args, config):
    db = Database(config.db_file)
    detector = FaceDetector(config)
    recognizer = FaceRecognizer(config, detector)
    manager = AttendanceManager(config, db, detector, recognizer)

    if args.source == "image":
        summary = manager.run_on_image(args.path, save_output=not args.no_save,
                                        assume_cropped=args.cropped)
    elif args.source == "video":
        summary = manager.run_on_video(args.path, save_output=not args.no_save,
                                        frame_skip=args.frame_skip)
    else:  # webcam
        summary = manager.run_on_webcam(args.camera_index, args.duration, display=args.display)

    print(f"[OK] Processed {summary.frames_processed} frame(s), "
          f"saw {summary.faces_seen} face detection(s).")
    if summary.unique_students_marked:
        print(f"     Attendance marked for: {', '.join(summary.unique_students_marked)}")
    else:
        print("     No new attendance was marked (no recognized faces, or already marked today).")


def cmd_report(args, config):
    db = Database(config.db_file)
    generator = ReportGenerator(config, db)

    if args.type == "daily":
        summary, csv_path = generator.daily_report(args.date)
        print(f"[OK] Daily report for {summary['date']}: {summary['total_present']} present")
        for line in summary["students"]:
            print(f"     - {line}")
        print(f"     CSV saved to {csv_path}")
    else:
        start, end = (args.start, args.end) if args.start and args.end else \
            ReportGenerator.last_n_days_range(args.days)
        summary, csv_path, chart_path = generator.range_report(start, end)
        print(f"[OK] Range report {summary['start']} -> {summary['end']}: "
              f"{summary['total_records']} record(s)")
        for name, count in summary["attendance_by_student"].items():
            print(f"     - {name}: {count} day(s) present")
        print(f"     CSV saved to {csv_path}")
        if chart_path:
            print(f"     Chart saved to {chart_path}")


def cmd_list_students(args, config):
    db = Database(config.db_file)
    students = db.list_students()
    if not students:
        print("No students enrolled yet.")
        return
    print(f"{'Student ID':<15}{'Name':<25}{'Registered On'}")
    print("-" * 60)
    for s in students:
        print(f"{s.student_id:<15}{s.name:<25}{s.registered_on}")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Face Recognition Attendance System (OpenCV + LBPH) - CLI",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (optional)")
    sub = parser.add_subparsers(dest="command", required=True)

    # enroll -------------------------------------------------------------
    p_enroll = sub.add_parser("enroll", help="Register a new student / person")
    p_enroll.add_argument("--id", required=True, help="Unique student ID, e.g. VIT2026001")
    p_enroll.add_argument("--name", required=True, help="Full name")
    p_enroll.add_argument("--source", choices=["webcam", "images"], default="images",
                           help="Capture face samples live from a webcam, or from an "
                                "existing folder of photos (default: images)")
    p_enroll.add_argument("--path", help="Folder of photos (required when --source images)")
    p_enroll.add_argument("--cropped", action="store_true",
                           help="Treat images in --path as already face-cropped "
                                "(skips Haar detection; useful for pre-cropped datasets)")
    p_enroll.add_argument("--samples", type=int, default=None,
                           help="Number of samples to capture from webcam")
    p_enroll.add_argument("--camera-index", type=int, default=0)
    p_enroll.set_defaults(func=cmd_enroll)

    # train ---------------------------------------------------------------
    p_train = sub.add_parser("train", help="Train the LBPH model on all enrolled students")
    p_train.set_defaults(func=cmd_train)

    # recognize -------------------------------------------------------------
    p_rec = sub.add_parser("recognize", help="Run face recognition + mark attendance")
    p_rec.add_argument("--source", choices=["webcam", "video", "image"], default="image",
                        help="Input source (default: image)")
    p_rec.add_argument("--path", help="Path to video/image file (required for those sources)")
    p_rec.add_argument("--cropped", action="store_true",
                        help="Treat --source image as an already face-cropped image "
                             "(skips Haar detection)")
    p_rec.add_argument("--camera-index", type=int, default=0)
    p_rec.add_argument("--duration", type=int, default=15, help="Webcam session length in seconds")
    p_rec.add_argument("--frame-skip", type=int, default=2, help="Process every Nth video frame")
    p_rec.add_argument("--no-save", action="store_true", help="Do not save annotated output")
    p_rec.add_argument("--display", action="store_true",
                        help="Show a live preview window (requires a GUI-capable environment)")
    p_rec.set_defaults(func=cmd_recognize)

    # report -------------------------------------------------------------
    p_report = sub.add_parser("report", help="Generate attendance reports/analytics")
    p_report.add_argument("--type", choices=["daily", "range"], default="daily")
    p_report.add_argument("--date", help="YYYY-MM-DD (daily report, default: today)")
    p_report.add_argument("--start", help="YYYY-MM-DD (range report)")
    p_report.add_argument("--end", help="YYYY-MM-DD (range report)")
    p_report.add_argument("--days", type=int, default=7,
                           help="If --start/--end omitted, report on the last N days")
    p_report.set_defaults(func=cmd_report)

    # list-students -------------------------------------------------------
    p_list = sub.add_parser("list-students", help="List all enrolled students")
    p_list.set_defaults(func=cmd_list_students)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config) if args.config else load_config()
        configure_logging(config.log_file, config.log_level)
    except AttendanceSystemError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        args.func(args, config)
        return 0
    except AttendanceSystemError as exc:
        logger.error("%s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level safety net for a CLI tool
        logger.exception("Unexpected error")
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
