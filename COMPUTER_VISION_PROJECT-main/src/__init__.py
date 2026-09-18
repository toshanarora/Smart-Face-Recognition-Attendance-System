"""
Face Recognition Attendance System
-----------------------------------
A modular, CLI-driven computer vision project built with OpenCV.

Modules:
    config              -> configuration loading
    logger              -> application-wide logging setup
    exceptions          -> custom exception hierarchy
    database            -> SQLite persistence layer
    face_detector       -> Haar-cascade based face detection
    face_trainer        -> dataset building + LBPH model training
    face_recognizer     -> LBPH-based face recognition
    attendance_manager  -> orchestrates detection + recognition + logging
    report_generator    -> analytics / CSV / chart generation
"""

__version__ = "1.0.0"
