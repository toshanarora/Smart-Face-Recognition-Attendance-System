"""Orchestration layer: ties detection + recognition + the database
together into the three supported recognition workflows: webcam,
video-file and single-image. This is the module the CLI's ``recognize``
command delegates to.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2

from src.config import AppConfig
from src.database import Database
from src.exceptions import CameraError, InvalidImageError
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer, Recognition
from src.logger import get_logger

logger = get_logger(__name__)

BOX_COLOR_KNOWN = (0, 200, 0)
BOX_COLOR_UNKNOWN = (0, 0, 220)


@dataclass
class SessionSummary:
    frames_processed: int
    faces_seen: int
    unique_students_marked: List[str]


class AttendanceManager:
    def __init__(self, config: AppConfig, db: Database, detector: FaceDetector,
                 recognizer: FaceRecognizer):
        self.config = config
        self.db = db
        self.detector = detector
        self.recognizer = recognizer

    # ------------------------------------------------------------------
    def _annotate(self, frame, recognitions: List[Recognition]):
        for r in recognitions:
            x, y, w, h = r.box
            color = BOX_COLOR_KNOWN if r.is_known else BOX_COLOR_UNKNOWN
            label = f"{r.name} ({r.confidence:.1f})" if r.is_known else "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        return frame

    def _process_frame(self, frame, marked: set, assume_cropped: bool = False) -> List[Recognition]:
        recognitions = (
            self.recognizer.recognize_cropped_image(frame)
            if assume_cropped
            else self.recognizer.recognize_frame(frame)
        )
        for r in recognitions:
            if r.is_known and r.student_id not in marked:
                inserted = self.db.mark_attendance(
                    r.student_id, r.confidence, once_per_day=self.config.mark_once_per_day
                )
                if inserted:
                    marked.add(r.student_id)
                    logger.info("Attendance marked for %s (%s)", r.student_id, r.name)
        return recognitions

    # ------------------------------------------------------------------
    def run_on_image(self, image_path: str, save_output: bool = True,
                      assume_cropped: bool = False) -> SessionSummary:
        frame = cv2.imread(image_path)
        if frame is None:
            raise InvalidImageError(f"Could not read image: {image_path}")

        marked: set = set()
        recognitions = self._process_frame(frame, marked, assume_cropped=assume_cropped)
        frame = self._annotate(frame, recognitions)

        if save_output:
            out_path = os.path.join(self.config.output_dir, "annotated_" + os.path.basename(image_path))
            cv2.imwrite(out_path, frame)
            logger.info("Annotated image saved to %s", out_path)

        return SessionSummary(1, len(recognitions), list(marked))

    def run_on_video(self, video_path: str, save_output: bool = True,
                      frame_skip: int = 2) -> SessionSummary:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise CameraError(f"Could not open video file: {video_path}")

        writer = None
        marked: set = set()
        frames_processed, faces_seen, frame_idx = 0, 0, 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % max(frame_skip, 1) != 0:
                    continue

                recognitions = self._process_frame(frame, marked)
                faces_seen += len(recognitions)
                frames_processed += 1
                frame = self._annotate(frame, recognitions)

                if save_output:
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_path = os.path.join(self.config.output_dir, "annotated_output.mp4")
                        writer = cv2.VideoWriter(out_path, fourcc, 15.0, (w, h))
                        logger.info("Writing annotated video to %s", out_path)
                    writer.write(frame)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        return SessionSummary(frames_processed, faces_seen, list(marked))

    def run_on_webcam(self, camera_index: int = 0, max_seconds: int = 30,
                       display: bool = False) -> SessionSummary:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera index {camera_index}. "
                "Use `recognize --source image` or `--source video` in headless environments."
            )

        marked: set = set()
        frames_processed, faces_seen = 0, 0
        start = time.time()

        try:
            while time.time() - start < max_seconds:
                ok, frame = cap.read()
                if not ok:
                    raise CameraError("Failed to read a frame from the camera.")

                recognitions = self._process_frame(frame, marked)
                faces_seen += len(recognitions)
                frames_processed += 1
                frame = self._annotate(frame, recognitions)

                if display:
                    cv2.imshow("Attendance - press q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        return SessionSummary(frames_processed, faces_seen, list(marked))
