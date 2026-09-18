"""Dataset construction and LBPH model training.

Two enrollment sources are supported so the project is fully runnable
without a webcam (important for CLI-only evaluation environments):

  1. ``capture_from_webcam``  - live capture, samples the largest face
     across several frames.
  2. ``capture_from_images``  - build a student's sample set from a folder
     of already-existing photographs.

After samples exist on disk for at least one student, ``train_model``
builds an LBPH (Local Binary Patterns Histograms) face recognizer -
OpenCV-contrib's classic, lightweight, CPU-only face recognition
algorithm - and persists it to disk together with a JSON label map.
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import List

import cv2
import numpy as np

from src.config import AppConfig
from src.database import Database
from src.exceptions import (
    CameraError,
    InsufficientSamplesError,
    InvalidImageError,
    NoFaceDetectedError,
)
from src.face_detector import FaceDetector
from src.logger import get_logger

logger = get_logger(__name__)


class FaceTrainer:
    def __init__(self, config: AppConfig, db: Database, detector: FaceDetector):
        self.config = config
        self.db = db
        self.detector = detector

    # ------------------------------------------------------------------
    def _student_dir(self, student_id: str) -> str:
        path = os.path.join(self.config.dataset_dir, student_id)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    def capture_from_webcam(self, student_id: str, name: str, num_samples: int = None,
                             camera_index: int = 0) -> int:
        """Capture face samples live from a webcam. Returns number of
        samples saved."""
        num_samples = num_samples or self.config.samples_per_person
        self.db.add_student(student_id, name)
        out_dir = self._student_dir(student_id)

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera index {camera_index}. "
                "If no webcam is available, use --source images instead."
            )

        saved = 0
        try:
            while saved < num_samples:
                ok, frame = cap.read()
                if not ok:
                    raise CameraError("Failed to read a frame from the camera.")

                boxes = self.detector.detect_faces(frame)
                box = self.detector.largest_face(boxes)
                if box is not None:
                    face = self.detector.crop_and_normalize(frame, box)
                    fname = os.path.join(out_dir, f"{saved:03d}.png")
                    cv2.imwrite(fname, face)
                    saved += 1
                    logger.debug("Saved sample %d/%d for %s", saved, num_samples, student_id)
                time.sleep(0.05)
        finally:
            cap.release()

        if saved == 0:
            raise NoFaceDetectedError("No face was captured during the webcam session.")
        logger.info("Captured %d webcam samples for %s (%s)", saved, student_id, name)
        return saved

    # ------------------------------------------------------------------
    def capture_from_images(self, student_id: str, name: str, source_folder: str,
                             assume_cropped: bool = False) -> int:
        """Build a student's sample set from an existing folder of photos.

        By default every image is run through Haar-cascade face detection
        and the largest detected face is cropped out (use this for raw,
        un-cropped photographs, e.g. straight from a phone camera).

        Set ``assume_cropped=True`` when the source folder already
        contains face-only images (a common format for public benchmark
        datasets such as ORL/AT&T Faces) - in that case the whole image
        is used directly, skipping the detection step entirely.
        """
        if not os.path.isdir(source_folder):
            raise InvalidImageError(f"Source folder not found: {source_folder}")

        image_paths: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.pgm"):
            image_paths.extend(glob.glob(os.path.join(source_folder, ext)))
        if not image_paths:
            raise InvalidImageError(f"No image files found in {source_folder}")

        self.db.add_student(student_id, name)
        out_dir = self._student_dir(student_id)

        saved = 0
        for path in sorted(image_paths):
            frame = cv2.imread(path)
            if frame is None:
                logger.warning("Skipping unreadable image: %s", path)
                continue

            if assume_cropped:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                gray = cv2.equalizeHist(gray)
                face = cv2.resize(gray, self.config.face_size, interpolation=cv2.INTER_LINEAR)
            else:
                boxes = self.detector.detect_faces(frame)
                box = self.detector.largest_face(boxes)
                if box is None:
                    logger.warning("No face detected in %s - skipped", path)
                    continue
                face = self.detector.crop_and_normalize(frame, box)

            fname = os.path.join(out_dir, f"{saved:03d}.png")
            cv2.imwrite(fname, face)
            saved += 1

        if saved == 0:
            raise NoFaceDetectedError(
                f"No usable faces were found among the images in {source_folder}"
            )
        logger.info("Captured %d image samples for %s (%s)", saved, student_id, name)
        return saved

    # ------------------------------------------------------------------
    def train_model(self) -> int:
        """Train the LBPH recognizer on every sample currently in
        data/dataset/<student_id>/*.png. Returns number of students trained."""
        students = self.db.list_students()
        if not students:
            raise InsufficientSamplesError("No students enrolled yet - nothing to train.")

        faces, labels = [], []
        for student in students:
            student_dir = self._student_dir(student.student_id)
            samples = glob.glob(os.path.join(student_dir, "*.png"))
            for sample_path in samples:
                img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(student.label_id)

        if len(faces) < 2:
            raise InsufficientSamplesError(
                "Not enough training samples across all students (need at least 2)."
            )

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        os.makedirs(self.config.model_dir, exist_ok=True)
        recognizer.save(self.config.model_file)

        label_map = {str(s.label_id): {"student_id": s.student_id, "name": s.name} for s in students}
        with open(self.config.labels_file, "w", encoding="utf-8") as fh:
            json.dump(label_map, fh, indent=2)

        logger.info(
            "Trained LBPH model on %d samples from %d students -> %s",
            len(faces), len(students), self.config.model_file,
        )
        return len(students)
