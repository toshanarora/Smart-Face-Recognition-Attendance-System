"""Face detection using OpenCV's Haar Cascade classifier.

Haar cascades are used (instead of a heavier DNN) deliberately: they ship
with OpenCV (no extra model download required), run fast on CPU, and are
sufficient for a frontal-face attendance kiosk use case - a reasonable
engineering trade-off that keeps the project fully offline-runnable.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.config import AppConfig
from src.logger import get_logger

logger = get_logger(__name__)

BoundingBox = Tuple[int, int, int, int]  # x, y, w, h


class FaceDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        logger.debug("Loaded Haar cascade from %s", cascade_path)

    def detect_faces(self, frame: np.ndarray) -> List[BoundingBox]:
        """Return a list of (x, y, w, h) bounding boxes for faces in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size,
        )
        return [tuple(map(int, box)) for box in boxes]

    def crop_and_normalize(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray:
        """Crop the face region, convert to grayscale, equalize, and resize
        to the configured face_size so every sample fed to the model is
        geometrically consistent."""
        x, y, w, h = box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = gray[y:y + h, x:x + w]
        face = cv2.equalizeHist(face)
        face = cv2.resize(face, self.config.face_size, interpolation=cv2.INTER_LINEAR)
        return face

    def largest_face(self, boxes: List[BoundingBox]):
        """Convenience helper: pick the largest detected face (closest to
        camera) - used when enrolling one person at a time."""
        if not boxes:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])
