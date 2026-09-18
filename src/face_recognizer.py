"""LBPH-based face recognition (inference side).

Loads the model produced by ``FaceTrainer.train_model`` and predicts a
label + confidence (distance - lower means more similar) for each
detected face. Predictions above the configured confidence threshold are
reported as "Unknown" rather than forced onto the nearest known label,
which is the project's core false-positive mitigation strategy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from src.config import AppConfig
from src.exceptions import ModelNotTrainedError
from src.face_detector import BoundingBox, FaceDetector
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Recognition:
    box: BoundingBox
    student_id: Optional[str]
    name: str
    confidence: float
    is_known: bool


class FaceRecognizer:
    def __init__(self, config: AppConfig, detector: FaceDetector):
        self.config = config
        self.detector = detector
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.isfile(self.config.model_file) or not os.path.isfile(self.config.labels_file):
            raise ModelNotTrainedError(
                "No trained model found. Run `python main.py train` after enrolling students."
            )
        self.recognizer.read(self.config.model_file)
        with open(self.config.labels_file, "r", encoding="utf-8") as fh:
            self.label_map = json.load(fh)
        logger.debug("Loaded model with %d known labels", len(self.label_map))

    def _predict_face(self, face: np.ndarray, box: BoundingBox) -> Recognition:
        label_id, confidence = self.recognizer.predict(face)
        is_known = confidence <= self.config.confidence_threshold
        entry = self.label_map.get(str(label_id))
        if is_known and entry is not None:
            return Recognition(box, entry["student_id"], entry["name"], confidence, True)
        return Recognition(box, None, "Unknown", confidence, False)

    def recognize_cropped_image(self, frame: np.ndarray) -> List[Recognition]:
        """Recognize a single image that is already a tightly-cropped face
        (skips Haar detection entirely) - mirrors FaceTrainer's
        ``assume_cropped`` enrollment mode for pre-cropped datasets."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.equalizeHist(gray)
        face = cv2.resize(gray, self.config.face_size, interpolation=cv2.INTER_LINEAR)
        h, w = frame.shape[:2]
        return [self._predict_face(face, (0, 0, w, h))]

    def recognize_frame(self, frame: np.ndarray) -> List[Recognition]:
        results = []
        for box in self.detector.detect_faces(frame):
            face = self.detector.crop_and_normalize(frame, box)
            results.append(self._predict_face(face, box))
        return results
