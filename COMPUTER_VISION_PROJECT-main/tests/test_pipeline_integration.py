"""End-to-end pipeline test: enroll (from synthetic image folders) ->
train -> recognize.

Note on methodology: the bundled Haar cascade is trained on real human
faces, so it will not reliably fire on the simple procedurally-drawn
"cartoon" faces used for offline testing (see
scripts/generate_synthetic_faces.py and docs/testing_approach.md for the
full rationale). To keep this test deterministic and independent of
Haar's sensitivity to non-photographic input, we monkeypatch
FaceDetector.detect_faces to report "the whole image is the face",
which lets us validate the parts that matter here in isolation: dataset
building, LBPH training, model persistence, and prediction/labeling
logic. Detector behaviour itself is covered separately in
test_face_detector.py.
"""
from __future__ import annotations

from src.database import Database
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_trainer import FaceTrainer


def _stub_whole_frame_as_face(monkeypatch):
    def fake_detect_faces(self, frame):
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]

    monkeypatch.setattr(FaceDetector, "detect_faces", fake_detect_faces)


def test_enroll_train_recognize_pipeline(tmp_config, tmp_db, synthetic_faces_dir, monkeypatch):
    _stub_whole_frame_as_face(monkeypatch)

    detector = FaceDetector(tmp_config)
    trainer = FaceTrainer(tmp_config, tmp_db, detector)

    n1 = trainer.capture_from_images("P0", "Person Zero", f"{synthetic_faces_dir}/person_0")
    n2 = trainer.capture_from_images("P1", "Person One", f"{synthetic_faces_dir}/person_1")
    assert n1 == 6
    assert n2 == 6

    n_students = trainer.train_model()
    assert n_students == 2

    recognizer = FaceRecognizer(tmp_config, detector)
    assert set(recognizer.label_map.keys()) == {"0", "1"}

    students = tmp_db.list_students()
    assert {s.student_id for s in students} == {"P0", "P1"}


def test_training_fails_with_no_students(tmp_config, tmp_db):
    from src.exceptions import InsufficientSamplesError
    import pytest

    detector = FaceDetector(tmp_config)
    trainer = FaceTrainer(tmp_config, tmp_db, detector)
    with pytest.raises(InsufficientSamplesError):
        trainer.train_model()


def test_capture_from_images_missing_folder_raises(tmp_config, tmp_db):
    from src.exceptions import InvalidImageError
    import pytest

    detector = FaceDetector(tmp_config)
    trainer = FaceTrainer(tmp_config, tmp_db, detector)
    with pytest.raises(InvalidImageError):
        trainer.capture_from_images("X", "Nobody", "/no/such/folder")
