import numpy as np

from src.face_detector import FaceDetector


def test_detect_faces_on_blank_frame_returns_empty(tmp_config):
    detector = FaceDetector(tmp_config)
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    boxes = detector.detect_faces(blank)
    assert boxes == []


def test_crop_and_normalize_output_shape(tmp_config):
    detector = FaceDetector(tmp_config)
    frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    box = (50, 50, 100, 100)
    face = detector.crop_and_normalize(frame, box)
    assert face.shape == tuple(reversed(tmp_config.face_size)) or face.shape == tmp_config.face_size


def test_largest_face_picks_biggest_box(tmp_config):
    detector = FaceDetector(tmp_config)
    boxes = [(0, 0, 40, 40), (0, 0, 100, 100), (0, 0, 60, 60)]
    assert detector.largest_face(boxes) == (0, 0, 100, 100)


def test_largest_face_empty_list(tmp_config):
    detector = FaceDetector(tmp_config)
    assert detector.largest_face([]) is None
