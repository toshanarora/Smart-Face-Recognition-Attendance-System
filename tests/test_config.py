import os

from src.config import load_config


def test_load_config_creates_directories(tmp_path, monkeypatch):
    cfg = load_config()  # loads the real project config.yaml
    assert os.path.isdir(cfg.dataset_dir)
    assert os.path.isdir(cfg.model_dir)
    assert os.path.isdir(cfg.reports_dir)
    assert cfg.confidence_threshold > 0
    assert cfg.samples_per_person > 0


def test_face_size_and_min_size_are_tuples():
    cfg = load_config()
    assert isinstance(cfg.face_size, tuple)
    assert isinstance(cfg.min_size, tuple)
    assert len(cfg.face_size) == 2
