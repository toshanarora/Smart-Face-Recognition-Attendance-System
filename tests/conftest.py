"""Shared pytest fixtures.

Tests run against a temporary, isolated copy of the project's directory
layout so they never touch real data in data/ and can be run repeatedly
/ in parallel / in CI without side effects.
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig  # noqa: E402
from src.database import Database  # noqa: E402


@pytest.fixture
def tmp_config(tmp_path) -> AppConfig:
    """A fully isolated AppConfig rooted inside pytest's tmp_path."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.yaml"), "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = AppConfig(raw=raw, base_dir=str(tmp_path))
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def tmp_db(tmp_config) -> Database:
    return Database(tmp_config.db_file)


@pytest.fixture
def synthetic_faces_dir(tmp_path):
    """Generate a small synthetic dataset (see scripts/generate_synthetic_faces.py)
    for pipeline tests that need real image files on disk."""
    import importlib.util

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_root, "scripts", "generate_synthetic_faces.py")
    spec = importlib.util.spec_from_file_location("generate_synthetic_faces", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    out_dir = str(tmp_path / "sample_faces")
    for person_idx in range(2):
        person_dir = os.path.join(out_dir, f"person_{person_idx}")
        os.makedirs(person_dir, exist_ok=True)
        for i in range(6):
            import cv2
            img = module.draw_face(seed=person_idx * 1000 + i)
            cv2.imwrite(os.path.join(person_dir, f"{i:03d}.png"), img)
    return out_dir
