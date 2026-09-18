"""Configuration loading utilities.

Centralizes access to config.yaml so the rest of the codebase never
hard-codes paths or tunable parameters. Also guarantees that every
directory the application needs actually exists (created on first run),
which is part of the project's error-handling / reliability strategy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

from src.exceptions import ConfigError

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


@dataclass
class AppConfig:
    """Typed, convenient wrapper around the raw YAML configuration."""

    raw: Dict[str, Any] = field(default_factory=dict)
    base_dir: str = field(default_factory=lambda: os.path.dirname(DEFAULT_CONFIG_PATH))

    # ---- path helpers -------------------------------------------------
    def _abs(self, relative: str) -> str:
        return os.path.join(self.base_dir, relative)

    @property
    def dataset_dir(self) -> str:
        return self._abs(self.raw["paths"]["dataset_dir"])

    @property
    def model_dir(self) -> str:
        return self._abs(self.raw["paths"]["model_dir"])

    @property
    def model_file(self) -> str:
        return self._abs(self.raw["paths"]["model_file"])

    @property
    def labels_file(self) -> str:
        return self._abs(self.raw["paths"]["labels_file"])

    @property
    def db_file(self) -> str:
        return self._abs(self.raw["paths"]["db_file"])

    @property
    def reports_dir(self) -> str:
        return self._abs(self.raw["paths"]["reports_dir"])

    @property
    def log_file(self) -> str:
        return self._abs(self.raw["paths"]["log_file"])

    @property
    def output_dir(self) -> str:
        return self._abs(self.raw["paths"]["output_dir"])

    # ---- detection / recognition tunables ------------------------------
    @property
    def scale_factor(self) -> float:
        return float(self.raw["detection"]["scale_factor"])

    @property
    def min_neighbors(self) -> int:
        return int(self.raw["detection"]["min_neighbors"])

    @property
    def min_size(self):
        return tuple(self.raw["detection"]["min_size"])

    @property
    def face_size(self):
        return tuple(self.raw["capture"]["face_size"])

    @property
    def samples_per_person(self) -> int:
        return int(self.raw["capture"]["samples_per_person"])

    @property
    def confidence_threshold(self) -> float:
        return float(self.raw["recognition"]["confidence_threshold"])

    @property
    def mark_once_per_day(self) -> bool:
        return bool(self.raw["attendance"]["mark_once_per_day"])

    @property
    def log_level(self) -> str:
        return str(self.raw["logging"]["level"])

    def ensure_directories(self) -> None:
        """Create every directory the app relies on (idempotent)."""
        for path in (
            self.dataset_dir,
            self.model_dir,
            self.reports_dir,
            self.output_dir,
            os.path.dirname(self.log_file),
            os.path.dirname(self.db_file),
        ):
            os.makedirs(path, exist_ok=True)


def load_config(path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate config.yaml, returning an AppConfig instance."""
    if not os.path.isfile(path):
        raise ConfigError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    required_top_level = ("paths", "detection", "capture", "recognition", "attendance", "logging")
    missing = [key for key in required_top_level if key not in raw]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")

    cfg = AppConfig(raw=raw, base_dir=os.path.dirname(os.path.abspath(path)))
    cfg.ensure_directories()
    return cfg
