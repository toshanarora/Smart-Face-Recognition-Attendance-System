"""Application-wide logging configuration.

Every module gets a logger via get_logger(__name__). Logs go both to the
console (for interactive CLI usage) and to a rotating log file on disk
(for monitoring / audit / debugging after the fact) - this satisfies the
project's "logging & monitoring" non-functional requirement.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

_CONFIGURED = False


def configure_logging(log_file: str, level: str = "INFO", max_bytes: int = 1_048_576,
                       backup_count: int = 3) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
