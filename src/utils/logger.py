"""Stdlib logging configuration (console + optional file)."""
from __future__ import annotations

import logging

from . import paths

_CONFIGURED: set[str] = set()
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file=None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(_FORMAT, _DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_path = paths.resolve(log_file)
        paths.ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED.add(name)
    return logger
