# src/fandom_span_id_retrieval/utils/logging_utils.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union


def _resolve_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    level_str = str(level).upper()
    return getattr(logging, level_str, logging.DEBUG)


def create_logger(
    log_dir: Path,
    script_name: str,
    level: Union[int, str] = "DEBUG",
    to_console: bool = True,
    to_file: bool = True,
) -> Tuple[logging.Logger, Path]:
    """
    Create a logger that logs to both console and a timestamped file.

    log_dir: directory where log file will be created
    script_name: short name of the script, e.g. "10_train_span_id"
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{script_name}.log"

    logger = logging.getLogger(script_name)
    resolved_level = _resolve_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False  # avoid duplicate logs if root logger is configured

    # Clear existing handlers for this logger (in case of re-use in notebooks)
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    if to_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(resolved_level)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    # Console handler
    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(resolved_level)
        console_fmt = logging.Formatter(
            fmt="%(asctime)s — %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        ch.setFormatter(console_fmt)
        logger.addHandler(ch)

    logger.info(f"Logger initialized for {script_name}")
    if to_file:
        logger.info(f"Log file: {log_file}")

    return logger, log_file
