# src/fandom_span_id_retrieval/utils/version_utils.py
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /.../Fandom_Span_ID_retrieval


def get_version_file() -> Optional[str]:
    version_file = PROJECT_ROOT / "VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip()
    return None


def get_git_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_version_info() -> Dict[str, Optional[str]]:
    """
    Return a small dict you can embed into metrics.json / stats.json entries.
    """
    return {
        "version": get_version_file(),
        "git_hash": get_git_hash(),
        "timestamp": get_timestamp(),
    }
