from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def write_model_manifest(model_dir: Path, metadata: Dict[str, object]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), **metadata}
    path = model_dir / "model_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path
