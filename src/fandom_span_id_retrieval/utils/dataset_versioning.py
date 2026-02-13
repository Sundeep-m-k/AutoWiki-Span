from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_manifest(domain: str, processed_dir: Path) -> Path:
    files = [
        processed_dir / f"pages_{domain}.jsonl",
        processed_dir / f"paragraphs_{domain}.jsonl",
        processed_dir / f"sentences_{domain}.jsonl",
        processed_dir / f"paragraph_links_{domain}.csv",
        processed_dir / f"sentence_links_{domain}.csv",
        processed_dir / f"articles_{domain}.jsonl",
    ]

    manifest: Dict[str, object] = {
        "domain": domain,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": [],
    }

    for path in files:
        if not path.exists():
            continue
        stat = path.stat()
        manifest["files"].append({
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": _sha256(path),
        })

    out_path = processed_dir / "dataset_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_path
