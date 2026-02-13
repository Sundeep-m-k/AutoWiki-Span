from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class RunInfo:
    run_id: str
    run_dir: Path
    created_at: str


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_git_info(repo_root: Path) -> Dict[str, str]:
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return {}

    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        ref_path = git_dir / ref
        commit = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else ""
        branch = ref.split("/")[-1]
        return {"git_branch": branch, "git_commit": commit}

    return {"git_commit": head}


def create_run_id(prefix: str = "run") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def init_run_dir(repo_root: Path, run_id: str) -> RunInfo:
    run_dir = repo_root / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunInfo(run_id=run_id, run_dir=run_dir, created_at=_utc_now())


def snapshot_configs(run_dir: Path, config_paths: Iterable[Path]) -> List[Path]:
    out_dir = run_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for path in config_paths:
        if not path.exists():
            continue
        rel = path.name
        dest = out_dir / rel
        shutil.copy2(path, dest)
        copied.append(dest)

    return copied


def write_run_metadata(run_dir: Path, metadata: Dict[str, object]) -> Path:
    meta_path = run_dir / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


def write_registry_entry(repo_root: Path, entry: Dict[str, object]) -> Path:
    registry = repo_root / "outputs" / "runs" / "registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return registry


def write_task_metadata(output_dir: Path, metadata: Dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "task_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def collect_outputs(root: Path, patterns: Optional[Iterable[str]] = None) -> List[Dict[str, object]]:
    if patterns is None:
        patterns = ["outputs/**", "stats/**"]

    files: List[Dict[str, object]] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append({
                "path": str(path.relative_to(root)),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return files


def build_run_entry(
    repo_root: Path,
    run_info: RunInfo,
    description: str,
    config_paths: Iterable[Path],
) -> Dict[str, object]:
    entry: Dict[str, object] = {
        "run_id": run_info.run_id,
        "created_at": run_info.created_at,
        "description": description,
        "configs": [str(p) for p in config_paths if p.exists()],
    }
    entry.update(_read_git_info(repo_root))
    return entry
