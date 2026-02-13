# src/fandom_span_id_retrieval/utils/stats_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT, ExperimentConfig
from .version_utils import get_version_info


STATS_DIR = PROJECT_ROOT / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


def _stats_path(domain: str) -> Path:
    """
    e.g. stats/money-heist.json
    """
    safe_domain = domain.replace(" ", "-")
    return STATS_DIR / f"{safe_domain}.json"


def load_stats(domain: str) -> Dict[str, Any]:
    """
    Load per-domain stats JSON, or initialize a skeleton if missing.
    """
    path = _stats_path(domain)
    if not path.is_file():
        return {
            "domain": domain,
            "scraping_stats": {},          # filled by scraping stage
            "dataset_stats": {},           # filled by ground_truth builder
            "span_identification_metrics": {},
            "retrieval_metrics": {},
            "experiments": [],             # list of experiment entries
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_stats(domain: str, stats: Dict[str, Any]) -> None:
    path = _stats_path(domain)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def update_scraping_stats(domain: str, scraping_stats: Dict[str, Any]) -> None:
    """
    Set / overwrite the 'scraping_stats' block for this domain.
    """
    stats = load_stats(domain)
    stats["scraping_stats"] = scraping_stats
    save_stats(domain, stats)


def upsert_experiment_entry(
    domain: str,
    exp_cfg: ExperimentConfig,
    metrics: Dict[str, Any],
    task_type: Optional[str] = None,
) -> None:
    """
    Add or update a single experiment entry in stats["experiments"].

    task_type: optional coarse type, e.g. "span-id", "retrieval", "rerank", "pipeline".
               If None, uses exp_cfg.task.
    """
    stats = load_stats(domain)
    experiments: List[Dict[str, Any]] = stats.get("experiments", [])

    key = {
        "task": task_type or exp_cfg.task,
        "model": exp_cfg.model,
        "variant": exp_cfg.variant,
        "seed": exp_cfg.seed,
    }

    # merge metrics with version info (version, git_hash, timestamp)
    entry_metrics = {**metrics, **get_version_info()}

    existing_idx = None
    for i, e in enumerate(experiments):
        if (
            e.get("task") == key["task"]
            and e.get("model") == key["model"]
            and e.get("variant") == key["variant"]
            and e.get("seed") == key["seed"]
        ):
            existing_idx = i
            break

    new_entry = {
        **key,
        "output_dir": str(exp_cfg.outputs_dir),
        "metrics": entry_metrics,
    }

    if existing_idx is None:
        experiments.append(new_entry)
    else:
        experiments[existing_idx] = new_entry

    stats["experiments"] = experiments
    save_stats(domain, stats)
