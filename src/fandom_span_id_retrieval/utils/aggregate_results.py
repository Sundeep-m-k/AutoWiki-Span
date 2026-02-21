from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.utils.stats_utils import load_stats, save_stats


OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CSV_PATH = OUTPUTS_ROOT / "rerank" / "article_retrieval_results.csv"


def _parse_run_name(task: str, run_name: str) -> Tuple[str, str, Optional[int]]:
    prefix = f"{task}__"
    if run_name.startswith(prefix) and "__seed" in run_name:
        parts = run_name.split("__")
        if len(parts) >= 4 and parts[0] == task and parts[-1].startswith("seed"):
            model = parts[1]
            variant = "__".join(parts[2:-1])
            seed_str = parts[-1].replace("seed", "")
            if seed_str.isdigit():
                return model, variant, int(seed_str)
            return model, variant, None
    return "", run_name, None


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_test_metrics(task: str) -> List[Dict[str, Any]]:
    task_dir = OUTPUTS_ROOT / task
    if not task_dir.is_dir():
        return []

    entries: List[Dict[str, Any]] = []
    for metrics_path in task_dir.rglob("test_metrics.json"):
        try:
            rel = metrics_path.relative_to(task_dir)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue

        domain = rel.parts[0]
        run_name = metrics_path.parent.name
        model, variant, seed = _parse_run_name(task, run_name)
        metrics = _load_metrics(metrics_path)
        timestamp = datetime.fromtimestamp(metrics_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        entries.append(
            {
                "timestamp": timestamp,
                "task": task,
                "domain": domain,
                "model": model,
                "variant": variant,
            }
        )

    return entries


def _write_csv(entries: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = set()
    for entry in entries:
        metric_keys.update(entry.get("metrics", {}).keys())

    fields = [
        "timestamp",
        "task",
        "domain",
        "model",
        "variant",
        "seed",
        "output_dir",
    ] + sorted(metric_keys)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = {k: entry.get(k, "") for k in fields}
            for key, value in entry.get("metrics", {}).items():
                row[key] = value
            writer.writerow(row)


def _collect_test_metrics(task: str) -> List[Dict[str, Any]]:
    # Use data/processed/{domain} as the root for aggregation
    task_dir = PROJECT_ROOT / "data" / "processed"
    entries: List[Dict[str, Any]] = []
    for domain_dir in task_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        for metrics_path in domain_dir.rglob("test_metrics.json"):
            try:
                rel = metrics_path.relative_to(domain_dir)
            except ValueError:
                continue
            if len(rel.parts) < 1:
                continue
            domain = domain_dir.name
            run_name = metrics_path.parent.name
            model, variant, seed = _parse_run_name(task, run_name)
            metrics = _load_metrics(metrics_path)
            timestamp = datetime.fromtimestamp(metrics_path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            entries.append(
                {
                    "timestamp": timestamp,
                    "task": task,
                    "domain": domain,
                    "model": model,
                    "variant": variant,
                    "seed": seed,
                    "output_dir": str(metrics_path.parent),
                    "metrics": metrics,
                }
            )
    return entries
def _entry_for_stats(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": entry.get("model", ""),
        "variant": entry.get("variant", ""),
        "seed": entry.get("seed"),
        "output_dir": entry.get("output_dir", ""),
        "timestamp": entry.get("timestamp", ""),
        "metrics": entry.get("metrics", {}),
    }


def _update_stats(entries: List[Dict[str, Any]]) -> None:
    by_domain: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for entry in entries:
        domain = entry["domain"]
        task = entry["task"]
        by_domain.setdefault(domain, {}).setdefault(task, []).append(_entry_for_stats(entry))

    for domain, task_entries in by_domain.items():
        stats = load_stats(domain)
        stats["retrieval_metrics"] = {
            "article_retrieval": task_entries.get("retrieval", []),
            "rerank": task_entries.get("rerank", []),
        }
        save_stats(domain, stats)


def aggregate_results_main() -> None:
    retrieval_entries = _collect_test_metrics("retrieval")
    rerank_entries = _collect_test_metrics("rerank")

    all_entries = retrieval_entries + rerank_entries
    _write_csv(all_entries, CSV_PATH)
    _update_stats(all_entries)

    print(f"Wrote CSV: {CSV_PATH}")
    print(f"Updated stats for {len(set(e['domain'] for e in all_entries))} domains")


if __name__ == "__main__":
    aggregate_results_main()
