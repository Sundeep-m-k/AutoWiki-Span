#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.utils.logging_utils import create_logger
from fandom_span_id_retrieval.utils.experiment_registry import (
    build_run_entry,
    collect_outputs,
    create_run_id,
    init_run_dir,
    snapshot_configs,
    write_registry_entry,
    write_run_metadata,
)


def _run(cmd: list[str], logger) -> None:
    logger.info(f"=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Failed: {' '.join(cmd)} (returncode={result.returncode})")
        sys.exit(result.returncode)
    logger.info(f"Finished: {' '.join(cmd)}")


def _load_step_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"completed": []}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_step_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Reuse an existing run id under outputs/runs to resume a run",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Start at a specific step: scrape, ground_truth, span_id, retrieval, linking, report",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        help="Stop after a specific step name",
    )
    # Resume examples:
    # python scripts/run_research_pipeline.py --run-id research_YYYYMMDD_HHMMSS
    # python scripts/run_research_pipeline.py --start-at retrieval
    # python scripts/run_research_pipeline.py --start-at span_id --stop-after span_id
    args = parser.parse_args()

    run_id = args.run_id or create_run_id("research")
    run_info = init_run_dir(PROJECT_ROOT, run_id)

    log_dir = PROJECT_ROOT / "outputs" / "logs" / "research_pipeline"
    logger, _ = create_logger(log_dir, script_name="research_pipeline")

    config_paths = [
        PROJECT_ROOT / "configs" / "scraping.yaml",
        PROJECT_ROOT / "configs" / "span_id" / "experiment.yaml",
        PROJECT_ROOT / "configs" / "retrieval" / "both_levels.yaml",
        PROJECT_ROOT / "configs" / "retrieval" / "experiment.yaml",
        PROJECT_ROOT / "configs" / "retrieval" / "paragraph_faiss.yaml",
        PROJECT_ROOT / "configs" / "rerank" / "experiment.yaml",
        PROJECT_ROOT / "configs" / "linking_pipeline" / "experiment.yaml",
    ]

    snapshot_configs(run_info.run_dir, config_paths)
    entry = build_run_entry(
        repo_root=PROJECT_ROOT,
        run_info=run_info,
        description="Full research pipeline run",
        config_paths=config_paths,
    )

    write_registry_entry(PROJECT_ROOT, entry)

    scripts_dir = PROJECT_ROOT / "scripts"
    python = sys.executable

    steps = [
        ("scrape", [python, str(scripts_dir / "01_Data_processing" / "00_scrape_fandom.py")]),
        ("ground_truth", [python, str(scripts_dir / "01_Data_processing" / "01_build_ground_truth.py")]),
        (
            "span_id",
            [
                python,
                str(scripts_dir / "02_Span_Identification" / "01_run_span_id_grid.py"),
                "--parallel",
                "--num-workers", "2",
                "--progressive",
            ],
        ),
        (
            "retrieval",
            [
                python,
                str(scripts_dir / "03_Article_Retrieval" / "run_variant_pipeline.py"),
                "--parallel",
                "--num-workers", "2",
                "--progressive",
            ],
        ),
        ("linking", [python, str(scripts_dir / "04_Linking_Pipeline" / "run_linking_pipeline.py")]),
        ("report", [python, str(scripts_dir / "05_reports" / "build_research_report.py")]),
    ]

    state_path = run_info.run_dir / "step_state.json"
    state = _load_step_state(state_path)
    completed = set(state.get("completed", []))

    if args.start_at:
        completed = set()

    start_found = args.start_at is None
    for name, cmd in steps:
        if not start_found:
            if name == args.start_at:
                start_found = True
            else:
                continue

        if name in completed:
            logger.info(f"Skipping completed step: {name}")
        else:
            _run(cmd, logger)
            completed.add(name)
            _write_step_state(state_path, {"completed": sorted(completed)})

        if args.stop_after and name == args.stop_after:
            logger.info(f"Stopping after step: {name}")
            break

    outputs = collect_outputs(PROJECT_ROOT)
    write_run_metadata(run_info.run_dir, {"outputs": outputs})
    logger.info(f"Run complete. Run dir: {run_info.run_dir}")


if __name__ == "__main__":
    main()
