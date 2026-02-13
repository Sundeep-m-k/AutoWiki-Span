#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.utils.experiment_registry import (
    build_run_entry,
    collect_outputs,
    create_run_id,
    init_run_dir,
    snapshot_configs,
    write_registry_entry,
    write_run_metadata,
)


def _run(cmd: list[str]) -> None:
    print(f"\n=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Failed: {' '.join(cmd)} (returncode={result.returncode})")
        sys.exit(result.returncode)
    print(f"✅ Finished: {' '.join(cmd)}")


def main() -> None:
    run_id = create_run_id("research")
    run_info = init_run_dir(PROJECT_ROOT, run_id)

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

    _run([python, str(scripts_dir / "01_Data_processing" / "00_scrape_fandom.py")])
    _run([python, str(scripts_dir / "01_Data_processing" / "01_build_ground_truth.py")])
    _run([python, str(scripts_dir / "02_Span_Identification" / "01_run_span_id_grid.py")])
    _run([python, str(scripts_dir / "03_Article_Retrieval" / "run_both_levels.py")])
    _run([python, str(scripts_dir / "03_Article_Retrieval" / "run_paragraph_retrieval.py")])
    _run([python, str(scripts_dir / "04_Linking_Pipeline" / "run_linking_pipeline.py")])
    _run([python, str(scripts_dir / "05_reports" / "build_research_report.py")])

    outputs = collect_outputs(PROJECT_ROOT)
    write_run_metadata(run_info.run_dir, {"outputs": outputs})
    print(f"\nRun complete. Run dir: {run_info.run_dir}")


if __name__ == "__main__":
    main()
