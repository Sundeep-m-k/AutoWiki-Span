#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
import sys

from fandom_span_id_retrieval.utils.logging_utils import create_logger


def run(cmd: list[str], logger):
    logger.info(f"=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Failed: {' '.join(cmd)} (returncode={result.returncode})")
        sys.exit(result.returncode)
    logger.info(f"Finished: {' '.join(cmd)}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"

    log_dir = repo_root / "outputs" / "logs" / "data_pipeline"
    logger, _ = create_logger(log_dir, script_name="data_pipeline")

    # -----------------------------
    # 1️⃣ DATA PROCESSING
    # -----------------------------
    data_dir = scripts_dir / "01_Data_processing"

    scrape_script = data_dir / "00_scrape_fandom.py"
    gt_script = data_dir / "01_build_ground_truth.py"

    run(["python", str(scrape_script)], logger)
    run(["python", str(gt_script)], logger)

    # -----------------------------
    # 2️⃣ SPAN IDENTIFICATION
    # -----------------------------
    span_dir = scripts_dir / "02_Span_Identification"
    span_script = span_dir / "01_run_span_id_grid.py"

    run(["python", str(span_script), "--parallel", "--num-workers", "2", "--progressive"], logger)

    # -----------------------------
    # 3️⃣ ARTICLE RETRIEVAL
    # -----------------------------
    retrieval_dir = scripts_dir / "03_Article_Retrieval"

    variant_script = retrieval_dir / "run_variant_pipeline.py"
    run(["python", str(variant_script), "--parallel", "--num-workers", "2", "--progressive"], logger)

    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
