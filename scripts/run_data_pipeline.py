#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
import sys


def run(cmd: list[str]):
    print(f"\n=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Failed: {' '.join(cmd)} (returncode={result.returncode})")
        sys.exit(result.returncode)
    print(f"✅ Finished: {' '.join(cmd)}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"

    # -----------------------------
    # 1️⃣ DATA PROCESSING
    # -----------------------------
    data_dir = scripts_dir / "01_Data_processing"

    scrape_script = data_dir / "00_scrape_fandom.py"
    gt_script = data_dir / "01_build_ground_truth.py"

    run(["python", str(scrape_script)])
    run(["python", str(gt_script)])

    # -----------------------------
    # 2️⃣ SPAN IDENTIFICATION
    # -----------------------------
    span_dir = scripts_dir / "02_Span_Identification"
    span_script = span_dir / "01_run_span_id_grid.py"

    run(["python", str(span_script)])

    # -----------------------------
    # 3️⃣ ARTICLE RETRIEVAL
    # -----------------------------
    retrieval_dir = scripts_dir / "03_Article_Retrieval"

    both_levels_script = retrieval_dir / "run_both_levels.py"
    paragraph_script = retrieval_dir / "run_paragraph_retrieval.py"

    run(["python", str(both_levels_script)])
    run(["python", str(paragraph_script)])

    print("\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY 🎉")


if __name__ == "__main__":
    main()
