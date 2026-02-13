#!/usr/bin/env python3

from pathlib import Path

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.linking_pipeline.pipeline import run_linking_pipeline


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "linking_pipeline" / "experiment.yaml"
    outputs = run_linking_pipeline(config_path)
    for path in outputs:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
