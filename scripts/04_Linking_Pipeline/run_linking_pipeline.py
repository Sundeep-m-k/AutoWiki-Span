#!/usr/bin/env python3

from pathlib import Path
import sys
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fandom_span_id_retrieval.linking_pipeline.pipeline import run_linking_pipeline
from fandom_span_id_retrieval.linking_pipeline.eval import run_linking_evaluation
from fandom_span_id_retrieval.utils.seed_utils import set_seed
from fandom_span_id_retrieval.utils.experiment_registry import write_task_metadata
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _run_variant_linking(config_path: Path, logger) -> None:
    """Worker function for parallel variant processing"""
    try:
        outputs = run_linking_pipeline(config_path)
        for path in outputs:
            logger.info(f"Wrote: {path}")
    except Exception as e:
        logger.error(f"Error in linking pipeline: {e}")
        raise


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "linking_pipeline" / "experiment.yaml"

    log_dir = PROJECT_ROOT / "outputs" / "logs" / "linking_pipeline"
    logger, _ = create_logger(log_dir, script_name="linking_pipeline")
    
    if config_path.exists():
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        seed = int(cfg.get("linking_pipeline", {}).get("seed", 0))
        set_seed(seed, deterministic=True)
    
    # Run linking pipeline
    outputs = run_linking_pipeline(config_path)
    for path in outputs:
        logger.info(f"Wrote: {path}")
    
    # Run evaluation
    summary_path = run_linking_evaluation(config_path)
    logger.info(f"Wrote summary: {summary_path}")

    # Write metadata
    domain = ""
    if config_path.exists():
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        domain = str(cfg.get("linking_pipeline", {}).get("domain", ""))
    out_dir = PROJECT_ROOT / "outputs" / "linking_pipeline" / domain
    write_task_metadata(
        out_dir,
        {
            "task": "linking_pipeline",
            "domain": domain,
            "config": str(config_path),
            "summary": str(summary_path),
        },
    )
    
    logger.info("Linking pipeline completed successfully")
    logger.info(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
