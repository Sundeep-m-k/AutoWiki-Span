#!/usr/bin/env python3
"""
01_build_ground_truth.py

Builds paragraph-level and sentence-level ground truth for a Fandom domain:

- Reads domain from configs/scraping.yaml (via base_url)
- Reads HTML from data/raw/fandom_html/<domain>/
- Writes:
    data/processed/<domain>/pages_<domain>.jsonl
    data/processed/<domain>/paragraphs_<domain>.jsonl
    data/processed/<domain>/sentences_<domain>.jsonl
    data/processed/<domain>/paragraphs_<domain>_by_article/*.jsonl
    data/processed/<domain>/sentences_<domain>_by_article/*.jsonl
    data/processed/<domain>/paragraphs_<domain>.csv
    data/processed/<domain>/paragraph_links_<domain>.csv
    data/processed/<domain>/sentences_<domain>.csv
    data/processed/<domain>/sentence_links_<domain>.csv
- Updates stats/<domain>.json with dataset_stats
"""

import time
from urllib.parse import urlparse

import yaml

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.utils.logging_utils import create_logger
from fandom_span_id_retrieval.data.ground_truth import build_ground_truth_for_domain


def load_domain_from_scraping_config() -> str:
    cfg_path = PROJECT_ROOT / "configs" / "scraping.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if "base_url" not in cfg:
        raise ValueError("'base_url' must be defined in configs/scraping.yaml")

    base_url = cfg["base_url"].rstrip("/")
    netloc = urlparse(base_url).netloc
    domain = netloc.split(".")[0]  # e.g. money-heist.fandom.com -> money-heist
    return domain


def main() -> None:
    domain = load_domain_from_scraping_config()
    log_dir = PROJECT_ROOT / "data" / "logs" / domain
    logger, log_path = create_logger(log_dir=log_dir, script_name="01_build_ground_truth")

    start = time.time()
    logger.info(f"=== Starting 01_build_ground_truth for domain={domain} ===")

    try:
        out_path = build_ground_truth_for_domain(domain=domain, logger=logger)
    except Exception as e:
        logger.error(f"Unhandled error in 01_build_ground_truth: {e}", exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 01_build_ground_truth in {elapsed:.2f} seconds")
    logger.info(f"Paragraph master file: {out_path}")
    logger.info(f"Log file: {log_path}")

    print(f"Paragraph master written to: {out_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
