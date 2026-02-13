#!/usr/bin/env python3
"""
00_scrape_fandom.py

High-level wrapper around scraping pipeline:
- Reads configs/scraping.yaml
- Generates URL list
- Downloads HTML + plain text
"""

import time
from pathlib import Path

from fandom_span_id_retrieval.scraping.scrape_pipeline import run_full_scrape
from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.utils.logging_utils import create_logger
def main() -> None:
    log_dir = PROJECT_ROOT / "data" / "logs" / "scraping"
    logger, log_path = create_logger(log_dir=log_dir, script_name="00_scrape_fandom")

    start = time.time()
    logger.info("=== Starting 00_scrape_fandom ===")

    try:
        url_list_path = run_full_scrape()
    except Exception as e:
        logger.error(f"Unhandled error in 00_scrape_fandom: {e}", exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 00_scrape_fandom in {elapsed:.2f} seconds")
    logger.info(f"URL list: {url_list_path}")
    logger.info(f"Log file: {log_path}")

    print(f"Saved URL list to: {url_list_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()