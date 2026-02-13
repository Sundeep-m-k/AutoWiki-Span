# scripts/02_Span_Identification/01_run_span_id_grid.py

from __future__ import annotations

import os
import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml  # pip install pyyaml

from fandom_span_id_retrieval.span_id.preprocess import build_token_dataset_from_cfg
from fandom_span_id_retrieval.span_id.model import (
    train_model_from_cfg,
    evaluate_model_from_cfg,
)
from fandom_span_id_retrieval.utils.seed_utils import set_seed

# Hard-disable wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


GRID_SETTINGS = [
    # (level, normalize_punctuation)
    ("paragraph", False),
    ("paragraph", True),
    ("page", False),
    ("page", True),
]


def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one(span_cfg_base: dict, level: str, normalize_punct: bool):
    span_cfg = copy.deepcopy(span_cfg_base)
    span_cfg["level"] = level
    span_cfg["normalize_punctuation"] = normalize_punct

    seed = int(span_cfg.get("train", {}).get("seed", 0))
    set_seed(seed, deterministic=True)

    domain = span_cfg["domain"]
    raw_model_name = span_cfg["model_name"]
    model_name = raw_model_name.split("/")[-1]

    suffix = f"{model_name}_{level}_{'punc' if not normalize_punct else 'nopunc'}"

    # Token datasets
    base_dir = f"data/processed/{domain}/span_id_{suffix}"
    span_cfg["token_dataset_dir"] = base_dir
    span_cfg["train_file"] = f"{base_dir}/train.jsonl"
    span_cfg["dev_file"] = f"{base_dir}/dev.jsonl"
    span_cfg["test_file"] = f"{base_dir}/test.jsonl"

    # Logs and TB under outputs/span_id
    span_cfg["log_dir"] = f"outputs/span_id/logs/{domain}/{suffix}"
    span_cfg["tensorboard_dir"] = f"outputs/span_id/tensorboard/{domain}/{suffix}"

    # Results stay in outputs/span_id as defined in YAML
    # span_cfg["results_dir"] and span_cfg["results_csv"] come from base cfg

    print(f"=== [SPAN-ID] model={model_name}, level={level}, normalize_punctuation={normalize_punct} ===")

    if span_cfg["train"].get("do_preprocess", True):
        build_token_dataset_from_cfg(span_cfg)

    if span_cfg["train"].get("do_train", True):
        train_model_from_cfg(span_cfg)

    if span_cfg["train"].get("do_eval", True):
        evaluate_model_from_cfg(span_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/span_id/experiment.yaml",
        help="Path to span-id experiment YAML",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="span_id",
        help="YAML section name for span-id config",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    span_cfg_base = cfg[args.section]

    seed = int(span_cfg_base.get("train", {}).get("seed", 0))
    set_seed(seed, deterministic=True)

    for level, np_flag in GRID_SETTINGS:
        run_one(span_cfg_base, level, np_flag)


if __name__ == "__main__":
    main()
