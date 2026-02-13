#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import yaml

from fandom_span_id_retrieval.retrieval.paragraph_embeddings import build_paragraph_embeddings, expand_config
from fandom_span_id_retrieval.retrieval.paragraph_faiss import build_paragraph_faiss_indices
from fandom_span_id_retrieval.retrieval.paragraph_queries import build_paragraph_queries
from fandom_span_id_retrieval.retrieval.paragraph_retrieval_eval import eval_paragraph_retrieval
from fandom_span_id_retrieval.retrieval.paragraph_rerank_prep import prepare_paragraph_rerank_data
from fandom_span_id_retrieval.rerank.trainer import train_rerank_from_config
from fandom_span_id_retrieval.rerank.eval import eval_rerank_from_config
from fandom_span_id_retrieval.utils.aggregate_results import aggregate_results_main
from fandom_span_id_retrieval.utils.seed_utils import set_seed
from fandom_span_id_retrieval.utils.experiment_registry import write_task_metadata


Step = Tuple[str, str, Callable[[dict], None]]


def _sanitize_variant(name: str) -> str:
    return name.replace("+", "_plus_")


def _run_rerank(cfg: dict) -> None:
    emb_cfg = cfg.get("embeddings", {})
    rerank_cfg = cfg.get("rerank", {})
    eval_cfg = cfg.get("retrieval_eval", {})

    data_root = Path(rerank_cfg["output_dir"])
    variants = emb_cfg.get("text_variants", ["paragraph_text"])

    for model_name in emb_cfg.get("models", []):
        model_tag = model_name.replace("/", "__")
        for variant in variants:
            variant_tag = _sanitize_variant(variant)
            data_dir = data_root / model_tag / variant_tag
            out_dir = data_root / f"{model_tag}__{variant_tag}__cross-encoder"
            out_dir.mkdir(parents=True, exist_ok=True)

            train_cfg = {
                "dataset": {
                    "data_dir": str(data_dir),
                    "max_length": int(rerank_cfg.get("max_length", 256)),
                },
                "model": {
                    "encoder_name": rerank_cfg["model_name"],
                },
                "train": {
                    "batch_size": int(rerank_cfg.get("batch_size", 16)),
                    "epochs": int(rerank_cfg.get("epochs", 3)),
                    "lr": float(rerank_cfg.get("lr", 2e-5)),
                    "weight_decay": float(rerank_cfg.get("weight_decay", 0.01)),
                    "warmup_ratio": float(rerank_cfg.get("warmup_ratio", 0.1)),
                },
                "output": {
                    "dir": str(out_dir),
                },
            }

            train_rerank_from_config(train_cfg)

            eval_config = {
                "dataset": {
                    "data_path": str(data_dir / "test.jsonl"),
                    "max_length": int(rerank_cfg.get("max_length", 256)),
                    "k_list": eval_cfg.get("k_list", [1, 5, 10]),
                },
                "output": {
                    "dir": str(out_dir),
                    "ckpt_name": "best_model.pt",
                },
            }
            eval_rerank_from_config(eval_config)


def _run_aggregate_results() -> None:
    aggregate_results_main()


def _steps() -> List[Step]:
    return [
        ("01", "generate_queries", lambda cfg: build_paragraph_queries(cfg)),
        ("02", "build_embeddings", lambda cfg: build_paragraph_embeddings(cfg)),
        ("03", "build_faiss", lambda cfg: build_paragraph_faiss_indices(cfg)),
        ("04", "eval_retrieval", lambda cfg: eval_paragraph_retrieval(cfg)),
        ("05", "prepare_rerank", lambda cfg: prepare_paragraph_rerank_data(cfg)),
        ("06", "run_rerank", _run_rerank),
        ("07", "aggregate_results", lambda cfg: _run_aggregate_results()),
    ]


def _select_steps(start: str, end: str) -> List[Step]:
    steps = _steps()
    ids = [s[0] for s in steps]
    if start not in ids:
        raise ValueError(f"Unknown start step: {start}. Valid: {ids}")
    if end not in ids:
        raise ValueError(f"Unknown end step: {end}. Valid: {ids}")

    start_idx = ids.index(start)
    end_idx = ids.index(end)
    if start_idx > end_idx:
        raise ValueError("start must be <= end")

    return steps[start_idx:end_idx + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval/paragraph_faiss.yaml",
        help="Path to paragraph FAISS config",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="01",
        help="Start step (01-06)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="07",
        help="End step (01-07)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = expand_config(cfg)

    seed = int(cfg.get("experiment", {}).get("seed", 0))
    set_seed(seed, deterministic=True)

    for step_id, step_name, step_fn in _select_steps(args.start, args.end):
        print(f"\n=== Step {step_id}: {step_name} ===")
        step_fn(cfg)

    domain = str(cfg.get("experiment", {}).get("domain", ""))
    out_dir = Path("outputs") / "retrieval" / domain
    write_task_metadata(
        out_dir,
        {
            "task": "paragraph_retrieval",
            "domain": domain,
            "config": args.config,
        },
    )


if __name__ == "__main__":
    main()
