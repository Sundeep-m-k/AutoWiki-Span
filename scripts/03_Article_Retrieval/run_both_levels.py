#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from fandom_span_id_retrieval.eval.retrieval_eval import eval_retrieval_from_config
from fandom_span_id_retrieval.rerank.eval import eval_rerank_from_config
from fandom_span_id_retrieval.rerank.prep import prepare_rerank_data
from fandom_span_id_retrieval.rerank.trainer import train_rerank_from_config
from fandom_span_id_retrieval.retrieval.article_queries import build_article_queries
from fandom_span_id_retrieval.retrieval.faiss_index import build_faiss_index_from_config
from fandom_span_id_retrieval.retrieval.paragraph_embeddings import build_paragraph_embeddings, expand_config
from fandom_span_id_retrieval.retrieval.paragraph_faiss import build_paragraph_faiss_indices
from fandom_span_id_retrieval.retrieval.paragraph_queries import build_paragraph_queries
from fandom_span_id_retrieval.retrieval.paragraph_retrieval_eval import eval_paragraph_retrieval
from fandom_span_id_retrieval.retrieval.paragraph_rerank_prep import prepare_paragraph_rerank_data
from fandom_span_id_retrieval.retrieval.prep import prepare_retrieval_data
from fandom_span_id_retrieval.retrieval.trainer import train_retrieval_from_config
from fandom_span_id_retrieval.utils.aggregate_results import aggregate_results_main
from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return expand_config(cfg)


def _apply_domain_override(cfg: Dict[str, Any], domain: str) -> Dict[str, Any]:
    exp_cfg = cfg.setdefault("experiment", {})
    exp_cfg["domain"] = domain
    return expand_config(cfg)


def _variant_dir_name(variant: str) -> str:
    return variant.replace("+", "_plus_")


def _run_article_pipeline(cfg: Dict[str, Any]) -> None:
    retrieval_cfg = cfg.get("retrieval", {})
    data_cfg = retrieval_cfg.get("data", {})
    train_cfg = retrieval_cfg.get("train", {})
    model_cfg = retrieval_cfg.get("model", {})
    out_cfg = retrieval_cfg.get("output", {})

    if retrieval_cfg.get("queries", {}).get("do_build", False):
        build_article_queries(cfg)

    if retrieval_cfg.get("prepare", {}).get("do_prepare", False):
        prepare_retrieval_data(cfg)

    if train_cfg.get("do_train", False):
        train_retrieval_from_config({
            "dataset": {
                "data_dir": data_cfg["data_dir"],
                "max_length": int(train_cfg.get("max_length", 256)),
            },
            "model": model_cfg,
            "train": train_cfg,
            "output": out_cfg,
        })

    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("do_eval", False):
        eval_retrieval_from_config({
            "dataset": {
                "data_dir": data_cfg["data_dir"],
                "max_length": int(eval_cfg.get("max_length", train_cfg.get("max_length", 256))),
            },
            "output": {
                "dir": out_cfg["dir"],
                "ckpt_name": eval_cfg.get("ckpt_name", "best_model.pt"),
            },
        })

    if cfg.get("faiss", {}).get("do_build", False):
        build_faiss_index_from_config(cfg)


def _run_article_rerank(cfg: Dict[str, Any]) -> None:
    rerank_cfg = cfg.get("rerank", {})
    data_cfg = rerank_cfg.get("data", {})
    model_cfg = rerank_cfg.get("model", {})
    train_cfg = rerank_cfg.get("train", {})
    out_cfg = rerank_cfg.get("output", {})
    eval_cfg = cfg.get("eval", {})

    output_roots: List[tuple[str, Path]] = []
    if rerank_cfg.get("prepare", {}).get("do_prepare", False):
        output_roots = prepare_rerank_data(cfg)
    else:
        variants = rerank_cfg.get("text_variants", ["lead_paragraph"])
        for variant in variants:
            output_roots.append((variant, Path(data_cfg["data_dir"]) / "rerank" / _variant_dir_name(variant)))

    if train_cfg.get("do_train", False):
        for variant, data_dir in output_roots:
            out_dir = Path(out_cfg["dir"]) / _variant_dir_name(variant)
            train_rerank_from_config({
                "dataset": {
                    "data_dir": str(data_dir),
                    "max_length": int(train_cfg.get("max_length", 256)),
                },
                "model": model_cfg,
                "train": train_cfg,
                "output": {
                    "dir": str(out_dir),
                },
            })

    if eval_cfg.get("do_eval", False):
        for variant, data_dir in output_roots:
            out_dir = Path(out_cfg["dir"]) / _variant_dir_name(variant)
            eval_rerank_from_config({
                "dataset": {
                    "data_path": str(data_dir / "test.jsonl"),
                    "max_length": int(eval_cfg.get("max_length", train_cfg.get("max_length", 256))),
                    "k_list": eval_cfg.get("k_list", [1, 5, 10]),
                },
                "output": {
                    "dir": str(out_dir),
                    "ckpt_name": eval_cfg.get("ckpt_name", "best_model.pt"),
                },
            })


def _run_paragraph_pipeline(cfg: Dict[str, Any], do_rerank: bool) -> None:
    build_paragraph_queries(cfg)
    build_paragraph_embeddings(cfg)
    build_paragraph_faiss_indices(cfg)
    eval_paragraph_retrieval(cfg)

    if do_rerank:
        prepare_paragraph_rerank_data(cfg)
        _run_paragraph_rerank(cfg)


def _run_paragraph_rerank(cfg: Dict[str, Any]) -> None:
    emb_cfg = cfg.get("embeddings", {})
    rerank_cfg = cfg.get("rerank", {})
    eval_cfg = cfg.get("retrieval_eval", {})

    data_root = Path(rerank_cfg["output_dir"])
    variants = emb_cfg.get("text_variants", ["paragraph_text"])

    for model_name in emb_cfg.get("models", []):
        model_tag = model_name.replace("/", "__")
        for variant in variants:
            variant_tag = _variant_dir_name(variant)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval/both_levels.yaml",
        help="Path to combined pipeline config",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        pipeline_cfg = yaml.safe_load(f) or {}

    pipeline = pipeline_cfg.get("pipeline", {})
    do_article = bool(pipeline.get("do_article", True))
    do_paragraph = bool(pipeline.get("do_paragraph", True))
    do_article_rerank = bool(pipeline.get("do_article_rerank", True))
    do_paragraph_rerank = bool(pipeline.get("do_paragraph_rerank", True))
    do_aggregate = bool(pipeline.get("do_aggregate", True))
    domain_override = pipeline_cfg.get("domain")
    if not domain_override:
        raise ValueError("pipeline domain is required; set it in configs/retrieval/both_levels.yaml")

    article_cfg = _load_config(pipeline_cfg["article_config"])
    paragraph_cfg = _load_config(pipeline_cfg["paragraph_config"])
    article_rerank_cfg = _load_config(pipeline_cfg["article_rerank_config"])

    article_cfg = _apply_domain_override(article_cfg, domain_override)
    paragraph_cfg = _apply_domain_override(paragraph_cfg, domain_override)
    article_rerank_cfg = _apply_domain_override(article_rerank_cfg, domain_override)

    if do_article:
        _run_article_pipeline(article_cfg)
        if do_article_rerank:
            _run_article_rerank(article_rerank_cfg)

    if do_paragraph:
        _run_paragraph_pipeline(paragraph_cfg, do_paragraph_rerank)

    if do_aggregate:
        aggregate_results_main()


if __name__ == "__main__":
    main()
