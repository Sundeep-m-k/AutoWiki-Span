#!/usr/bin/env python3
"""
Clean Retrieval + Reranking Pipeline Architecture

Flow per variant:
- Article Retrieval → Article Reranking
- Paragraph Retrieval → Paragraph Reranking

Directory structure:
outputs/{domain}/
├── {variant}/
│   ├── article_level/
│   │   ├── retrieval/
│   │   └── reranking/
│   └── paragraph_level/
│       ├── retrieval/
│       └── reranking/
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

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
from fandom_span_id_retrieval.utils.seed_utils import set_seed
from fandom_span_id_retrieval.utils.experiment_registry import write_task_metadata


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return expand_config(cfg)


def _apply_overrides(cfg: Dict[str, Any], domain: str, variant: str) -> Dict[str, Any]:
    """Apply domain and variant overrides to config"""
    exp_cfg = cfg.setdefault("experiment", {})
    exp_cfg["domain"] = domain
    exp_cfg["variant"] = variant
    return expand_config(cfg)


def _variant_dir_name(variant: str) -> str:
    return variant.replace("+", "_plus_")


def _create_variant_output_paths(base_cfg: Dict[str, Any], domain: str, variant: str) -> Dict[str, Path]:
    """Create standardized output paths for variant-based architecture"""
    variant_tag = _variant_dir_name(variant)
    outputs_root = PROJECT_ROOT / "outputs" / domain / variant_tag
    
    return {
        "variant_root": outputs_root,
        "article_retrieval": outputs_root / "article_level" / "retrieval",
        "article_reranking": outputs_root / "article_level" / "reranking",
        "paragraph_retrieval": outputs_root / "paragraph_level" / "retrieval",
        "paragraph_reranking": outputs_root / "paragraph_level" / "reranking",
    }


def _run_article_retrieval(cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Run article-level retrieval"""
    print(f"\n{'='*60}")
    print(f"Running Article Retrieval")
    print(f"{'='*60}")
    
    retrieval_cfg = cfg.get("retrieval", {})
    data_cfg = retrieval_cfg.get("data", {})
    train_cfg = retrieval_cfg.get("train", {})
    model_cfg = retrieval_cfg.get("model", {})
    
    out_dir = paths["article_retrieval"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if retrieval_cfg.get("queries", {}).get("do_build", False):
        print("Building article queries...")
        build_article_queries(cfg)

    if retrieval_cfg.get("prepare", {}).get("do_prepare", False):
        print("Preparing retrieval data...")
        prepare_retrieval_data(cfg)

    if train_cfg.get("do_train", False):
        print("Training article retriever...")
        train_retrieval_from_config({
            "dataset": {
                "data_dir": data_cfg["data_dir"],
                "max_length": int(train_cfg.get("max_length", 256)),
            },
            "model": model_cfg,
            "train": train_cfg,
            "output": {
                "dir": str(out_dir),
            },
        })

    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("do_eval", False):
        print("Evaluating article retriever...")
        eval_retrieval_from_config({
            "dataset": {
                "data_dir": data_cfg["data_dir"],
                "max_length": int(eval_cfg.get("max_length", train_cfg.get("max_length", 256))),
            },
            "k_list": eval_cfg.get("k_list", [1, 3, 5, 10, 50, 100]),
            "output": {
                "dir": str(out_dir),
                "ckpt_name": eval_cfg.get("ckpt_name", "best_model.pt"),
            },
        })

    # Build FAISS index
    if cfg.get("faiss", {}).get("do_build", False):
        print("Building FAISS index...")
        faiss_cfg = cfg["faiss"].copy()
        faiss_cfg["output"] = {
            "dir": str(out_dir / "faiss"),
            "index_name": "articles.faiss",
            "embeddings_name": "article_embeddings.npy",
            "mapping_name": "article_ids.json",
        }
        faiss_cfg["retriever_ckpt"] = str(out_dir / "best_model.pt")
        
        modified_cfg = cfg.copy()
        modified_cfg["faiss"] = faiss_cfg
        build_faiss_index_from_config(modified_cfg)


def _run_article_reranking(cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Run article-level reranking"""
    print(f"\n{'='*60}")
    print(f"Running Article Reranking")
    print(f"{'='*60}")
    
    rerank_cfg = cfg.get("rerank", {})
    data_cfg = rerank_cfg.get("data", {})
    model_cfg = rerank_cfg.get("model", {})
    train_cfg = rerank_cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    
    out_dir = paths["article_reranking"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if rerank_cfg.get("prepare", {}).get("do_prepare", False):
        print("Preparing reranking data...")
        # Modify retriever checkpoint path
        rerank_cfg_modified = rerank_cfg.copy()
        rerank_cfg_modified["retriever"] = {
            "ckpt_path": str(paths["article_retrieval"] / "best_model.pt")
        }
        
        cfg_modified = cfg.copy()
        cfg_modified["rerank"] = rerank_cfg_modified
        prepare_rerank_data(cfg_modified)
        
        # Move prepared data to our output directory
        data_root = Path(data_cfg["data_dir"]) / "rerank"
        # The prepare_rerank_data function creates variant dirs, we'll use those

    if train_cfg.get("do_train", False):
        print("Training article reranker...")
        variant = cfg.get("experiment", {}).get("variant", "lead_paragraph")
        variant_tag = _variant_dir_name(variant)
        data_dir = Path(data_cfg["data_dir"]) / "rerank" / variant_tag
        
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
        print("Evaluating article reranker...")
        variant = cfg.get("experiment", {}).get("variant", "lead_paragraph")
        variant_tag = _variant_dir_name(variant)
        data_dir = Path(data_cfg["data_dir"]) / "rerank" / variant_tag
        
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


def _run_paragraph_retrieval(cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Run paragraph-level retrieval"""
    print(f"\n{'='*60}")
    print(f"Running Paragraph Retrieval")
    print(f"{'='*60}")
    
    # Build queries
    print("Building paragraph queries...")
    build_paragraph_queries(cfg)
    
    # Build embeddings
    print("Building paragraph embeddings...")
    build_paragraph_embeddings(cfg)
    
    # Build FAISS indices
    print("Building FAISS indices...")
    build_paragraph_faiss_indices(cfg)
    
    # Evaluate paragraph retrieval
    print("Evaluating paragraph retrieval...")
    eval_paragraph_retrieval(cfg)


def _run_paragraph_reranking(cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Run paragraph-level reranking"""
    print(f"\n{'='*60}")
    print(f"Running Paragraph Reranking")
    print(f"{'='*60}")
    
    rerank_cfg = cfg.get("rerank", {})
    retrieval_eval_cfg = cfg.get("retrieval_eval", {})
    emb_cfg = cfg.get("embeddings", {})
    
    out_dir = paths["paragraph_reranking"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing paragraph reranking data...")
    prepare_paragraph_rerank_data(cfg)
    
    # Train and evaluate for each model and variant
    data_root = Path(rerank_cfg["output_dir"])
    variant = cfg.get("experiment", {}).get("variant", "paragraph_text")
    variant_tag = _variant_dir_name(variant)
    
    for model_name in emb_cfg.get("models", []):
        model_tag = model_name.replace("/", "__")
        data_dir = data_root / model_tag / variant_tag
        
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            continue
        
        model_out_dir = out_dir / f"{model_tag}__cross-encoder"
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training paragraph reranker ({model_name})...")
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
                "dir": str(model_out_dir),
            },
        }
        
        train_rerank_from_config(train_cfg)
        
        print(f"Evaluating paragraph reranker ({model_name})...")
        eval_config = {
            "dataset": {
                "data_path": str(data_dir / "test.jsonl"),
                "max_length": int(rerank_cfg.get("max_length", 256)),
                "k_list": retrieval_eval_cfg.get("k_list", [1, 5, 10]),
            },
            "output": {
                "dir": str(model_out_dir),
                "ckpt_name": "best_model.pt",
            },
        }
        eval_rerank_from_config(eval_config)


def run_variant_pipeline(
    article_cfg: Dict[str, Any],
    paragraph_cfg: Dict[str, Any],
    article_rerank_cfg: Dict[str, Any],
    paragraph_rerank_cfg: Dict[str, Any],
    domain: str,
    variant: str,
    do_article: bool = True,
    do_article_rerank: bool = True,
    do_paragraph: bool = True,
    do_paragraph_rerank: bool = True,
) -> None:
    """Run complete pipeline for a single variant"""
    print(f"\n{'#'*60}")
    print(f"# Processing Variant: {variant}")
    print(f"# Domain: {domain}")
    print(f"{'#'*60}")
    
    # Apply overrides
    article_cfg = _apply_overrides(article_cfg, domain, variant)
    article_rerank_cfg = _apply_overrides(article_rerank_cfg, domain, variant)
    paragraph_cfg = _apply_overrides(paragraph_cfg, domain, variant)
    paragraph_rerank_cfg = _apply_overrides(paragraph_rerank_cfg, domain, variant)
    
    # Create output paths
    paths = _create_variant_output_paths(article_cfg, domain, variant)
    
    # Article Level
    if do_article:
        _run_article_retrieval(article_cfg, paths)
        if do_article_rerank:
            _run_article_reranking(article_rerank_cfg, paths)
    
    # Paragraph Level
    if do_paragraph:
        _run_paragraph_retrieval(paragraph_cfg, paths)
        if do_paragraph_rerank:
            _run_paragraph_reranking(paragraph_rerank_cfg, paths)
    
    # Write metadata
    write_task_metadata(
        paths["article_retrieval"],
        {
            "task": "article_retrieval",
            "domain": domain,
            "variant": variant,
            "config": "configs/retrieval/experiment.yaml",
        },
    )
    write_task_metadata(
        paths["article_reranking"],
        {
            "task": "article_reranking",
            "domain": domain,
            "variant": variant,
            "config": "configs/rerank/experiment.yaml",
        },
    )
    write_task_metadata(
        paths["paragraph_retrieval"],
        {
            "task": "paragraph_retrieval",
            "domain": domain,
            "variant": variant,
            "config": "configs/retrieval/paragraph_faiss.yaml",
        },
    )
    write_task_metadata(
        paths["paragraph_reranking"],
        {
            "task": "paragraph_reranking",
            "domain": domain,
            "variant": variant,
            "config": "configs/retrieval/paragraph_faiss.yaml",
        },
    )
    
    print(f"\n✓ Completed pipeline for variant: {variant}")


def _variant_pipeline_worker(args_tuple: tuple) -> tuple[str, bool]:
    """Worker function for parallel variant processing"""
    (variant, article_cfg, paragraph_cfg, article_rerank_cfg, 
     paragraph_rerank_cfg, domain, do_article, do_article_rerank, 
     do_paragraph, do_paragraph_rerank) = args_tuple
    
    try:
        run_variant_pipeline(
            article_cfg=article_cfg,
            paragraph_cfg=paragraph_cfg,
            article_rerank_cfg=article_rerank_cfg,
            paragraph_rerank_cfg=paragraph_rerank_cfg,
            domain=domain,
            variant=variant,
            do_article=do_article,
            do_article_rerank=do_article_rerank,
            do_paragraph=do_paragraph,
            do_paragraph_rerank=do_paragraph_rerank,
        )
        return (variant, True)
    except Exception as e:
        print(f"❌ Error processing variant {variant}: {e}")
        return (variant, False)


def _load_variant_metrics(domain: str, variant: str) -> Optional[Dict[str, Any]]:
    """Load metrics from a completed variant"""
    try:
        variant_tag = _variant_dir_name(variant)
        variant_root = PROJECT_ROOT / "outputs" / domain / variant_tag
        
        metrics = {"variant": variant, "timestamp": datetime.now().isoformat()}
        
        # Load article retrieval metrics
        art_ret_metrics = variant_root / "article_level" / "retrieval" / "test_metrics.json"
        if art_ret_metrics.exists():
            with open(art_ret_metrics) as f:
                metrics["article_retrieval"] = json.load(f)
        
        # Load article rerank metrics
        art_rerank_metrics = variant_root / "article_level" / "reranking" / "test_metrics.json"
        if art_rerank_metrics.exists():
            with open(art_rerank_metrics) as f:
                metrics["article_reranking"] = json.load(f)
        
        # Load paragraph retrieval metrics
        para_ret_metrics = variant_root / "paragraph_level" / "retrieval" / "test_metrics.json"
        if para_ret_metrics.exists():
            with open(para_ret_metrics) as f:
                metrics["paragraph_retrieval"] = json.load(f)
        
        # Load paragraph rerank metrics
        para_rerank_metrics = variant_root / "paragraph_level" / "reranking" / "test_metrics.json"
        if para_rerank_metrics.exists():
            with open(para_rerank_metrics) as f:
                metrics["paragraph_reranking"] = json.load(f)
        
        return metrics
    except Exception as e:
        print(f"⚠️  Could not load metrics for variant {variant}: {e}")
        return None


def _progressive_aggregate(domain: str, completed_variants: List[str]) -> None:
    """Update results CSV as variants complete (progressive output)"""
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  pandas not available, skipping progressive aggregation")
        return
    
    results = {"variant": [], "level": [], "task": [], "metric": [], "value": [], "timestamp": []}
    
    for variant in completed_variants:
        metrics = _load_variant_metrics(domain, variant)
        if not metrics:
            continue
        
        timestamp = metrics.get("timestamp", "")
        
        # Flatten metrics into rows
        for level in ["article_retrieval", "article_reranking", "paragraph_retrieval", "paragraph_reranking"]:
            if level in metrics:
                level_metrics = metrics[level]
                for metric_name, metric_value in level_metrics.items():
                    task_name = level.replace("_retrieval", "").replace("_reranking", "")
                    task_type = "retrieval" if "retrieval" in level else "reranking"
                    
                    results["variant"].append(variant)
                    results["level"].append(task_name)
                    results["task"].append(task_type)
                    results["metric"].append(metric_name)
                    results["value"].append(metric_value)
                    results["timestamp"].append(timestamp)
    
    # Write progressive results
    if results["variant"]:
        outputs_dir = PROJECT_ROOT / "outputs" / domain
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        output_path = outputs_dir / "results_progressive.csv"
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Progressive results updated: {output_path}")
        print(f"  Completed variants: {len(completed_variants)}/{len(set(results['variant']))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Variant-Based Retrieval + Reranking Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval/both_levels.yaml",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run only a specific variant (e.g., 'lead_paragraph'). If not specified, runs all variants.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run variants in parallel (requires multiprocessing)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2). Recommended: 2-3 for RTX A6000",
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Update results CSV as variants complete (requires pandas)",
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
    domain = pipeline_cfg.get("domain")
    
    if not domain:
        raise ValueError("pipeline domain is required; set it in your config")

    # Load base configs
    article_cfg = _load_config(pipeline_cfg.get("article_config", "configs/retrieval/experiment.yaml"))
    paragraph_cfg = _load_config(pipeline_cfg.get("paragraph_config", "configs/retrieval/paragraph_faiss.yaml"))
    article_rerank_cfg = _load_config(pipeline_cfg.get("article_rerank_config", "configs/rerank/experiment.yaml"))
    paragraph_rerank_cfg = _load_config(pipeline_cfg.get("paragraph_config", "configs/retrieval/paragraph_faiss.yaml"))
    
    seed = int(article_cfg.get("experiment", {}).get("seed", 0))
    set_seed(seed, deterministic=True)
    
    # Determine which variants to process
    variants = [args.variant] if args.variant else article_cfg.get("faiss", {}).get("text_variants", ["lead_paragraph"])
    
    # Run pipeline
    if args.parallel and len(variants) > 1:
        print(f"\n{'#'*60}")
        print(f"# Parallel Variant Pipeline")
        print(f"# Running {len(variants)} variants with {args.num_workers} workers")
        print(f"{'#'*60}")
        
        # Prepare worker arguments
        worker_args = [
            (
                variant,
                article_cfg.copy(),
                paragraph_cfg.copy(),
                article_rerank_cfg.copy(),
                paragraph_rerank_cfg.copy(),
                domain,
                do_article,
                do_article_rerank,
                do_paragraph,
                do_paragraph_rerank,
            )
            for variant in variants
        ]
        
        # Run variants in parallel
        completed_variants = []
        with Pool(processes=args.num_workers) as pool:
            results = pool.imap_unordered(_variant_pipeline_worker, worker_args)
            for variant, success in results:
                if success:
                    completed_variants.append(variant)
                    if args.progressive:
                        _progressive_aggregate(domain, completed_variants)
                    print(f"\n✓ Variant '{variant}' completed successfully")
                else:
                    print(f"\n❌ Variant '{variant}' failed")
    else:
        # Sequential execution
        print(f"\n{'#'*60}")
        print(f"# Sequential Variant Pipeline")
        print(f"# Running {len(variants)} variant(s) sequentially")
        print(f"{'#'*60}")
        
        for variant in variants:
            run_variant_pipeline(
                article_cfg=article_cfg.copy(),
                paragraph_cfg=paragraph_cfg.copy(),
                article_rerank_cfg=article_rerank_cfg.copy(),
                paragraph_rerank_cfg=paragraph_rerank_cfg.copy(),
                domain=domain,
                variant=variant,
                do_article=do_article,
                do_article_rerank=do_article_rerank,
                do_paragraph=do_paragraph,
                do_paragraph_rerank=do_paragraph_rerank,
            )
            
            if args.progressive:
                _progressive_aggregate(domain, [variant])
    
    if do_aggregate:
        print(f"\n{'='*60}")
        print("Aggregating final results...")
        print(f"{'='*60}")
        aggregate_results_main()
    
    print(f"\n{'#'*60}")
    print("✓ Pipeline completed successfully!")
    print(f"Output directory: {PROJECT_ROOT / 'outputs' / domain}")
    if args.progressive:
        print(f"Progressive results: {PROJECT_ROOT / 'outputs' / domain / 'results_progressive.csv'}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
