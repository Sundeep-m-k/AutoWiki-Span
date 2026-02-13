from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from fandom_span_id_retrieval.eval.retrieval_eval import load_retrieval_model_for_eval
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_placeholders(value: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, repl in mapping.items():
            value = value.replace("${" + key + "}", repl)
        return value
    if isinstance(value, dict):
        return {k: _expand_placeholders(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_placeholders(v, mapping) for v in value]
    return value


def expand_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp = cfg.get("experiment", {})
    domain = str(exp.get("domain", ""))
    return _expand_placeholders(cfg, {"domain": domain})


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_articles(path: Path) -> Dict[int, Dict[str, Any]]:
    articles: Dict[int, Dict[str, Any]] = {}
    for obj in _read_jsonl(path):
        aid = int(obj["article_id"])
        articles[aid] = obj
    return articles


def _normalize_title(title: str) -> str:
    return title.replace("_", " ") if title else title


def _article_text(article: Dict[str, Any], variant: str) -> str:
    lead = article.get("lead_paragraph", "")
    full_text = article.get("full_text", "")
    title = _normalize_title(article.get("title", ""))

    if variant == "lead_paragraph":
        return lead
    if variant == "full_text":
        return full_text
    if variant == "lead_paragraph+title":
        if title:
            return f"{title}\n{lead}" if lead else title
        return lead
    return full_text or lead


def _variant_dir_name(variant: str) -> str:
    return variant.replace("+", "_plus_")


def _get_topk_available(
    logits: torch.Tensor,
    available_ids: set[int],
    k: int,
) -> List[int]:
    max_k = logits.size(-1)
    k_try = min(max_k, max(k * 2, k))
    while True:
        topk = torch.topk(logits, k=k_try, dim=-1).indices.tolist()
        candidates = [aid for aid in topk if aid in available_ids]
        if len(candidates) >= k or k_try >= max_k:
            return candidates[:k]
        k_try = min(max_k, k_try * 2)


def _build_pairs_for_split(
    split_path: Path,
    articles: Dict[int, Dict[str, Any]],
    retriever_model,
    retriever_tokenizer,
    k: int,
    variant: str,
    seed: int,
    logger,
) -> List[Dict[str, Any]]:
    available_ids = set(articles.keys())
    pairs: List[Dict[str, Any]] = []

    for idx, q in enumerate(tqdm(_read_jsonl(split_path), desc=f"build {split_path.name}", unit="query")):
        query_text = q["query_text"]
        target_id = int(q.get("article_id", q.get("target_article_id")))
        query_id = int(q.get("query_id", idx))

        encoded = retriever_tokenizer(
            query_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = retriever_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            logits = outputs["logits"].squeeze(0)

        topk = _get_topk_available(logits, available_ids, k)

        if target_id not in topk and target_id in available_ids:
            topk = topk[:-1] + [target_id]

        rng = random.Random(seed + query_id)
        rng.shuffle(topk)

        for aid in topk:
            article = articles.get(aid)
            if not article:
                continue
            doc_text = _article_text(article, variant)
            label = 1 if aid == target_id else 0
            pairs.append({
                "query_id": query_id,
                "query_text": query_text,
                "article_id": aid,
                "doc_text": doc_text,
                "label": label,
            })

    logger.info(f"Built {len(pairs)} pairs from {split_path.name}")

    return pairs


def prepare_rerank_data(cfg: Dict[str, Any]) -> List[Tuple[str, Path]]:
    exp = cfg.get("experiment", {})
    rerank = cfg.get("rerank", {})
    data_cfg = rerank.get("data", {})

    data_dir = Path(data_cfg["data_dir"])
    log_dir = data_dir / "rerank" / "logs"
    logger, _ = create_logger(log_dir, script_name="rerank_prep")
    logger.info(f"Data dir: {data_dir}")
    articles_path = data_dir / "articles.jsonl"
    if not articles_path.is_file():
        raise FileNotFoundError(f"articles.jsonl not found in {data_dir}")

    articles = _load_articles(articles_path)
    logger.info(f"Articles loaded: {len(articles)}")

    retriever_cfg = rerank.get("retriever", {})
    ckpt_path = Path(retriever_cfg["ckpt_path"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"retriever checkpoint not found: {ckpt_path}")

    retriever_model, encoder_name = load_retrieval_model_for_eval(str(ckpt_path))
    retriever_model.eval()
    logger.info(f"Retriever checkpoint: {ckpt_path}")

    retriever_tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    k = int(rerank.get("candidate_k", 10))
    seed = int(exp.get("seed", 0))
    variants = rerank.get("text_variants", ["lead_paragraph"])
    logger.info(f"candidate_k={k}")
    logger.info(f"text_variants={variants}")

    split_paths = {
        "train": data_dir / "queries_train.jsonl",
        "val": data_dir / "queries_val.jsonl",
        "test": data_dir / "queries_test.jsonl",
    }
    for split, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} split: {path}")

    output_roots: List[Tuple[str, Path]] = []
    for variant in variants:
        variant_dir = data_dir / "rerank" / _variant_dir_name(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Building pairs for variant: {variant}")

        for split, path in split_paths.items():
            pairs = _build_pairs_for_split(
                split_path=path,
                articles=articles,
                retriever_model=retriever_model,
                retriever_tokenizer=retriever_tokenizer,
                k=k,
                variant=variant,
                seed=seed,
                logger=logger,
            )

            out_path = variant_dir / f"{split}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for row in pairs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {out_path}")

        output_roots.append((variant, variant_dir))

    return output_roots
