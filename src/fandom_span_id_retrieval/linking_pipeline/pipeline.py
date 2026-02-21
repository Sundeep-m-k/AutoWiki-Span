from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from fandom_span_id_retrieval.linking_pipeline.link_predict import (
    ArticleClassifierRetriever,
    ArticleFaissRetriever,
    ParagraphFaissRetriever,
)
from fandom_span_id_retrieval.linking_pipeline.span_predict import load_span_model, predict_spans
from fandom_span_id_retrieval.span_id.preprocess import normalize_punctuation
from fandom_span_id_retrieval.retrieval.paragraph_embeddings import _expand_placeholders
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_span_cache(span_cache_path: Path) -> Dict[str, Dict[str, object]]:
    cache: Dict[str, Dict[str, object]] = {}
    if not span_cache_path.exists():
        return cache
    with span_cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_id = str(rec.get("id", ""))
            if not rec_id:
                continue
            cache[rec_id] = rec
    return cache


def _build_context(text: str, anchor: str, start: int, end: int, window: int) -> str:
    if not text:
        return anchor
    n = len(text)
    start = max(0, min(start, n))
    end = max(0, min(end, n))
    win_start = max(0, start - window)
    win_end = min(n, end + window)
    snippet = text[win_start:win_end]

    rel_start = start - win_start
    rel_end = end - win_start
    if 0 <= rel_start <= rel_end <= len(snippet):
        return snippet[:rel_start] + "[ANCHOR] " + anchor + snippet[rel_end:]

    if anchor and anchor in snippet:
        return snippet.replace(anchor, "[ANCHOR] " + anchor, 1)

    return snippet


def _build_query(text: str, anchor: str, start: int, end: int, window: int, mode: str) -> str:
    if mode == "anchor_only":
        return anchor
    if mode == "full_text":
        return f"{text} [ANCHOR] {anchor}" if text else anchor
    # default: context with anchor
    return _build_context(text, anchor, start, end, window)


def _detect_retrieval_level(variant: str) -> str:
    """Detect which retrieval level a variant belongs to"""
    if variant in {"minilm-l6"}:
        level = "article"
    elif variant in {"lead_paragraph", "full_text"}:
        level = "article"
    elif variant in {"paragraph_text", "paragraph_text+title"}:
        level = "paragraph"
    else:
        level = "article"
    return level


def _load_retriever_from_variant(variant: str, domain: str, outputs_root: Path, lp_cfg: Dict[str, object]):
    """Load retriever using pre-computed models from variant pipeline outputs"""
    # Determine retrieval level
    retrieval_level = _detect_retrieval_level(variant)
    variant_tag = variant.replace("+", "_plus_")
    variant_output_root = Path(f"data/processed/{domain}") / variant_tag
    
    if retrieval_level == "article":
        # Use article-level retriever
        retrieval_dir = variant_output_root / "article_level" / "retrieval"
        
        # Try FAISS first (for lead_paragraph, full_text variants)
        faiss_dir = retrieval_dir / "faiss"
        if faiss_dir.exists():
            return ArticleFaissRetriever(
                index_dir=str(faiss_dir),
                variant=variant,
                encoder_name=lp_cfg.get("retrieval", {}).get("article_faiss", {}).get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
                retriever_ckpt=str(retrieval_dir / "best_model.pt"),
                max_length=int(lp_cfg.get("retrieval", {}).get("article_faiss", {}).get("max_length", 256)),
                normalize=bool(lp_cfg.get("retrieval", {}).get("article_faiss", {}).get("normalize", True)),
            )
        
        # Fall back to classifier
        return ArticleClassifierRetriever(
            ckpt_path=str(retrieval_dir / "best_model.pt"),
            max_length=int(lp_cfg.get("retrieval", {}).get("article_classifier", {}).get("max_length", 256)),
        )
    
    else:  # paragraph
        retrieval_dir = variant_output_root / "paragraph_level" / "retrieval"
        data_dir = Path(lp_cfg.get("data_dir", f"data/processed/{domain}"))
        
        return ParagraphFaissRetriever(
            index_root=str(retrieval_dir),
            model_name=lp_cfg.get("retrieval", {}).get("paragraph_faiss", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            variant=variant,
            paragraphs_csv=str(data_dir / f"paragraphs_{domain}.csv"),
            paragraph_id_field=str(lp_cfg.get("retrieval", {}).get("paragraph_faiss", {}).get("paragraph_id_field", "paragraph_id")),
            max_length=int(lp_cfg.get("retrieval", {}).get("paragraph_faiss", {}).get("max_length", 256)),
            normalize=bool(lp_cfg.get("retrieval", {}).get("paragraph_faiss", {}).get("normalize", True)),
        )


def _normalize_variant_name(variant: str) -> str:
    """Normalize variant name for directory paths"""
    return variant.replace("+", "_plus_")


def _variant_mode(variant: str) -> str:
    if variant in {"minilm-l6"}:
        return "article_classifier"
    if variant in {"lead_paragraph", "full_text"}:
        return "article_faiss"
    if variant in {"paragraph_text", "paragraph_text+title"}:
        return "paragraph_faiss"
    return "article_classifier"


def _load_retriever(variant: str, lp_cfg: Dict[str, object]):
    mode = _variant_mode(variant)
    retrieval_cfg = lp_cfg.get("retrieval", {})

    if mode == "article_classifier":
        art_cfg = retrieval_cfg.get("article_classifier", {})
        return ArticleClassifierRetriever(
            ckpt_path=str(art_cfg["ckpt_path"]),
            max_length=int(art_cfg.get("max_length", 256)),
        )

    if mode == "article_faiss":
        art_cfg = retrieval_cfg.get("article_faiss", {})
        return ArticleFaissRetriever(
            index_dir=str(art_cfg["index_dir"]),
            variant=variant,
            encoder_name=art_cfg.get("encoder_name"),
            retriever_ckpt=art_cfg.get("retriever_ckpt"),
            max_length=int(art_cfg.get("max_length", 256)),
            normalize=bool(art_cfg.get("normalize", True)),
        )

    para_cfg = retrieval_cfg.get("paragraph_faiss", {})
    return ParagraphFaissRetriever(
        index_root=str(para_cfg["index_root"]),
        model_name=str(para_cfg["model_name"]),
        variant=variant,
        paragraphs_csv=str(para_cfg["paragraphs_csv"]),
        paragraph_id_field=str(para_cfg.get("paragraph_id_field", "paragraph_id")),
        max_length=int(para_cfg.get("max_length", 256)),
        normalize=bool(para_cfg.get("normalize", True)),
    )


def run_linking_pipeline(config_path: Path) -> List[Path]:
    cfg_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lp_cfg_raw = cfg_raw.get("linking_pipeline", {})
    domain = str(lp_cfg_raw.get("domain", ""))
    cfg = _expand_placeholders(cfg_raw, {"domain": domain})
    lp_cfg = cfg.get("linking_pipeline", {})
    levels = lp_cfg.get("levels", ["paragraph"])
    variants = lp_cfg.get("retrieval_variants", ["lead_paragraph", "full_text"])
    run_all_levels = bool(lp_cfg.get("run_all_levels", True))
    run_all_variants = bool(lp_cfg.get("run_all_variants", True))

    if not run_all_levels:
        levels = [str(lp_cfg.get("level", "paragraph"))]
    if not run_all_variants:
        variants = [str(lp_cfg.get("retrieval_variant", "lead_paragraph"))]

    # Point to variant pipeline outputs instead of old structure
    outputs_root = Path(lp_cfg.get("output_dir", "outputs")).parent / "linking_pipeline" / domain
    outputs_root.mkdir(parents=True, exist_ok=True)

    pairs_csv = outputs_root / "predictions_pairs.csv"
    summary_csv = outputs_root / "predictions_summary.csv"

    pairs_fields = [
        "level",
        "variant",
        "id",
        "start",
        "end",
        "span_text",
        "target_article_id",
        "rank",
        "score",
    ]
    summary_fields = [
        "level",
        "variant",
        "id",
        "num_pairs",
    ]

    pairs_file_exists = pairs_csv.exists()
    summary_file_exists = summary_csv.exists()

    log_dir = outputs_root / "logs"
    logger, _ = create_logger(log_dir, script_name="linking_pipeline")
    logger.info(f"Domain: {domain}")
    logger.info(f"Levels: {levels}")
    logger.info(f"Variants: {variants}")

    span_cfg = lp_cfg.get("span_model", {})
    model_dir = span_cfg.get("model_dir")
    model_name = span_cfg.get("model_name", "bert-base-uncased")
    max_seq_length = int(span_cfg.get("max_seq_length", 512))
    stride = int(span_cfg.get("stride", 128))
    normalize_punct = bool(span_cfg.get("normalize_punctuation", False))

    use_cached_spans = bool(lp_cfg.get("use_cached_spans", False))
    span_cache_path_tpl = str(lp_cfg.get("span_cache_path", ""))

    model, tokenizer = load_span_model(model_dir, model_name)

    data_dir = Path(lp_cfg.get("data_dir", f"data/processed/{domain}"))
    max_examples = int(lp_cfg.get("max_examples", 0))

    query_cfg = lp_cfg.get("query", {})
    context_window = int(query_cfg.get("context_window", 200))
    query_mode = str(query_cfg.get("mode", "context_anchor"))

    top_k = int(lp_cfg.get("top_k", 1))
    include_scores = bool(lp_cfg.get("include_scores", True))

    outputs: List[Path] = []

    for level in levels:
        if level == "page":
            input_path = data_dir / f"pages_{domain}.jsonl"
        else:
            input_path = data_dir / f"paragraphs_{domain}.jsonl"

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        span_cache: Dict[str, Dict[str, object]] = {}
        if use_cached_spans and span_cache_path_tpl:
            cache_path = Path(span_cache_path_tpl.replace("{level}", level))
            span_cache = _load_span_cache(cache_path)
            if span_cache:
                logger.info(f"Loaded span cache: {cache_path} ({len(span_cache)} records)")
            else:
                logger.info(f"Span cache not found or empty: {cache_path}")

        for variant in variants:
            logger.info(f"Running level={level} variant={variant}")
            
            # Load retriever from pre-computed variant pipeline outputs
            try:
                retriever = _load_retriever_from_variant(variant, domain, outputs_root, lp_cfg)
            except Exception as e:
                logger.warning(f"Failed to load retriever for variant {variant}: {e}")
                logger.info(f"Falling back to legacy config-based loading...")
                retriever = _load_retriever(variant, lp_cfg)

            variant_tag = _normalize_variant_name(variant)
            out_dir = outputs_root / variant_tag / level
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "predictions.jsonl"

            count = 0
            with out_path.open("w", encoding="utf-8") as f, \
                pairs_csv.open("a", encoding="utf-8", newline="") as pairs_f, \
                summary_csv.open("a", encoding="utf-8", newline="") as summary_f:
                pairs_writer = csv.DictWriter(pairs_f, fieldnames=pairs_fields)
                summary_writer = csv.DictWriter(summary_f, fieldnames=summary_fields)
                if not pairs_file_exists:
                    pairs_writer.writeheader()
                    pairs_file_exists = True
                if not summary_file_exists:
                    summary_writer.writeheader()
                    summary_file_exists = True
                for rec in _read_jsonl(input_path):
                    raw_text, rec_id = _get_text_and_id(level, rec)
                    cached = span_cache.get(rec_id) if span_cache else None
                    if cached:
                        text = str(cached.get("text", raw_text))
                        spans = cached.get("spans", []) or []
                    else:
                        text = normalize_punctuation(raw_text) if normalize_punct else raw_text
                        if not text.strip():
                            continue
                        spans = predict_spans(
                            text=text,
                            model=model,
                            tokenizer=tokenizer,
                            max_seq_length=max_seq_length,
                            stride=stride,
                        )

                    pairs = []
                    for sp in spans:
                        anchor = str(sp.get("span_text", ""))
                        start = int(sp.get("start", 0))
                        end = int(sp.get("end", 0))
                        query_text = _build_query(text, anchor, start, end, context_window, query_mode)
                        preds = retriever.predict(query_text, top_k=top_k)
                        if not preds:
                            continue
                        for rank, (aid, score) in enumerate(preds, start=1):
                            item = {
                                "start": start,
                                "end": end,
                                "target_article_id": aid,
                                "rank": rank,
                            }
                            if include_scores:
                                item["score"] = score
                            pairs.append(item)

                            pairs_writer.writerow({
                                "level": level,
                                "variant": variant,
                                "id": rec_id,
                                "start": start,
                                "end": end,
                                "span_text": anchor,
                                "target_article_id": aid,
                                "rank": rank,
                                "score": score if include_scores else "",
                            })

                    row = {
                        "level": level,
                        "id": rec_id,
                        "text": text,
                        "predicted_pairs": pairs,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    summary_writer.writerow({
                        "level": level,
                        "variant": variant,
                        "id": rec_id,
                        "num_pairs": len(pairs),
                    })
                    count += 1
                    if max_examples > 0 and count >= max_examples:
                        break

            logger.info(f"Wrote {count} records to {out_path}")
            outputs.append(out_path)

    return outputs
