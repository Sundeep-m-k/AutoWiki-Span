from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def _variant_mode(variant: str) -> str:
    if variant in {"minilm-l6"}:
        return "article_classifier"
    if variant in {"lead_paragraph", "full_text"}:
        return "article_faiss"
    if variant in {"paragraph_text", "paragraph_text+title"}:
        return "paragraph_faiss"
    return "article_classifier"


def _get_text_and_id(level: str, rec: Dict[str, object]) -> Tuple[str, str]:
    if level == "page":
        text = str(rec.get("page_text", ""))
        pid = str(rec.get("page_id", ""))
        return text, pid
    text = str(rec.get("paragraph_text", ""))
    pid = str(rec.get("paragraph_id", ""))
    return text, pid


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
    variants = lp_cfg.get("retrieval_variants", ["minilm-l6"])
    run_all_levels = bool(lp_cfg.get("run_all_levels", True))
    run_all_variants = bool(lp_cfg.get("run_all_variants", True))

    if not run_all_levels:
        levels = [str(lp_cfg.get("level", "paragraph"))]
    if not run_all_variants:
        variants = [str(lp_cfg.get("retrieval_variant", "minilm-l6"))]

    out_root = Path(lp_cfg.get("output_dir", f"outputs/linking_pipeline/{domain}"))
    out_root.mkdir(parents=True, exist_ok=True)

    pairs_csv = out_root / "predictions_pairs.csv"
    summary_csv = out_root / "predictions_summary.csv"

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

    log_dir = out_root / "logs"
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

        for variant in variants:
            logger.info(f"Running level={level} variant={variant}")
            retriever = _load_retriever(variant, lp_cfg)

            out_dir = out_root / level / variant.replace("+", "_plus_")
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
