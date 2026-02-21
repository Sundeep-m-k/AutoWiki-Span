# src/fandom_span_id_retrieval/span_id/preprocess.py

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from fandom_span_id_retrieval.utils.logging_utils import create_logger
from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT


BILOU_LABELS = ["O", "B-SPAN", "I-SPAN", "L-SPAN", "U-SPAN"]
LABEL2ID = {label: i for i, label in enumerate(BILOU_LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def md5_unit_interval(s: str) -> float:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) / (2**128)


def normalize_punctuation(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def assign_bilou_labels(
    text: str,
    spans: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
) -> Tuple[List[int], List[int], List[int]]:
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=True,
        padding=True,
    )
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    labels = [LABEL2ID["O"]] * len(input_ids)

    for sp in spans:
        start_char = int(sp["start"])
        end_char = int(sp["end"])
        if end_char <= start_char:
            continue

        token_indices: List[int] = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:  # special tokens
                continue
            if tok_end <= start_char or tok_start >= end_char:
                continue
            token_indices.append(i)

        if not token_indices:
            continue

        if len(token_indices) == 1:
            labels[token_indices[0]] = LABEL2ID["U-SPAN"]
        else:
            labels[token_indices[0]] = LABEL2ID["B-SPAN"]
            for ti in token_indices[1:-1]:
                labels[ti] = LABEL2ID["I-SPAN"]
            labels[token_indices[-1]] = LABEL2ID["L-SPAN"]

    return input_ids, attention_mask, labels


def build_token_dataset_from_cfg(span_cfg: Dict[str, Any]) -> None:
    """
    Convert paragraph- or page-level ground-truth JSONL into token-level BILOU splits.
    Uses:
      - data/processed/<domain>/paragraphs_<domain>.jsonl when level == "paragraph"
    - data/processed/<domain>/pages_<domain>.jsonl       when level == "page"
    """
    log_dir = PROJECT_ROOT / span_cfg["log_dir"]
    logger, _ = create_logger(log_dir, script_name="02_build_token_dataset")
    logger.info("Step 2: Building token-level BILOU dataset from ground truth")

    domain = span_cfg["domain"]
    level = span_cfg.get("level", "paragraph")

    if level == "paragraph":
        in_name = f"paragraphs_{domain}.jsonl"
    elif level == "page":
        in_name = f"pages_{domain}.jsonl"
    else:
        raise ValueError(f"Unknown level: {level}")

    input_path = PROJECT_ROOT / "data" / "processed" / domain / in_name
    if not input_path.exists():
        logger.error(f"Ground-truth JSONL not found: {input_path}")
        raise FileNotFoundError(input_path)

    logger.info(f"Reading {level}-level data from: {input_path}")

    model_name = span_cfg["model_name"]
    max_seq_length = int(span_cfg.get("max_seq_length", 512))
    normalize_punct = span_cfg.get("normalize_punctuation", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logger.info(f"Loaded tokenizer: {model_name}, max_seq_length={max_seq_length}")
    logger.info(f"Punctuation normalization: {normalize_punct}")

    out_dir = PROJECT_ROOT / span_cfg["token_dataset_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = PROJECT_ROOT / span_cfg["train_file"]
    dev_path = PROJECT_ROOT / span_cfg["dev_file"]
    test_path = PROJECT_ROOT / span_cfg["test_file"]
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_f = train_path.open("w", encoding="utf-8")
    dev_f = dev_path.open("w", encoding="utf-8")
    test_f = test_path.open("w", encoding="utf-8")

    train_ratio = 0.8
    dev_ratio = 0.1

    num_examples = 0
    num_skipped_empty = 0
    total_spans = 0
    total_tokens = 0

    with input_path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            text = rec.get("paragraph_text") or rec.get("page_text") or ""
            if not text.strip():
                num_skipped_empty += 1
                continue

            if normalize_punct:
                text = normalize_punctuation(text)

            links = rec.get("links") or []
            spans: List[Dict[str, Any]] = []
            for lk in links:
                if lk.get("link_type") != "internal":
                    continue
                spans.append(
                    {
                        "start": int(lk["link_rel_start"]),
                        "end": int(lk["link_rel_end"]),
                        "anchor_text": lk.get("anchor_text", ""),
                    }
                )

            input_ids, attention_mask, labels = assign_bilou_labels(
                text=text,
                spans=spans,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )

            total_spans += len(spans)
            total_tokens += len(input_ids)

            example = {
                "article_id": rec.get("article_id"),
                "page_name": rec.get("page_name"),
                "paragraph_id": rec.get("paragraph_id") or rec.get("page_id"),
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "label_ids": labels,
            }

            key_str = f"{rec.get('article_id')}::{example['paragraph_id']}"
            r = md5_unit_interval(key_str)

            if r < train_ratio:
                tgt = train_f
            elif r < train_ratio + dev_ratio:
                tgt = dev_f
            else:
                tgt = test_f

            tgt.write(json.dumps(example, ensure_ascii=False) + "\n")
            num_examples += 1

            if idx % 500 == 0:
                logger.info(f"Processed {idx} records -> {num_examples} examples")

    train_f.close()
    dev_f.close()
    test_f.close()

    logger.info("Token dataset written:")
    logger.info(f"  train: {train_path}")
    logger.info(f"  dev:   {dev_path}")
    logger.info(f"  test:  {test_path}")
    logger.info(f"Total examples: {num_examples}")
    logger.info(f"Records skipped (empty/invalid): {num_skipped_empty}")
    if num_examples > 0:
        logger.info(f"Avg spans per example: {total_spans / num_examples:.3f}")
        logger.info(f"Avg tokens per example: {total_tokens / num_examples:.3f}")
