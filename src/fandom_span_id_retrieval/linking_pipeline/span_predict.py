from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from fandom_span_id_retrieval.span_id.preprocess import ID2LABEL


def _bilou_decode(tags: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None

    for i, tag in enumerate(tags):
        if tag == "O" or tag == "O-SPAN":
            if start is not None:
                spans.append((start, i - 1))
                start = None
            continue

        if tag.startswith("U-"):
            spans.append((i, i))
            start = None
        elif tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag.startswith("I-"):
            continue
        elif tag.startswith("L-"):
            if start is None:
                start = i
            spans.append((start, i))
            start = None

    if start is not None:
        spans.append((start, len(tags) - 1))

    return spans


def _token_spans_to_char_spans(
    offsets: List[Tuple[int, int]],
    token_spans: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    char_spans: List[Tuple[int, int]] = []
    for t_start, t_end in token_spans:
        toks = offsets[t_start:t_end + 1]
        toks = [t for t in toks if t[1] > t[0]]
        if not toks:
            continue
        start = min(t[0] for t in toks)
        end = max(t[1] for t in toks)
        if end > start:
            char_spans.append((start, end))
    return char_spans


def load_span_model(model_dir: str | None, model_name: str) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    source = model_name
    if model_dir and Path(model_dir).exists():
        source = model_dir
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(source)
    return model, tokenizer


def predict_spans(
    text: str,
    model,
    tokenizer,
    max_seq_length: int = 512,
    stride: int = 128,
    device: str | None = None,
) -> List[Dict[str, object]]:
    if not text:
        return []

    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offset_mapping = encoded["offset_mapping"].tolist()

    all_spans: List[Tuple[int, int]] = []

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

    for seq_idx, pred_ids in enumerate(preds):
        offsets = offset_mapping[seq_idx]
        tags = [ID2LABEL.get(int(i), "O") for i in pred_ids]
        token_spans = _bilou_decode(tags)
        char_spans = _token_spans_to_char_spans(offsets, token_spans)
        all_spans.extend(char_spans)

    # De-duplicate spans from overlapping windows
    unique = sorted(set(all_spans))
    results: List[Dict[str, object]] = []
    for start, end in unique:
        span_text = text[start:end]
        if not span_text.strip():
            continue
        results.append({
            "start": start,
            "end": end,
            "span_text": span_text,
        })

    return results
