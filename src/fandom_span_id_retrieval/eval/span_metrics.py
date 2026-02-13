# src/eval/span_metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall


@dataclass
class SpanMetrics:
    """
    Container for span- and token-level metrics.
    """
    span_precision: float
    span_recall: float
    span_f1: float

    token_precision: float
    token_recall: float
    token_f1: float

    exact_span_precision: float
    exact_span_recall: float
    exact_span_f1: float

    relaxed_span_precision: float
    relaxed_span_recall: float
    relaxed_span_f1: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "span_precision": self.span_precision,
            "span_recall": self.span_recall,
            "span_f1": self.span_f1,
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "token_f1": self.token_f1,
            "exact_span_precision": self.exact_span_precision,
            "exact_span_recall": self.exact_span_recall,
            "exact_span_f1": self.exact_span_f1,
            "relaxed_span_precision": self.relaxed_span_precision,
            "relaxed_span_recall": self.relaxed_span_recall,
            "relaxed_span_f1": self.relaxed_span_f1,
        }


def _bilou_decode(labels: List[str]) -> List[Tuple[int, int]]:
    """
    Decode BILOU tags into (start, end) token index spans (inclusive, inclusive).

    Tags are like B-SPAN, I-SPAN, L-SPAN, U-SPAN, O.
    """
    spans: List[Tuple[int, int]] = []
    start = None

    for i, tag in enumerate(labels):
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
        spans.append((start, len(labels) - 1))

    return spans


def _extract_spans_from_ids(
    label_ids: Iterable[int],
    id2label: Dict[int, str],
) -> List[Tuple[int, int]]:
    tags = [id2label[int(i)] for i in label_ids]
    return _bilou_decode(tags)


def _token_prf(y_true: np.ndarray, y_pred: np.ndarray, positive_ids: Iterable[int]) -> Tuple[float, float, float]:
    """
    Token-level precision/recall/F1 over the given positive label IDs.
    """
    positive_ids = set(positive_ids)

    y_true_pos = np.isin(y_true, list(positive_ids))
    y_pred_pos = np.isin(y_pred, list(positive_ids))

    tp = np.logical_and(y_true_pos, y_pred_pos).sum()
    fp = np.logical_and(~y_true_pos, y_pred_pos).sum()
    fn = np.logical_and(y_true_pos, ~y_pred_pos).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def _exact_and_relaxed_prf(
    gold_spans: List[List[Tuple[int, int]]],
    pred_spans: List[List[Tuple[int, int]]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute exact-span and relaxed-span (overlap) precision/recall/F1.

    - Exact: (start, end) must match exactly. [web:134][web:137]
    - Relaxed: counts as match if there is any overlap between predicted span and a gold span,
      matching span-based partial metrics ideas. [web:128][web:130]
    """
    # Exact
    exact_tp = exact_fp = exact_fn = 0
    # Relaxed
    relaxed_tp = relaxed_fp = relaxed_fn = 0

    for g_spans, p_spans in zip(gold_spans, pred_spans):
        g_set = set(g_spans)
        p_set = set(p_spans)

        # Exact: set intersection
        exact_tp_i = len(g_set & p_set)
        exact_fp_i = len(p_set - g_set)
        exact_fn_i = len(g_set - p_set)

        exact_tp += exact_tp_i
        exact_fp += exact_fp_i
        exact_fn += exact_fn_i

        # Relaxed: one-to-one matching based on any overlap
        # We'll greedily match each predicted span to at most one gold span.
        matched_gold = set()
        tp_relaxed_i = 0
        for ps in p_spans:
            p_start, p_end = ps
            found = False
            for gi, gs in enumerate(g_spans):
                if gi in matched_gold:
                    continue
                g_start, g_end = gs
                # overlap if intervals intersect (1 or more tokens shared)
                if not (p_end < g_start or p_start > g_end):
                    matched_gold.add(gi)
                    tp_relaxed_i += 1
                    found = True
                    break
            if not found:
                # unmatched prediction => FP (relaxed)
                pass

        fp_relaxed_i = len(p_spans) - tp_relaxed_i
        fn_relaxed_i = len(g_spans) - tp_relaxed_i

        relaxed_tp += tp_relaxed_i
        relaxed_fp += fp_relaxed_i
        relaxed_fn += fn_relaxed_i

    def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    exact_metrics = _prf(exact_tp, exact_fp, exact_fn)
    relaxed_metrics = _prf(relaxed_tp, relaxed_fp, relaxed_fn)
    return exact_metrics, relaxed_metrics


def compute_span_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str],
    ignore_index: int = -100,
) -> SpanMetrics:
    """
    Compute span-level and token-level metrics given raw logits and gold labels.
    """
    preds = np.argmax(logits, axis=-1)

    # Mask out ignored positions for token-level metrics
    mask = labels != ignore_index
    labels_masked = labels[mask]
    preds_masked = preds[mask]

    # Token-level metrics over any non-"O" span tokens
    span_ids = [i for i, lab in id2label.items() if lab != "O"]
    token_p, token_r, token_f = _token_prf(labels_masked, preds_masked, span_ids)

    # Span-level via tag sequences and BILOU decoding
    true_tag_seqs: List[List[str]] = []
    pred_tag_seqs: List[List[str]] = []
    gold_spans: List[List[Tuple[int, int]]] = []
    pred_spans: List[List[Tuple[int, int]]] = []

    for true_row, pred_row in zip(labels, preds):
        m = true_row != ignore_index
        true_ids = [int(i) for i in true_row[m]]
        pred_ids = [int(i) for i in pred_row[m]]

        true_tags = [id2label[i] for i in true_ids]
        pred_tags = [id2label[i] for i in pred_ids]

        true_tag_seqs.append(true_tags)
        pred_tag_seqs.append(pred_tags)

        gold_spans.append(_bilou_decode(true_tags))
        pred_spans.append(_bilou_decode(pred_tags))

    span_precision = seqeval_precision(true_tag_seqs, pred_tag_seqs)
    span_recall = seqeval_recall(true_tag_seqs, pred_tag_seqs)
    span_f1 = seqeval_f1(true_tag_seqs, pred_tag_seqs)

    (exact_p, exact_r, exact_f), (relaxed_p, relaxed_r, relaxed_f) = _exact_and_relaxed_prf(
        gold_spans, pred_spans
    )

    return SpanMetrics(
        span_precision=span_precision,
        span_recall=span_recall,
        span_f1=span_f1,
        token_precision=token_p,
        token_recall=token_r,
        token_f1=token_f,
        exact_span_precision=exact_p,
        exact_span_recall=exact_r,
        exact_span_f1=exact_f,
        relaxed_span_precision=relaxed_p,
        relaxed_span_recall=relaxed_r,
        relaxed_span_f1=relaxed_f,
    )


def compute_span_metrics_for_trainer(eval_pred, id2label=None, ignore_index: int = -100) -> Dict[str, float]:
    """
    Drop-in compute_metrics hook for HF Trainer.

    If id2label is None, assumes IDs 0..4 map to ["O", "B-SPAN", "I-SPAN", "L-SPAN", "U-SPAN"].
    """
    logits, labels = eval_pred
    logits = np.array(logits)
    labels = np.array(labels)

    if id2label is None:
        id2label = {i: lab for i, lab in enumerate(["O", "B-SPAN", "I-SPAN", "L-SPAN", "U-SPAN"])}

    metrics = compute_span_metrics_from_logits(
        logits=logits,
        labels=labels,
        id2label=id2label,
        ignore_index=ignore_index,
    )
    return metrics.as_dict()
