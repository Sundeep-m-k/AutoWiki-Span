from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from fandom_span_id_retrieval.rerank.dataset import RerankDataset
from fandom_span_id_retrieval.rerank.trainer import load_rerank_model
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _group_by_query(examples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for ex in examples:
        qid = int(ex["query_id"])
        groups.setdefault(qid, []).append(ex)
    return groups


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_rerank(
    data_path: Path,
    ckpt_path: Path,
    max_length: int,
    k_list: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    model, encoder_name = load_rerank_model(ckpt_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    rows = _read_jsonl(data_path)
    grouped = _group_by_query(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct_at_k = {k: 0 for k in k_list}
    total = 0

    for qid, examples in tqdm(grouped.items(), desc="rerank eval", unit="query"):
        total += 1
        scores: List[Tuple[float, int]] = []
        for ex in examples:
            encoded = tokenizer(
                ex["query_text"],
                ex["doc_text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits.squeeze(0)
            if logits.numel() == 1:
                score = float(logits.squeeze())
            else:
                score = float(logits[1])
            scores.append((score, int(ex["label"])))

        scores.sort(key=lambda x: x[0], reverse=True)
        labels_sorted = [lbl for _, lbl in scores]

        for k in k_list:
            if any(labels_sorted[:k]):
                correct_at_k[k] += 1

    metrics = {f"recall@{k}": correct_at_k[k] / total for k in k_list}
    return metrics


def eval_rerank_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    data_cfg = config["dataset"]
    out_cfg = config["output"]

    data_path = Path(data_cfg["data_path"])
    ckpt_path = Path(out_cfg["dir"]) / out_cfg.get("ckpt_name", "best_model.pt")

    log_dir = Path(out_cfg["dir"]) / "logs"
    logger, _ = create_logger(log_dir, script_name="rerank_eval")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Checkpoint: {ckpt_path}")

    metrics = evaluate_rerank(
        data_path=data_path,
        ckpt_path=ckpt_path,
        max_length=int(data_cfg["max_length"]),
        k_list=tuple(data_cfg.get("k_list", (1, 5, 10))),
    )

    with open(Path(out_cfg["dir"]) / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics: {metrics}")

    return metrics
