from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from fandom_span_id_retrieval.retrieval.paragraph_embeddings import expand_config, _mean_pool
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "__")


def _sanitize_variant_name(name: str) -> str:
    return name.replace("+", "_plus_")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_paragraph_map(csv_path: Path, id_field: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get(id_field, "")
            article_id = row.get("article_id", "")
            if pid and article_id:
                mapping[pid] = article_id
    return mapping


def _encode_queries(
    queries: List[str],
    tokenizer,
    encoder,
    max_length: int,
    batch_size: int,
    device,
    normalize: bool,
) -> np.ndarray:
    encoder.to(device)
    encoder.eval()

    all_embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(queries), batch_size), desc="encode queries", unit="batch"):
        batch_texts = queries[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = encoder(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            emb = pooled.detach().cpu().numpy()
        all_embs.append(emb)

    embeddings = np.vstack(all_embs).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embeddings = embeddings / norms
    return embeddings


def eval_paragraph_retrieval(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "faiss is not installed. Install faiss-cpu or faiss-gpu to use this feature."
        ) from exc

    cfg = expand_config(cfg)
    para_cfg = cfg.get("paragraphs", {})
    emb_cfg = cfg.get("embeddings", {})
    faiss_cfg = cfg.get("faiss", {})
    query_cfg = cfg.get("queries", {})
    eval_cfg = cfg.get("retrieval_eval", {})

    queries_path = Path(query_cfg["output_path"])
    if not queries_path.is_file():
        raise FileNotFoundError(f"queries not found: {queries_path}")

    paragraph_map = _load_paragraph_map(
        csv_path=Path(para_cfg["csv_path"]),
        id_field=para_cfg.get("id_field", "paragraph_id"),
    )

    out_root = Path(eval_cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    log_dir = out_root / "logs"
    logger, _ = create_logger(log_dir, script_name="paragraph_eval")
    logger.info(f"Queries path: {queries_path}")

    queries = list(_read_jsonl(queries_path))
    query_texts = [q["query_text"] for q in queries]
    target_article_ids = [str(q["target_article_id"]) for q in queries]

    k_list = tuple(eval_cfg.get("k_list", [1, 5, 10]))
    batch_size = int(emb_cfg.get("batch_size", 32))
    max_length = int(emb_cfg.get("max_length", 256))
    normalize = bool(emb_cfg.get("normalize", True))
    trust_remote_code = bool(emb_cfg.get("trust_remote_code", False))

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    variants = emb_cfg.get("text_variants", ["paragraph_text"])

    for model_name in emb_cfg.get("models", []):
        results[model_name] = {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for variant in variants:
            logger.info(f"Evaluating model={model_name} variant={variant}")
            model_dir = Path(faiss_cfg["output_dir"]) / _sanitize_model_name(model_name) / _sanitize_variant_name(variant)
            index_path = model_dir / "paragraphs.faiss"
            ids_path = model_dir / "paragraph_ids.json"
            if not index_path.is_file() or not ids_path.is_file():
                raise FileNotFoundError(f"Missing FAISS index for {model_name}/{variant}")

            index = faiss.read_index(str(index_path))
            paragraph_ids = json.loads(ids_path.read_text(encoding="utf-8"))

            query_embs = _encode_queries(
                queries=query_texts,
                tokenizer=tokenizer,
                encoder=encoder,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
                normalize=normalize,
            )

            scores, idxs = index.search(query_embs, max(k_list))

            correct_at_k = {k: 0 for k in k_list}
            total = len(queries)

            for i in range(total):
                retrieved_paragraphs = [
                    paragraph_ids[j] for j in idxs[i].tolist() if j >= 0
                ]
                retrieved_article_ids = [paragraph_map.get(pid, "") for pid in retrieved_paragraphs]
                target_id = target_article_ids[i]
                for k in k_list:
                    if target_id in retrieved_article_ids[:k]:
                        correct_at_k[k] += 1

            metrics = {f"recall@{k}": correct_at_k[k] / total for k in k_list}
            results[model_name][variant] = metrics

            out_dir = out_root / _sanitize_model_name(model_name) / _sanitize_variant_name(variant)
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Metrics for {model_name}/{variant}: {metrics}")

    summary_path = out_root / "all_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
