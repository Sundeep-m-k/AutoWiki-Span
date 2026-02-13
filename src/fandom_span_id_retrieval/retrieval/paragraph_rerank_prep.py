from __future__ import annotations

import csv
import json
import random
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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_paragraphs(
    csv_path: Path,
    id_field: str,
    text_field: str,
    title_field: str,
    page_name_field: str,
    variant: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    text_map: Dict[str, str] = {}
    article_map: Dict[str, str] = {}
    by_article: Dict[str, List[str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get(id_field, "")
            text = row.get(text_field, "")
            title = row.get(title_field, "")
            page_name = row.get(page_name_field, "")
            article_id = row.get("article_id", "")
            if not pid or not article_id:
                continue
            if variant in {"lead_paragraph", "paragraph_text", "full_text"}:
                text_map[pid] = text
            elif variant in {"lead_paragraph+title", "paragraph_text+title"}:
                text_map[pid] = f"{title}\n{text}" if title else text
            elif variant == "paragraph_text+page_name":
                text_map[pid] = f"{page_name}\n{text}" if page_name else text
            else:
                text_map[pid] = text
            article_map[pid] = article_id
            by_article.setdefault(article_id, []).append(pid)
    return text_map, article_map, by_article


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


def _split_queries(rows: List[Dict[str, Any]], splits: Dict[str, float], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * splits.get("train", 0.8))
    n_val = int(n * splits.get("val", 0.1))

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    return {"train": train, "val": val, "test": test}


def prepare_paragraph_rerank_data(cfg: Dict[str, Any]) -> List[Path]:
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
    rerank_cfg = cfg.get("rerank", {})
    exp_cfg = cfg.get("experiment", {})

    queries_path = Path(query_cfg["output_path"])
    if not queries_path.is_file():
        raise FileNotFoundError(f"queries not found: {queries_path}")

    out_root = Path(rerank_cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    log_dir = out_root / "logs"
    logger, _ = create_logger(log_dir, script_name="paragraph_rerank_prep")
    logger.info(f"Queries path: {queries_path}")

    rows = _read_jsonl(queries_path)
    splits = rerank_cfg.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = int(exp_cfg.get("seed", 0))
    split_rows = _split_queries(rows, splits, seed)

    candidate_k = int(rerank_cfg.get("candidate_k", 10))
    max_length = int(emb_cfg.get("max_length", 256))
    batch_size = int(emb_cfg.get("batch_size", 32))
    normalize = bool(emb_cfg.get("normalize", True))
    trust_remote_code = bool(emb_cfg.get("trust_remote_code", False))
    variants = emb_cfg.get("text_variants", ["paragraph_text"])

    output_dirs: List[Path] = []

    for model_name in emb_cfg.get("models", []):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for variant in variants:
            logger.info(f"Preparing rerank data for model={model_name} variant={variant}")
            index_dir = Path(faiss_cfg["output_dir"]) / _sanitize_model_name(model_name) / _sanitize_variant_name(variant)
            index_path = index_dir / "paragraphs.faiss"
            ids_path = index_dir / "paragraph_ids.json"
            if not index_path.is_file() or not ids_path.is_file():
                raise FileNotFoundError(f"Missing FAISS index for {model_name}/{variant}")

            text_map, article_map, by_article = _load_paragraphs(
                csv_path=Path(para_cfg["csv_path"]),
                id_field=para_cfg.get("id_field", "paragraph_id"),
                text_field=para_cfg.get("text_field", "paragraph_text"),
                title_field=para_cfg.get("title_field", "title"),
                page_name_field=para_cfg.get("page_name_field", "page_name"),
                variant=variant,
            )

            index = faiss.read_index(str(index_path))
            paragraph_ids = json.loads(ids_path.read_text(encoding="utf-8"))

            model_out = out_root / _sanitize_model_name(model_name) / _sanitize_variant_name(variant)
            model_out.mkdir(parents=True, exist_ok=True)

            for split_name, split_data in split_rows.items():
                queries = [q["query_text"] for q in split_data]
                query_embs = _encode_queries(
                    queries=queries,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    max_length=max_length,
                    batch_size=batch_size,
                    device=device,
                    normalize=normalize,
                )

                scores, idxs = index.search(query_embs, candidate_k)

                out_path = model_out / f"{split_name}.jsonl"
                with out_path.open("w", encoding="utf-8") as f:
                    for i, q in enumerate(split_data):
                        target_article_id = str(q["target_article_id"])
                        candidates = [
                            paragraph_ids[j] for j in idxs[i].tolist() if j >= 0
                        ]

                        if target_article_id in by_article:
                            if not any(article_map.get(pid) == target_article_id for pid in candidates):
                                candidates = candidates[:-1] + [by_article[target_article_id][0]]

                        for pid in candidates:
                            doc_text = text_map.get(pid, "")
                            if not doc_text:
                                continue
                            label = 1 if article_map.get(pid) == target_article_id else 0
                            row = {
                                "query_id": q["query_id"],
                                "query_text": q["query_text"],
                                "article_id": article_map.get(pid),
                                "paragraph_id": pid,
                                "doc_text": doc_text,
                                "label": label,
                            }
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                logger.info(f"Wrote {out_path}")

            output_dirs.append(model_out)

    return output_dirs
