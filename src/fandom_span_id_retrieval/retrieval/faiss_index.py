from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from fandom_span_id_retrieval.eval.retrieval_eval import load_retrieval_model_for_eval
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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


def _load_articles(path: Path, variant: str) -> Tuple[List[int], List[str]]:
    article_ids: List[int] = []
    texts: List[str] = []
    for obj in _read_jsonl(path):
        aid = int(obj["article_id"])
        text = _article_text(obj, variant)
        article_ids.append(aid)
        texts.append(text)
    return article_ids, texts


def _encode_texts(
    texts: List[str],
    tokenizer,
    encoder,
    max_length: int,
    batch_size: int,
    device,
) -> np.ndarray:
    encoder.to(device)
    encoder.eval()

    all_embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode articles", unit="batch"):
        batch_texts = texts[i:i + batch_size]
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
            cls = outputs.last_hidden_state[:, 0, :]
            emb = cls.detach().cpu().numpy()
        all_embs.append(emb)

    embeddings = np.vstack(all_embs).astype(np.float32)
    return embeddings


def build_faiss_index_from_config(config: Dict[str, Any]) -> List[Path]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "faiss is not installed. Install faiss-cpu or faiss-gpu to use this feature."
        ) from exc

    faiss_cfg = config.get("faiss", {})
    data_cfg = config.get("retrieval", {}).get("data", {})
    out_cfg = faiss_cfg.get("output", {})

    data_dir = Path(data_cfg["data_dir"])
    articles_path = data_dir / "articles.jsonl"
    if not articles_path.is_file():
        raise FileNotFoundError(f"articles.jsonl not found in {data_dir}")

    out_dir = Path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    logger, _ = create_logger(log_dir, script_name="faiss_build")
    logger.info(f"Articles path: {articles_path}")
    logger.info(f"Output dir: {out_dir}")

    text_variants = faiss_cfg.get("text_variants")
    if not text_variants:
        text_variants = [faiss_cfg.get("text_variant", "lead_paragraph")]
    max_length = int(faiss_cfg.get("max_length", 256))
    batch_size = int(faiss_cfg.get("batch_size", 32))
    normalize = bool(faiss_cfg.get("normalize", True))
    metric = str(faiss_cfg.get("metric", "ip")).lower()
    use_gpu = bool(faiss_cfg.get("use_gpu", True))
    gpu_device = int(faiss_cfg.get("gpu_device", 0))
    ckpt_path = faiss_cfg.get("retriever_ckpt")
    encoder_name = faiss_cfg.get("encoder_name")

    logger.info(f"text_variants={text_variants}")
    logger.info(f"max_length={max_length}, batch_size={batch_size}")
    logger.info(f"normalize={normalize}, metric={metric}")
    logger.info(f"use_gpu={use_gpu}, gpu_device={gpu_device}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if ckpt_path:
        model, enc_name = load_retrieval_model_for_eval(str(ckpt_path))
        encoder = model.encoder
        encoder_name = enc_name
        logger.info(f"Using retriever checkpoint: {ckpt_path}")
    else:
        if not encoder_name:
            raise ValueError("encoder_name is required when retriever_ckpt is not set")
        encoder = AutoModel.from_pretrained(encoder_name)
        logger.info(f"Using encoder: {encoder_name}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    output_paths: List[Path] = []

    for variant in text_variants:
        variant_dir = out_dir
        if len(text_variants) > 1:
            variant_dir = out_dir / _variant_dir_name(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)

        variant_log_dir = variant_dir / "logs"
        logger, _ = create_logger(variant_log_dir, script_name="faiss_build")
        logger.info(f"text_variant={variant}")

        article_ids, texts = _load_articles(articles_path, variant)
        logger.info(f"Articles loaded: {len(article_ids)}")

        embeddings = _encode_texts(
            texts=texts,
            tokenizer=tokenizer,
            encoder=encoder,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            embeddings = embeddings / norms

        dim = embeddings.shape[1]
        if metric == "ip":
            index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("metric must be 'ip' or 'l2'")

        if use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("FAISS GPU support not available. Install faiss-gpu.")
            logger.info("Moving FAISS index to GPU.")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_device, index)

        index.add(embeddings)

        index_name = out_cfg.get("index_name", "articles.faiss")
        emb_name = out_cfg.get("embeddings_name", "article_embeddings.npy")
        map_name = out_cfg.get("mapping_name", "article_ids.json")

        if use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, str(variant_dir / index_name))
        np.save(variant_dir / emb_name, embeddings)
        with (variant_dir / map_name).open("w", encoding="utf-8") as f:
            json.dump(article_ids, f, indent=2)

        logger.info(f"Saved FAISS index: {variant_dir / index_name}")
        logger.info(f"Saved embeddings: {variant_dir / emb_name}")
        logger.info(f"Saved article id map: {variant_dir / map_name}")

        output_paths.append(variant_dir / index_name)

    return output_paths
