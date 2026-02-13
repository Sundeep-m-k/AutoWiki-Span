from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from fandom_span_id_retrieval.utils.logging_utils import create_logger


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


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "__")


def _sanitize_variant_name(name: str) -> str:
    return name.replace("+", "_plus_")


def _build_paragraph_text(row: Dict[str, str], variant: str) -> str:
    text = row.get("text", "")
    title = row.get("title", "")
    page_name = row.get("page_name", "")

    if variant in {"lead_paragraph", "paragraph_text", "full_text"}:
        return text
    if variant in {"lead_paragraph+title", "paragraph_text+title"}:
        if title:
            return f"{title}\n{text}" if text else title
        return text
    if variant in {"paragraph_text+page_name"}:
        if page_name:
            return f"{page_name}\n{text}" if text else page_name
        return text
    return text


def _load_paragraphs(
    csv_path: Path,
    id_field: str,
    text_field: str,
    title_field: str,
    page_name_field: str,
) -> Tuple[List[str], List[Dict[str, str]]]:
    paragraph_ids: List[str] = []
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get(id_field, "")
            text = row.get(text_field, "")
            if not pid or not text:
                continue
            paragraph_ids.append(pid)
            rows.append({
                "text": text,
                "title": row.get(title_field, ""),
                "page_name": row.get(page_name_field, ""),
            })
    return paragraph_ids, rows


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


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
    for i in tqdm(range(0, len(texts), batch_size), desc="encode paragraphs", unit="batch"):
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
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            emb = pooled.detach().cpu().numpy()
        all_embs.append(emb)

    embeddings = np.vstack(all_embs).astype(np.float32)
    return embeddings


def build_paragraph_embeddings(cfg: Dict[str, Any]) -> List[Path]:
    cfg = expand_config(cfg)
    exp = cfg.get("experiment", {})
    para_cfg = cfg.get("paragraphs", {})
    emb_cfg = cfg.get("embeddings", {})

    csv_path = Path(para_cfg["csv_path"])
    if not csv_path.is_file():
        raise FileNotFoundError(f"paragraph CSV not found: {csv_path}")

    output_dir = Path(emb_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    logger, _ = create_logger(log_dir, script_name="paragraph_embed")
    logger.info(f"CSV path: {csv_path}")

    paragraph_ids, rows = _load_paragraphs(
        csv_path=csv_path,
        id_field=para_cfg.get("id_field", "paragraph_id"),
        text_field=para_cfg.get("text_field", "paragraph_text"),
        title_field=para_cfg.get("title_field", "title"),
        page_name_field=para_cfg.get("page_name_field", "page_name"),
    )
    logger.info(f"Paragraphs loaded: {len(paragraph_ids)}")

    models = emb_cfg.get("models", [])
    variants = emb_cfg.get("text_variants", ["paragraph_text"])
    max_length = int(emb_cfg.get("max_length", 256))
    batch_size = int(emb_cfg.get("batch_size", 32))
    normalize = bool(emb_cfg.get("normalize", True))
    trust_remote_code = bool(emb_cfg.get("trust_remote_code", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_paths: List[Path] = []

    for model_name in models:
        model_dir = output_dir / _sanitize_model_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        for variant in variants:
            variant_dir = model_dir / _sanitize_variant_name(variant)
            variant_dir.mkdir(parents=True, exist_ok=True)
            texts = [_build_paragraph_text(row, variant) for row in rows]

            logger.info(f"Encoding model={model_name} variant={variant}")
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

            np.save(variant_dir / "paragraph_embeddings.npy", embeddings)
            with (variant_dir / "paragraph_ids.json").open("w", encoding="utf-8") as f:
                json.dump(paragraph_ids, f, indent=2)
            (variant_dir / "model_name.txt").write_text(model_name, encoding="utf-8")
            (variant_dir / "variant_name.txt").write_text(variant, encoding="utf-8")

            logger.info(f"Saved embeddings to {variant_dir}")
            output_paths.append(variant_dir)

    return output_paths
