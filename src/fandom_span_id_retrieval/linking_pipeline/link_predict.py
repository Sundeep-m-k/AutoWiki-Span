from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fandom_span_id_retrieval.eval.retrieval_eval import load_retrieval_model_for_eval
from fandom_span_id_retrieval.retrieval.paragraph_embeddings import _mean_pool


def _variant_dir_name(variant: str) -> str:
    return variant.replace("+", "_plus_")


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "__")


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


def _validate_manifest(manifest_path: Path, expected: Dict[str, str]) -> None:
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected_val in expected.items():
        actual = data.get(key)
        if expected_val and actual and str(actual) != str(expected_val):
            raise ValueError(
                f"FAISS manifest mismatch for {manifest_path}: {key}={actual} expected={expected_val}"
            )


class ArticleClassifierRetriever:
    def __init__(self, ckpt_path: str, max_length: int = 256):
        self.model, encoder_name = load_retrieval_model_for_eval(ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, query_text: str, top_k: int = 1) -> List[Tuple[int, float]]:
        encoded = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded)["logits"].squeeze(0)
        scores, idxs = torch.topk(logits, k=top_k)
        return [(int(i), float(s)) for i, s in zip(idxs.tolist(), scores.tolist())]


class ArticleFaissRetriever:
    def __init__(
        self,
        index_dir: str,
        variant: str,
        encoder_name: str | None,
        retriever_ckpt: str | None,
        max_length: int = 256,
        normalize: bool = True,
    ):
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise ImportError("faiss is not installed. Install faiss-cpu or faiss-gpu.") from exc

        variant_dir = Path(index_dir) / _variant_dir_name(variant)
        index_path = variant_dir / "articles.faiss"
        ids_path = variant_dir / "article_ids.json"
        if not index_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(f"Missing FAISS index or ids in {variant_dir}")

        _validate_manifest(
            variant_dir / "manifest.json",
            {
                "text_variant": variant,
                "encoder_name": encoder_name or "",
            },
        )

        self.index = faiss.read_index(str(index_path))
        self.article_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self.max_length = max_length
        self.normalize = normalize

        if retriever_ckpt:
            model, enc_name = load_retrieval_model_for_eval(retriever_ckpt)
            self.encoder = model.encoder
            encoder_name = enc_name
        else:
            if not encoder_name:
                raise ValueError("encoder_name is required when retriever_ckpt is not set")
            self.encoder = AutoModel.from_pretrained(encoder_name)

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

    def _encode_query(self, query_text: str) -> np.ndarray:
        encoded = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.encoder(**encoded)
            cls = outputs.last_hidden_state[:, 0, :]
        emb = cls.detach().cpu().numpy().astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            emb = emb / norms
        return emb

    def predict(self, query_text: str, top_k: int = 1) -> List[Tuple[int, float]]:
        emb = self._encode_query(query_text)
        scores, idxs = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            results.append((int(self.article_ids[idx]), float(score)))
        return results


class ParagraphFaissRetriever:
    def __init__(
        self,
        index_root: str,
        model_name: str,
        variant: str,
        paragraphs_csv: str,
        paragraph_id_field: str = "paragraph_id",
        max_length: int = 256,
        normalize: bool = True,
    ):
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise ImportError("faiss is not installed. Install faiss-cpu or faiss-gpu.") from exc

        model_dir = Path(index_root) / _sanitize_model_name(model_name) / _variant_dir_name(variant)
        index_path = model_dir / "paragraphs.faiss"
        ids_path = model_dir / "paragraph_ids.json"
        if not index_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(f"Missing paragraph FAISS index in {model_dir}")

        _validate_manifest(
            model_dir / "manifest.json",
            {
                "model_name": model_name,
                "text_variant": variant,
            },
        )

        self.index = faiss.read_index(str(index_path))
        self.paragraph_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self.paragraph_map = _load_paragraph_map(Path(paragraphs_csv), paragraph_id_field)
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

    def _encode_query(self, query_text: str) -> np.ndarray:
        encoded = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.encoder(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        emb = pooled.detach().cpu().numpy().astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            emb = emb / norms
        return emb

    def predict(self, query_text: str, top_k: int = 1) -> List[Tuple[int, float]]:
        emb = self._encode_query(query_text)
        scores, idxs = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            pid = str(self.paragraph_ids[idx])
            aid = self.paragraph_map.get(pid)
            if not aid:
                continue
            results.append((int(aid), float(score)))
        return results
