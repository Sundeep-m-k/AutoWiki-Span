from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys
import numpy as np
from tqdm import tqdm
from fandom_span_id_retrieval.retrieval.paragraph_embeddings import expand_config
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "__")


def _sanitize_variant_name(name: str) -> str:
    return name.replace("+", "_plus_")


def build_paragraph_faiss_indices(cfg: Dict[str, Any]) -> List[Path]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "faiss is not installed. Install faiss-cpu or faiss-gpu to use this feature."
        ) from exc

    cfg = expand_config(cfg)
    emb_cfg = cfg.get("embeddings", {})
    faiss_cfg = cfg.get("faiss", {})

    emb_root = Path(emb_cfg["output_dir"])
    out_root = Path(faiss_cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    log_dir = out_root / "logs"
    logger, _ = create_logger(log_dir, script_name="paragraph_faiss")
    logger.info(f"Embeddings root: {emb_root}")
    logger.info(f"Output root: {out_root}")

    metric = str(faiss_cfg.get("metric", "ip")).lower()
    use_gpu = bool(faiss_cfg.get("use_gpu", True))
    gpu_device = int(faiss_cfg.get("gpu_device", 0))

    output_paths: List[Path] = []

    variants = emb_cfg.get("text_variants", ["paragraph_text"])

    for model_name in emb_cfg.get("models", []):
        for variant in variants:
            model_dir = emb_root / _sanitize_model_name(model_name) / _sanitize_variant_name(variant)
            emb_path = model_dir / "paragraph_embeddings.npy"
            ids_path = model_dir / "paragraph_ids.json"
            if not emb_path.is_file() or not ids_path.is_file():
                raise FileNotFoundError(f"Missing embeddings for {model_name}/{variant} in {model_dir}")

            embeddings = np.load(emb_path).astype(np.float32)
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
                logger.info(f"Moving index to GPU for {model_name}/{variant}.")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, gpu_device, index)

            chunk_size = int(faiss_cfg.get("add_batch_size", 10000))
            if chunk_size <= 0:
                chunk_size = len(embeddings)
            show_tqdm = bool(faiss_cfg.get("show_progress", True))
            progress = None
            if show_tqdm:
                progress = tqdm(
                    total=len(embeddings),
                    desc="faiss add",
                    unit="vecs",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
            total_batches = (len(embeddings) + chunk_size - 1) // chunk_size
            batch_index = 0
            for start in range(0, len(embeddings), chunk_size):
                end = min(start + chunk_size, len(embeddings))
                batch_index += 1
                batch_msg = f"FAISS add batch {batch_index}/{total_batches}: {end - start} vecs"
                logger.info(batch_msg)
                if show_tqdm:
                    print(batch_msg, flush=True)
                index.add(embeddings[start:end])
                if progress is not None:
                    progress.update(end - start)
                msg = f"FAISS add progress: {end}/{len(embeddings)}"
                logger.info(msg)
                if show_tqdm:
                    print(msg, flush=True)
            if progress is not None:
                progress.close()

            index_dir = out_root / _sanitize_model_name(model_name) / _sanitize_variant_name(variant)
            index_dir.mkdir(parents=True, exist_ok=True)

            if use_gpu:
                index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index, str(index_dir / "paragraphs.faiss"))

            # Save a copy of ids for convenience
            with (index_dir / "paragraph_ids.json").open("w", encoding="utf-8") as f:
                json.dump(json.loads(ids_path.read_text(encoding="utf-8")), f, indent=2)

            manifest = {
                "model_name": model_name,
                "text_variant": variant,
                "metric": metric,
            }
            with (index_dir / "manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"Saved FAISS index for {model_name}/{variant} to {index_dir}")
            output_paths.append(index_dir)

    return output_paths
