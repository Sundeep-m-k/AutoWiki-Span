from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fandom_span_id_retrieval.retrieval.paragraph_embeddings import expand_config
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _read_paragraphs(csv_path: Path, id_field: str, text_field: str) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get(id_field, "")
            if not pid:
                continue
            rows[pid] = row.get(text_field, "")
    return rows


def _read_links(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def build_article_queries(cfg: Dict[str, Any]) -> Path | None:
    cfg = expand_config(cfg)
    exp = cfg.get("experiment", {})
    retrieval_cfg = cfg.get("retrieval", {})
    query_cfg = retrieval_cfg.get("queries", {})
    data_cfg = retrieval_cfg.get("data", {})

    if not query_cfg.get("do_build", False):
        return None

    domain = str(exp.get("domain", ""))
    paragraphs_csv = Path(
        query_cfg.get(
            "paragraphs_csv",
            f"data/processed/{domain}/paragraphs_{domain}.csv",
        )
    )
    links_csv = Path(
        query_cfg.get(
            "links_csv",
            f"data/processed/{domain}/paragraph_links_{domain}.csv",
        )
    )
    output_path = Path(
        query_cfg.get(
            "output_path",
            data_cfg.get(
                "queries_src",
                f"data/processed/{domain}/article_queries_{domain}.jsonl",
            ),
        )
    )

    if not paragraphs_csv.is_file():
        raise FileNotFoundError(f"paragraph CSV not found: {paragraphs_csv}")
    if not links_csv.is_file():
        raise FileNotFoundError(f"links CSV not found: {links_csv}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overwrite = bool(query_cfg.get("overwrite", data_cfg.get("overwrite", False)))
    if output_path.exists() and not overwrite:
        print(f"Skipping article queries (exists): {output_path}")
        return output_path

    log_dir = output_path.parent / "logs"
    logger, _ = create_logger(log_dir, script_name="article_queries")
    logger.info(f"Paragraph CSV: {paragraphs_csv}")
    logger.info(f"Links CSV: {links_csv}")
    logger.info(f"Output path: {output_path}")

    paragraphs = _read_paragraphs(
        csv_path=paragraphs_csv,
        id_field=query_cfg.get("paragraph_id_field", "paragraph_id"),
        text_field=query_cfg.get("paragraph_text_field", "paragraph_text"),
    )

    links: List[Dict[str, str]] = list(_read_links(links_csv))
    if query_cfg.get("shuffle", False):
        seed = int(query_cfg.get("seed", exp.get("seed", 0)))
        rng = random.Random(seed)
        rng.shuffle(links)

    max_queries = int(query_cfg.get("max_queries", 0))
    out_rows: List[Dict[str, Any]] = []

    for link in links:
        if link.get("link_type") != "internal":
            continue
        target_id = link.get("article_id_of_internal_link")
        if not target_id:
            continue
        paragraph_id = link.get("paragraph_id", "")
        if paragraph_id not in paragraphs:
            continue
        anchor_text = (link.get("anchor_text") or "").strip()
        if not anchor_text:
            continue

        paragraph_text = (paragraphs.get(paragraph_id) or "").strip()
        if not paragraph_text:
            continue

        query_text = f"{paragraph_text} [ANCHOR] {anchor_text}"
        out_rows.append({
            "query_id": len(out_rows),
            "query_text": query_text,
            "target_article_id": int(target_id),
        })

        if max_queries > 0 and len(out_rows) >= max_queries:
            break

    if not out_rows:
        raise ValueError("No article queries generated; check link and paragraph inputs")

    with output_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(out_rows)} queries to {output_path}")
    return output_path
