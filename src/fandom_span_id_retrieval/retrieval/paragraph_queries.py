from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from fandom_span_id_retrieval.retrieval.paragraph_embeddings import expand_config
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _read_paragraphs(csv_path: Path, id_field: str, text_field: str, title_field: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get(id_field, "")
            if not pid:
                continue
            rows[pid] = {
                "text": row.get(text_field, ""),
                "title": row.get(title_field, ""),
                "article_id": row.get("article_id", ""),
            }
    return rows


def _read_links(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _domain_label(domain: str) -> str:
    return domain.replace("-", " ").title()


def _build_context(text: str, anchor: str, start: int, end: int, window: int) -> str:
    if not text:
        return anchor
    n = len(text)
    start = max(0, min(start, n))
    end = max(0, min(end, n))
    win_start = max(0, start - window)
    win_end = min(n, end + window)
    snippet = text[win_start:win_end]

    rel_start = start - win_start
    rel_end = end - win_start
    if 0 <= rel_start <= rel_end <= len(snippet):
        return snippet[:rel_start] + "[ANCHOR] " + anchor + snippet[rel_end:]

    if anchor and anchor in snippet:
        return snippet.replace(anchor, "[ANCHOR] " + anchor, 1)

    return snippet


def _templates(domain_label: str) -> List[str]:
    return [
        "{anchor}",
        "Who is {anchor}?",
        "What is {anchor}?",
        "Tell me about {anchor}.",
        "Information about {anchor}",
        "{anchor} wiki",
        "{anchor} character",
        "{anchor} in {domain}",
        "Background on {anchor}",
        "Profile of {anchor}",
        "{context}",
        "{context} [ANCHOR] {anchor}",
        "{anchor} appears in: {context}",
        "Context: {context}",
        "From the paragraph: {context}",
        "Question about {anchor}: {context}",
        "In the paragraph, {anchor} is mentioned: {context}",
        "Find article for {anchor} given: {context}",
        "{anchor} — {context}",
        "Which article matches {anchor} in: {context}",
    ]


def build_paragraph_queries(cfg: Dict[str, Any]) -> Path:
    cfg = expand_config(cfg)
    exp = cfg.get("experiment", {})
    para_cfg = cfg.get("paragraphs", {})
    query_cfg = cfg.get("queries", {})

    csv_path = Path(para_cfg["csv_path"])
    links_path = Path(query_cfg["links_csv"])
    if not csv_path.is_file():
        raise FileNotFoundError(f"paragraph CSV not found: {csv_path}")
    if not links_path.is_file():
        raise FileNotFoundError(f"links CSV not found: {links_path}")

    output_path = Path(query_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = output_path.parent / "logs"
    logger, _ = create_logger(log_dir, script_name="paragraph_queries")
    logger.info(f"Paragraph CSV: {csv_path}")
    logger.info(f"Links CSV: {links_path}")

    paragraphs = _read_paragraphs(
        csv_path=csv_path,
        id_field=para_cfg.get("id_field", "paragraph_id"),
        text_field=para_cfg.get("text_field", "paragraph_text"),
        title_field=para_cfg.get("title_field", "title"),
    )

    domain_label = _domain_label(str(exp.get("domain", "domain")))
    templates = _templates(domain_label)
    variants = int(query_cfg.get("variants", 20))
    window = int(query_cfg.get("context_window", 200))

    internal_links: List[Dict[str, str]] = []
    for row in _read_links(links_path):
        if row.get("link_type") != "internal":
            continue
        if not row.get("article_id_of_internal_link"):
            continue
        if row.get("paragraph_id") not in paragraphs:
            continue
        internal_links.append(row)

    logger.info(f"Internal links: {len(internal_links)}")

    max_queries = int(query_cfg.get("max_queries", 1000))
    seed = int(query_cfg.get("sample_seed", exp.get("seed", 0)))
    rng = random.Random(seed)
    rng.shuffle(internal_links)
    sampled_links = internal_links[:max_queries]

    out_rows: List[Dict[str, Any]] = []
    query_id = 0
    for link in sampled_links:
        pid = link["paragraph_id"]
        anchor = link.get("anchor_text", "")
        target_article_id = link.get("article_id_of_internal_link")
        para = paragraphs[pid]

        text = para.get("text", "")
        title = para.get("title", "")
        rel_start = int(link.get("link_rel_start") or 0)
        rel_end = int(link.get("link_rel_end") or rel_start)
        context = _build_context(text, anchor, rel_start, rel_end, window)

        filled = []
        for template in templates:
            filled.append(
                template.format(
                    anchor=anchor,
                    context=context,
                    title=title,
                    domain=domain_label,
                )
            )

        if len(filled) < variants:
            filled.extend([anchor] * (variants - len(filled)))

        for variant_id, query_text in enumerate(filled[:variants]):
            out_rows.append({
                "query_id": query_id,
                "query_text": query_text,
                "variant_id": variant_id,
                "anchor_text": anchor,
                "source_paragraph_id": pid,
                "target_article_id": int(target_article_id),
            })
            query_id += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(out_rows)} queries to {output_path}")
    return output_path
