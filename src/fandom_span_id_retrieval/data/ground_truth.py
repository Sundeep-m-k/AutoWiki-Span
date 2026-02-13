from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, unquote

from bs4 import BeautifulSoup, NavigableString, Tag

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.utils.stats_utils import load_stats, save_stats


SKIP_NAMESPACES = (
    "File:",
    "User:",
    "User_blog:",
    "Template:",
    "Help:",
    "Special:",
    "Forum:",
    "Message_Wall:",
    "Category:",
    "Talk:",
    "MediaWiki:",
    "Module:",
)


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _append(buf: str, s: str) -> str:
    s = _norm_ws(s)
    if not s:
        return buf
    if not buf:
        return s
    if not buf.endswith((" ", "\n")):
        return buf + " " + s
    return buf + s


# ---------- metadata helpers ----------

def get_article_id(soup: BeautifulSoup) -> Optional[int]:
    meta = soup.find("meta", {"property": "mw:pageId"})
    if meta and meta.get("content") and str(meta["content"]).isdigit():
        return int(meta["content"])

    meta2 = soup.find("meta", {"name": "pageId"})
    if meta2 and meta2.get("content") and str(meta2["content"]).isdigit():
        return int(meta2["content"])

    wg_article_id_re = re.compile(r'"wgArticleId"\s*:\s*(\d+)')
    for script in soup.find_all("script"):
        if not script.string:
            continue
        m = wg_article_id_re.search(script.string)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def get_page_name(soup: BeautifulSoup) -> Optional[str]:
    wg_pagename_re = re.compile(r'"wgPageName"\s*:\s*"([^"]+)"')
    for script in soup.find_all("script"):
        if not script.string:
            continue
        m = wg_pagename_re.search(script.string)
        if m:
            return m.group(1)
    return None


def get_title_from_html(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.select_one("h1.page-header__title")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    h1 = soup.select_one("h1#firstHeading") or soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return None


def get_main_content_div(soup: BeautifulSoup) -> Tag:
    for sel in [
        "#mw-content-text .mw-parser-output",
        "div.mw-parser-output",
        "div.page-content div.mw-parser-output",
        "#mw-content-text",
        "div.page-content",
        "main",
        "article",
    ]:
        node = soup.select_one(sel)
        if node:
            return node
    return soup.body or soup


def build_page_map(html_dir: Path, logger) -> Dict[str, int]:
    """
    Build mapping from page_name -> article_id by scanning HTML once.
    """
    page_map: Dict[str, int] = {}
    html_files = sorted(list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")))
    logger.info(f"Building page_map from {len(html_files)} HTML files")

    for i, path in enumerate(html_files, start=1):
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"[page_map {i}/{len(html_files)}] Failed to read {path}: {e}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        page_name = get_page_name(soup)
        article_id = get_article_id(soup)

        if page_name is not None and article_id is not None:
            page_map[page_name] = article_id

        if i % 200 == 0:
            logger.info(f"[page_map] Processed {i}/{len(html_files)} files")

    logger.info(f"[page_map] Built mapping for {len(page_map)} pages")
    return page_map


# ---------- link classification ----------

def classify_link(href: Optional[str], base_url: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Return (link_type, target_page_name, resolved_url)
    link_type in: internal / external / category / file / other
    """
    if not href:
        return "other", None, None

    href = href.strip()
    base_url = base_url.rstrip("/")

    if href.startswith("#"):
        return "other", None, None

    if href.startswith("http://") or href.startswith("https://"):
        if href.startswith(base_url + "/wiki/"):
            path = href[len(base_url):]
            if path.startswith("/wiki/"):
                page = unquote(path[len("/wiki/"):]).replace(" ", "_")
                if any(page.startswith(ns) for ns in SKIP_NAMESPACES):
                    ns = page.split(":", 1)[0] + ":"
                    if ns.lower() == "file:":
                        return "file", None, href
                    if ns.lower() == "category:":
                        return "category", None, href
                    return "other", None, href
                return "internal", page, href
            return "other", None, href
        return "external", None, href

    if href.startswith("/wiki/"):
        page = unquote(href[len("/wiki/"):]).replace(" ", "_")
        resolved = urljoin(base_url, href)
        if any(page.startswith(ns) for ns in SKIP_NAMESPACES):
            ns = page.split(":", 1)[0] + ":"
            if ns.lower() == "file:":
                return "file", None, resolved
            if ns.lower() == "category:":
                return "category", None, resolved
            return "other", None, resolved
        return "internal", page, resolved

    resolved = urljoin(base_url, href)
    return "other", None, resolved


# ---------- section-level text + absolute link offsets ----------

def build_section_text_and_links(
    content_div: Tag,
    base_url: str,
    page_map: Dict[str, int],
) -> Tuple[str, List[Dict]]:
    """
    Build a single text buffer from content_div and absolute link offsets.
    """
    text = ""
    links: List[Dict] = []

    for node in content_div.descendants:
        if isinstance(node, NavigableString):
            if node.parent and isinstance(node.parent, Tag) and node.parent.name == "a":
                continue
            text = _append(text, str(node))
            continue

        if isinstance(node, Tag) and node.name == "a":
            anchor = node.get_text(" ", strip=True)
            if not anchor:
                continue
            href = node.get("href")
            link_type, target_page_name, resolved_url = classify_link(href, base_url)

            start = len(text)
            text = _append(text, anchor)
            end = len(text)

            target_article_id = None
            if link_type == "internal" and target_page_name:
                target_article_id = page_map.get(target_page_name)

            links.append(
                {
                    "anchor_text": anchor,
                    "link_type": link_type,
                    "target_page_name": target_page_name,
                    "resolved_url": resolved_url,
                    "start": start,
                    "end": end,
                    "article_id_of_internal_link": target_article_id,
                }
            )

    return text, links


# ---------- span builders (paragraph and sentence) ----------

_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_NL_BOUNDARY_RE = re.compile(r"\n+")


def build_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return sentence spans (start, end) over original text.
    """
    if not text:
        return []

    cuts = [0]
    for m in _SENT_BOUNDARY_RE.finditer(text):
        cuts.append(m.end())
    cuts.append(len(text))

    spans: List[Tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if text[a:b].strip():
            spans.append((a, b))

    if spans:
        return spans

    cuts = [0]
    for m in _NL_BOUNDARY_RE.finditer(text):
        cuts.append(m.end())
    cuts.append(len(text))
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        if text[a:b].strip():
            spans.append((a, b))

    return spans


def build_paragraph_spans(
    text: str,
    max_chars: int = 512,
    max_sentences: int = 4,
    min_chars: int = 30,
    hard_max_chars: int = 1024,
) -> List[Tuple[int, int]]:
    """
    Build paragraph spans as (start, end) ranges over text, combining sentences.
    """
    if not text or not text.strip():
        return []

    sent_spans = build_sentence_spans(text)
    if not sent_spans:
        return []

    spans: List[Tuple[int, int]] = []
    cur_start = None
    cur_end = None
    cur_sent_count = 0

    for s_start, s_end in sent_spans:
        if cur_start is None:
            cur_start, cur_end, cur_sent_count = s_start, s_end, 1
            continue

        tentative_len = s_end - cur_start
        tentative_sent_count = cur_sent_count + 1

        if tentative_len > max_chars or tentative_sent_count > max_sentences:
            span_text = text[cur_start:cur_end]
            if len(span_text) >= min_chars and span_text.strip():
                spans.append((cur_start, cur_end))
            cur_start, cur_end, cur_sent_count = s_start, s_end, 1
        else:
            cur_end = s_end
            cur_sent_count = tentative_sent_count

    if cur_start is not None and cur_end is not None:
        span_text = text[cur_start:cur_end]
        if len(span_text) >= min_chars and span_text.strip():
            spans.append((cur_start, cur_end))

    final_spans: List[Tuple[int, int]] = []
    for s_start, s_end in spans:
        span_text = text[s_start:s_end]
        if len(span_text) <= hard_max_chars:
            final_spans.append((s_start, s_end))
            continue

        offset = 0
        while offset < len(span_text):
            chunk = span_text[offset: offset + hard_max_chars]
            if not chunk.strip():
                break
            chunk_start = s_start + offset
            chunk_end = chunk_start + len(chunk)
            if len(chunk) >= min_chars:
                final_spans.append((chunk_start, chunk_end))
            offset += hard_max_chars

    return final_spans or spans


# ---------- main builder ----------

def build_ground_truth_for_domain(domain: str, logger) -> Path:
    """
    HTML -> paragraph-level and sentence-level ground truth with hyperlink spans.
    """
    html_dir = PROJECT_ROOT / "data" / "raw" / "fandom_html" / domain
    processed_dir = PROJECT_ROOT / "data" / "processed" / domain
    processed_dir.mkdir(parents=True, exist_ok=True)

    paragraphs_master = processed_dir / f"paragraphs_{domain}.jsonl"
    sentences_master = processed_dir / f"sentences_{domain}.jsonl"
    pages_master = processed_dir / f"pages_{domain}.jsonl"
    articles_master = processed_dir / f"articles_{domain}.jsonl"
    paragraphs_by_article_dir = processed_dir / f"paragraphs_{domain}_by_article"
    sentences_by_article_dir = processed_dir / f"sentences_{domain}_by_article"
    paragraphs_by_article_dir.mkdir(parents=True, exist_ok=True)
    sentences_by_article_dir.mkdir(parents=True, exist_ok=True)

    paragraphs_csv = processed_dir / f"paragraphs_{domain}.csv"
    paragraph_links_csv = processed_dir / f"paragraph_links_{domain}.csv"
    sentences_csv = processed_dir / f"sentences_{domain}.csv"
    sentence_links_csv = processed_dir / f"sentence_links_{domain}.csv"

    if not html_dir.exists():
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")

    html_files = sorted(list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")))
    if not html_files:
        raise FileNotFoundError(f"No .html/.htm files found in: {html_dir}")

    logger.info(f"Building ground truth for domain={domain}")
    logger.info(f"HTML dir: {html_dir}")
    logger.info(f"Paragraph master: {paragraphs_master}")
    logger.info(f"Sentence master:  {sentences_master}")
    logger.info(f"Articles master:  {articles_master}")
    logger.info(f"Per-article paragraphs dir: {paragraphs_by_article_dir}")
    logger.info(f"Per-article sentences dir:  {sentences_by_article_dir}")

    page_map = build_page_map(html_dir, logger)

    para_fields = [
        "paragraph_id", "article_id", "page_name", "title",
        "paragraph_index", "paragraph_text", "url", "source_path",
    ]
    para_link_fields = [
        "paragraph_id", "anchor_index", "anchor_text", "link_type",
        "link_rel_start", "link_rel_end", "target_page_name",
        "article_id_of_internal_link", "resolved_url",
    ]
    sent_fields = [
        "sentence_id", "article_id", "page_name", "title",
        "sentence_index", "sentence_text", "url", "source_path",
    ]
    sent_link_fields = [
        "sentence_id", "anchor_index", "anchor_text", "link_type",
        "link_rel_start", "link_rel_end", "target_page_name",
        "article_id_of_internal_link", "resolved_url",
    ]

    para_csv_f = open(paragraphs_csv, "w", encoding="utf-8", newline="")
    para_links_f = open(paragraph_links_csv, "w", encoding="utf-8", newline="")
    sent_csv_f = open(sentences_csv, "w", encoding="utf-8", newline="")
    sent_links_f = open(sentence_links_csv, "w", encoding="utf-8", newline="")

    para_writer = csv.DictWriter(para_csv_f, fieldnames=para_fields)
    para_links_writer = csv.DictWriter(para_links_f, fieldnames=para_link_fields)
    sent_writer = csv.DictWriter(sent_csv_f, fieldnames=sent_fields)
    sent_links_writer = csv.DictWriter(sent_links_f, fieldnames=sent_link_fields)

    para_writer.writeheader()
    para_links_writer.writeheader()
    sent_writer.writeheader()
    sent_links_writer.writeheader()

    num_articles = 0
    num_paragraphs = 0
    num_sentences = 0
    num_pages = 0
    num_links = 0

    link_type_counts = {
        "internal": 0,
        "external": 0,
        "category": 0,
        "file": 0,
        "other": 0,
    }

    def safe_key(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

    with open(paragraphs_master, "w", encoding="utf-8") as f_para_master, \
        open(sentences_master, "w", encoding="utf-8") as f_sent_master, \
        open(pages_master, "w", encoding="utf-8") as f_page_master, \
        open(articles_master, "w", encoding="utf-8") as f_articles_master:

        paragraph_counter = 0
        sentence_counter = 0
        page_counter = 0

        for i, path in enumerate(html_files, start=1):
            try:
                html = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"[{i}/{len(html_files)}] Failed to read {path}: {e}")
                continue

            soup = BeautifulSoup(html, "html.parser")

            article_id = get_article_id(soup)
            page_name = get_page_name(soup)
            title = get_title_from_html(soup)
            if not title and page_name:
                title = page_name.replace("_", " ")

            content_div = get_main_content_div(soup)
            if not content_div:
                logger.warning(f"[{i}/{len(html_files)}] No content div for {path}")
                continue

            base_url = f"https://{domain}.fandom.com"
            section_text, links_abs = build_section_text_and_links(
                content_div, base_url=base_url, page_map=page_map
            )
            if not section_text.strip():
                logger.info(f"[{i}/{len(html_files)}] Empty section text for {path}")
                continue

            # update link type counts once per absolute link
            for lk in links_abs:
                lt = lk["link_type"]
                if lt in link_type_counts:
                    link_type_counts[lt] += 1
                else:
                    link_type_counts["other"] += 1

            article_url = None
            if page_name:
                article_url = base_url.rstrip("/") + "/wiki/" + page_name

            num_articles += 1

            if article_id is not None:
                article_key = str(article_id)
            elif page_name:
                article_key = page_name
            else:
                article_key = path.stem
            article_key_safe = safe_key(article_key)

            para_article_path = paragraphs_by_article_dir / f"{article_key_safe}.jsonl"
            sent_article_path = sentences_by_article_dir / f"{article_key_safe}.jsonl"

            page_counter += 1
            page_id = f"{domain}_page_{page_counter:07d}"
            page_links: List[Dict] = []
            for lk in links_abs:
                page_links.append(
                    {
                        "anchor_index": len(page_links) + 1,
                        "anchor_text": lk["anchor_text"],
                        "link_type": lk["link_type"],
                        "link_rel_start": lk["start"],
                        "link_rel_end": lk["end"],
                        "target_page_name": lk["target_page_name"],
                        "article_id_of_internal_link": lk["article_id_of_internal_link"],
                        "resolved_url": lk["resolved_url"],
                    }
                )

            page_record = {
                "granularity": "page",
                "article_id": article_id,
                "page_name": page_name,
                "title": title,
                "page_id": page_id,
                "page_text": section_text,
                "url": article_url,
                "source_path": str(path.relative_to(PROJECT_ROOT)),
                "links": page_links,
            }

            f_page_master.write(json.dumps(page_record, ensure_ascii=False) + "\n")
            num_pages += 1

            para_spans = build_paragraph_spans(section_text)
            sent_spans = build_sentence_spans(section_text)

            lead_paragraph = ""
            if para_spans:
                lead_start, lead_end = para_spans[0]
                lead_paragraph = section_text[lead_start:lead_end].strip()

            article_record = {
                "article_id": article_id,
                "title": page_name or title,
                "lead_paragraph": lead_paragraph,
                "full_text": section_text,
            }
            f_articles_master.write(json.dumps(article_record, ensure_ascii=False) + "\n")

            with open(para_article_path, "w", encoding="utf-8") as f_para_article:
                for p_idx, (p_start, p_end) in enumerate(para_spans):
                    paragraph_counter += 1
                    paragraph_id = f"{domain}_paragraph_{paragraph_counter:07d}"
                    para_text = section_text[p_start:p_end]

                    para_links: List[Dict] = []
                    for lk in links_abs:
                        l_start = lk["start"]
                        l_end = lk["end"]
                        if l_start < p_start or l_start >= p_end:
                            continue
                        if l_end > p_end:
                            l_end = p_end
                        para_links.append(
                            {
                                "anchor_index": len(para_links) + 1,
                                "anchor_text": lk["anchor_text"],
                                "link_type": lk["link_type"],
                                "link_rel_start": l_start - p_start,
                                "link_rel_end": l_end - p_start,
                                "target_page_name": lk["target_page_name"],
                                "article_id_of_internal_link": lk["article_id_of_internal_link"],
                                "resolved_url": lk["resolved_url"],
                            }
                        )

                    record = {
                        "granularity": "paragraph",
                        "article_id": article_id,
                        "page_name": page_name,
                        "title": title,
                        "paragraph_id": paragraph_id,
                        "paragraph_index": p_idx,
                        "paragraph_text": para_text,
                        "url": article_url,
                        "source_path": str(path.relative_to(PROJECT_ROOT)),
                        "links": para_links,
                    }

                    num_paragraphs += 1
                    num_links += len(para_links)

                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    f_para_master.write(line)
                    f_para_article.write(line)

                    para_writer.writerow({
                        "paragraph_id": paragraph_id,
                        "article_id": article_id,
                        "page_name": page_name,
                        "title": title,
                        "paragraph_index": p_idx,
                        "paragraph_text": para_text,
                        "url": article_url,
                        "source_path": str(path.relative_to(PROJECT_ROOT)),
                    })
                    for link in para_links:
                        para_links_writer.writerow({
                            "paragraph_id": paragraph_id,
                            "anchor_index": link["anchor_index"],
                            "anchor_text": link["anchor_text"],
                            "link_type": link["link_type"],
                            "link_rel_start": link["link_rel_start"],
                            "link_rel_end": link["link_rel_end"],
                            "target_page_name": link["target_page_name"],
                            "article_id_of_internal_link": link["article_id_of_internal_link"],
                            "resolved_url": link["resolved_url"],
                        })

            with open(sent_article_path, "w", encoding="utf-8") as f_sent_article:
                for s_idx, (s_start, s_end) in enumerate(sent_spans):
                    sentence_counter += 1
                    sentence_id = f"{domain}_sentence_{sentence_counter:07d}"
                    sent_text = section_text[s_start:s_end]

                    sent_links: List[Dict] = []
                    for lk in links_abs:
                        l_start = lk["start"]
                        l_end = lk["end"]
                        if l_start < s_start or l_start >= s_end:
                            continue
                        if l_end > s_end:
                            l_end = s_end
                        sent_links.append(
                            {
                                "anchor_index": len(sent_links) + 1,
                                "anchor_text": lk["anchor_text"],
                                "link_type": lk["link_type"],
                                "link_rel_start": l_start - s_start,
                                "link_rel_end": l_end - s_start,
                                "target_page_name": lk["target_page_name"],
                                "article_id_of_internal_link": lk["article_id_of_internal_link"],
                                "resolved_url": lk["resolved_url"],
                            }
                        )

                    record = {
                        "granularity": "sentence",
                        "article_id": article_id,
                        "page_name": page_name,
                        "title": title,
                        "sentence_id": sentence_id,
                        "sentence_index": s_idx,
                        "sentence_text": sent_text,
                        "url": article_url,
                        "source_path": str(path.relative_to(PROJECT_ROOT)),
                        "links": sent_links,
                    }

                    num_sentences += 1
                    num_links += len(sent_links)

                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    f_sent_master.write(line)
                    f_sent_article.write(line)

                    sent_writer.writerow({
                        "sentence_id": sentence_id,
                        "article_id": article_id,
                        "page_name": page_name,
                        "title": title,
                        "sentence_index": s_idx,
                        "sentence_text": sent_text,
                        "url": article_url,
                        "source_path": str(path.relative_to(PROJECT_ROOT)),
                    })
                    for link in sent_links:
                        sent_links_writer.writerow({
                            "sentence_id": sentence_id,
                            "anchor_index": link["anchor_index"],
                            "anchor_text": link["anchor_text"],
                            "link_type": link["link_type"],
                            "link_rel_start": link["link_rel_start"],
                            "link_rel_end": link["link_rel_end"],
                            "target_page_name": link["target_page_name"],
                            "article_id_of_internal_link": link["article_id_of_internal_link"],
                            "resolved_url": link["resolved_url"],
                        })

            if i % 50 == 0:
                logger.info(
                    f"[Progress] Processed {i}/{len(html_files)} HTML files; "
                    f"articles={num_articles}, paragraphs={num_paragraphs}, "
                    f"sentences={num_sentences}, links={num_links}"
                )

    para_csv_f.close()
    para_links_f.close()
    sent_csv_f.close()
    sent_links_f.close()

    logger.info("=== Ground truth summary ===")
    logger.info(f"Articles:    {num_articles}")
    logger.info(f"Paragraphs:  {num_paragraphs}")
    logger.info(f"Sentences:   {num_sentences}")
    logger.info(f"Links:       {num_links}")
    logger.info(f"Link type counts: {link_type_counts}")
    logger.info(f"Paragraph master: {paragraphs_master}")
    logger.info(f"Sentence master:  {sentences_master}")
    logger.info(f"Page master:      {pages_master}")
    logger.info(f"Articles master:  {articles_master}")
    logger.info(f"Per-article paragraphs dir: {paragraphs_by_article_dir}")
    logger.info(f"Per-article sentences dir:  {sentences_by_article_dir}")
    logger.info(f"Paragraph CSV: {paragraphs_csv}")
    logger.info(f"Paragraph links CSV: {paragraph_links_csv}")
    logger.info(f"Sentence CSV: {sentences_csv}")
    logger.info(f"Sentence links CSV: {sentence_links_csv}")

    stats = load_stats(domain)
    stats["dataset_stats"] = {
        "num_articles": num_articles,
        "num_paragraphs": num_paragraphs,
        "num_sentences": num_sentences,
        "num_pages": num_pages,
        "num_links": num_links,
        "link_type_counts": link_type_counts,
    }
    save_stats(domain, stats)

    return paragraphs_master
