# src/fandom_span_id_retrieval/scraping/scrape_pipeline.py
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit, unquote, urlparse

import requests
from bs4 import BeautifulSoup
import yaml

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.utils.logging_utils import create_logger
from fandom_span_id_retrieval.utils.stats_utils import update_scraping_stats

logger = logging.getLogger()


# ---------------------- CONFIG ----------------------


@dataclass
class ScrapingConfig:
    base_url: str
    start_url: Optional[str]
    category_urls: List[str]

    @property
    def domain(self) -> str:
        return urlparse(self.base_url).netloc.split(".")[0]


def load_scraping_config(config_path: Optional[Path] = None) -> ScrapingConfig:
    """
    Load configs/scraping.yaml from project root unless overridden.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "scraping.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "base_url" not in cfg:
        raise ValueError("'base_url' must be defined in configs/scraping.yaml")

    return ScrapingConfig(
        base_url=cfg["base_url"].rstrip("/"),
        start_url=cfg.get("start_url"),
        category_urls=(cfg.get("category_urls") or []),
    )


# ---------------------- URL GENERATION ----------------------


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    path = unquote(parts.path).replace(" ", "_")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def filter_article(url: str) -> bool:
    if "/wiki/" not in url:
        return False
    page = url.split("/wiki/", 1)[1]
    skip_prefixes = (
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
    return not any(page.startswith(p) for p in skip_prefixes)


def scrape_allpages_api(base_url: str, session: requests.Session, delay: float = 0.1) -> List[str]:
    api_url = urljoin(base_url, "/api.php")
    results: List[str] = []
    seen = set()
    params: Dict[str, str] = {
        "action": "query",
        "list": "allpages",
        "aplimit": "500",
        "format": "json",
    }

    while True:
        logger.info(f"[AllPages API] Fetching {api_url} with params={params}")
        r = session.get(api_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for page in (data.get("query", {}).get("allpages", []) or []):
            title = page.get("title")
            if not title:
                continue
            href = f"/wiki/{title.replace(' ', '_')}"
            full = normalize_url(urljoin(base_url, href))
            if full not in seen:
                seen.add(full)
                results.append(full)

        cont = data.get("continue", {}).get("apcontinue")
        if not cont:
            break
        params["apcontinue"] = cont
        time.sleep(delay)

    logger.info(f"[AllPages API] Collected {len(results)} raw URLs")
    return results


def scrape_allpages(base_url: str, start_url: str, session: requests.Session, delay: float = 0.5) -> List[str]:
    url = start_url
    results: List[str] = []
    seen = set()

    while url:
        logger.info(f"[AllPages] Fetching {url}")
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 403:
                logger.warning("[AllPages] 403 Forbidden, switching to API fallback.")
                return scrape_allpages_api(base_url, session)
            raise
        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.select(".mw-allpages-chunk li > a, .mw-allpages-group li > a"):
            href = a.get("href")
            if not href or not href.startswith("/wiki/"):
                continue
            full = normalize_url(urljoin(base_url, href))
            if full not in seen:
                seen.add(full)
                results.append(full)

        next_url = None

        head_next = soup.find("link", rel=lambda v: v and "next" in v.lower() if v else False)
        if head_next and head_next.get("href"):
            next_url = head_next.get("href")

        if not next_url:
            a_next = soup.select_one("a.mw-nextlink")
            if a_next and a_next.get("href"):
                next_url = a_next.get("href")

        if not next_url:
            for a in soup.select(".mw-allpages-nav a[href]"):
                if a.get_text(strip=True).lower().startswith("next page"):
                    next_url = a.get("href")
                    break

        url = urljoin(base_url, next_url) if next_url else None
        time.sleep(delay)

    logger.info(f"[AllPages] Collected {len(results)} raw URLs")
    return results


def scrape_category(base_url: str, category_url: str, session: requests.Session) -> List[str]:
    logger.info(f"[Category] Fetching {category_url}")
    r = session.get(category_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    urls: List[str] = []
    selectors = [
        "a.category-page__member-link",
        ".category-page__members a",
        ".category-page__content a",
    ]

    for sel in selectors:
        for a in soup.select(sel):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                full = normalize_url(base_url.rstrip("/") + href)
            else:
                full = normalize_url(href)
            urls.append(full)

    logger.info(f"[Category] Found {len(urls)} URLs in {category_url}")
    return urls


def scrape_all_categories(base_url: str, category_urls: List[str], session: requests.Session) -> List[str]:
    if not category_urls:
        logger.info("No category_urls in scraping.yaml — skipping category scrape.")
        return []

    collected: List[str] = []
    for cu in category_urls:
        try:
            collected.extend(scrape_category(base_url, cu, session))
        except Exception as e:
            logger.warning(f"Error in category {cu}: {e}", exc_info=True)

    logger.info(f"[Category] Total raw category URLs: {len(collected)}")
    return collected


def build_url_list(cfg: ScrapingConfig, project_root: Path) -> Path:
    base_url = cfg.base_url
    start_url = cfg.start_url
    category_urls = cfg.category_urls

    logger.info(f"Base URL: {base_url}")
    logger.info(f"Start URL (AllPages): {start_url}")
    logger.info(f"Category URLs: {category_urls}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    combined: List[str] = []

    if start_url:
        combined.extend(scrape_allpages(base_url, start_url, session))
    else:
        logger.info("start_url not set in scraping.yaml — skipping AllPages.")

    combined.extend(scrape_all_categories(base_url, category_urls, session))
    combined = list(set(combined))
    logger.info(f"Combined (before filtering): {len(combined)}")

    filtered = [u for u in combined if filter_article(u)]
    logger.info(f"After filtering to real article pages: {len(filtered)}")

    for u in filtered[:5]:
        logger.info(f"Sample URL: {u}")

    url_list_dir = project_root / "data" / "raw" / "url_lists"
    url_list_dir.mkdir(parents=True, exist_ok=True)
    output_path = url_list_dir / f"{cfg.domain}_urls.txt"

    output_path.write_text("\n".join(sorted(filtered)), encoding="utf-8")
    logger.info(f"Saved URL list to: {output_path}")

    return output_path


# ---------------------- PAGE SCRAPING ----------------------


def safe_filename_from_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.netloc.replace(".", "_")
    path = unquote(parts.path).strip("/").replace("/", "_").replace(" ", "_")

    base = f"{host}_{path}" if path else host
    base = base[:180]
    base = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
    if not base:
        base = "page"
    return base + ".html"


def read_url_list(path: Path) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def extract_plain_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    content = soup.find("div", class_="mw-parser-output")
    if content:
        text = content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join([line for line in lines if line])
    return cleaned


def fetch_html(session: requests.Session, url: str, timeout: int = 20) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_article_id(html: str) -> Optional[str]:
    m = re.search(r'wgArticleId"\s*:\s*(\d+)', html)
    if m:
        return m.group(1)
    m = re.search(r"\bwgArticleId\s*=\s*(\d+)", html)
    if m:
        return m.group(1)

    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", {"property": "mw:pageId"})
    if meta and meta.get("content") and str(meta["content"]).isdigit():
        return str(meta["content"])

    meta2 = soup.find("meta", {"name": "pageId"})
    if meta2 and meta2.get("content") and str(meta2["content"]).isdigit():
        return str(meta2["content"])

    return None


def scrape_pages(cfg: ScrapingConfig, project_root: Path) -> None:
    base_url = cfg.base_url
    domain = cfg.domain

    url_list_path = project_root / "data" / "raw" / "url_lists" / f"{domain}_urls.txt"
    output_dir = project_root / "data" / "raw" / "fandom_html" / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    if not url_list_path.exists():
        logger.error(f"URL list file not found: {url_list_path}")
        raise FileNotFoundError(url_list_path)

    urls = read_url_list(url_list_path)
    logger.info(f"Loaded {len(urls)} URLs from {url_list_path}")
    logger.info(f"Output directory: {output_dir}")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "FandomHTMLScraper/1.0 (+research; contact=you@example.com)"}
    )

    total = len(urls)
    success = skipped = failed = 0

    total_html_bytes = 0
    total_text_bytes = 0
    max_html_bytes = 0
    max_text_bytes = 0

    for i, url in enumerate(urls, start=1):
        logger.info(f"[{i}/{total}] Fetching → {url}")
        ok = False

        for attempt in range(1, 4):
            try:
                html = fetch_html(session, url)

                article_id = extract_article_id(html)
                if article_id:
                    fname = f"{article_id}.html"
                else:
                    fname = safe_filename_from_url(url)
                    logger.warning(
                        f"    Could not extract article ID; using URL filename: {fname}"
                    )

                out_path = output_dir / fname

                if out_path.exists():
                    logger.info(f"    SKIP (exists) → {out_path.name}")
                    skipped += 1
                    ok = True
                    break

                # HTML
                html_bytes = html.encode("utf-8")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(html)
                logger.info(f"    Saved HTML → {out_path}")

                total_html_bytes += len(html_bytes)
                max_html_bytes = max(max_html_bytes, len(html_bytes))

                # TEXT
                plain_text = extract_plain_text(html)
                txt_path = out_path.with_suffix(".txt")
                text_bytes = plain_text.encode("utf-8")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(plain_text)
                logger.info(f"    Saved TEXT → {txt_path}")

                total_text_bytes += len(text_bytes)
                max_text_bytes = max(max_text_bytes, len(text_bytes))

                success += 1
                ok = True
                break

            except Exception as e:
                logger.warning(
                    f"    Attempt {attempt}/3 failed for {url}: {e}"
                )
                time.sleep(1.0 * attempt)

        if not ok:
            logger.error(f"    FAILED: {url}")
            failed += 1

        # milestone progress every 50 articles
        if i % 50 == 0:
            logger.info(
                f"[Progress] Processed {i}/{total} pages "
                f"(success={success}, skipped={skipped}, failed={failed})"
            )

        time.sleep(0.5)

    logger.info("=== Scrape summary ===")
    logger.info(f"Total URLs:    {total}")
    logger.info(f"Downloaded:    {success}")
    logger.info(f"Skipped (had): {skipped}")
    logger.info(f"Failed:        {failed}")

    avg_html_bytes = total_html_bytes / success if success > 0 else 0
    avg_text_bytes = total_text_bytes / success if success > 0 else 0

    scraping_stats = {
        "total_urls": total,
        "downloaded": success,
        "skipped": skipped,
        "failed": failed,
        "html_bytes": {
            "total": total_html_bytes,
            "avg": avg_html_bytes,
            "max": max_html_bytes,
        },
        "text_bytes": {
            "total": total_text_bytes,
            "avg": avg_text_bytes,
            "max": max_text_bytes,
        },
    }

    logger.info(f"Scraping stats: {scraping_stats}")

    # write into stats/<domain>.json
    update_scraping_stats(domain, scraping_stats)

# ---------------------- PUBLIC ENTRYPOINT ----------------------


def run_full_scrape(config_path: Optional[Path] = None) -> Path:
    """
    High-level: load config, build URL list, then scrape pages.
    Returns path to URL list.
    """
    cfg = load_scraping_config(config_path)
    # logger is configured in the calling script via create_logger
    logger.info(f"Starting full scrape for domain={cfg.domain}")
    url_list_path = build_url_list(cfg, PROJECT_ROOT)
    scrape_pages(cfg, PROJECT_ROOT)
    logger.info("Finished full scrape.")
    return url_list_path