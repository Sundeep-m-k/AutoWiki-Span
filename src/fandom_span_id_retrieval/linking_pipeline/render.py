from __future__ import annotations

from typing import List, Dict


def render_html(text: str, pairs: List[Dict[str, object]], base_path: str = "/wiki/") -> str:
    # Assumes pairs are non-overlapping and sorted by start
    sorted_pairs = sorted(pairs, key=lambda x: int(x.get("start", 0)))
    out = []
    last = 0
    for pair in sorted_pairs:
        start = int(pair.get("start", 0))
        end = int(pair.get("end", 0))
        target = pair.get("target_page_name") or pair.get("target_article_id")
        out.append(text[last:start])
        anchor = text[start:end]
        out.append(f"<a href=\"{base_path}{target}\">{anchor}</a>")
        last = end
    out.append(text[last:])
    return "".join(out)
