#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    span_rows = _read_csv(PROJECT_ROOT / "outputs" / "span_id" / "all_experiments.csv")
    retrieval_rows = _read_csv(PROJECT_ROOT / "outputs" / "rerank" / "article_retrieval_results.csv")
    linking_rows = []
    for path in (PROJECT_ROOT / "outputs" / "linking_pipeline").glob("*/evaluation_summary.csv"):
        linking_rows.extend(_read_csv(path))

    report_csv = out_dir / "research_summary.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "section",
            "domain",
            "experiment_name",
            "model",
            "variant",
            "metric",
            "value",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in span_rows:
            writer.writerow({
                "section": "span_id",
                "domain": row.get("domain"),
                "experiment_name": row.get("experiment_name"),
                "model": row.get("model"),
                "variant": row.get("level"),
                "metric": "eval_f1",
                "value": row.get("eval_f1"),
            })

        for row in retrieval_rows:
            writer.writerow({
                "section": row.get("task"),
                "domain": row.get("domain"),
                "experiment_name": row.get("output_dir"),
                "model": row.get("model"),
                "variant": row.get("variant"),
                "metric": "recall@1",
                "value": row.get("recall@1"),
            })

        for row in linking_rows:
            writer.writerow({
                "section": "linking_pipeline",
                "domain": row.get("domain"),
                "experiment_name": f"{row.get('level')}__{row.get('variant')}",
                "model": "",
                "variant": row.get("metric"),
                "metric": "f1",
                "value": row.get("f1"),
            })

    report_md = out_dir / "research_summary.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Research Summary\n\n")
        f.write(f"- Span-ID experiments: {len(span_rows)}\n")
        f.write(f"- Retrieval/rerank results: {len(retrieval_rows)}\n")
        f.write(f"- Linking eval rows: {len(linking_rows)}\n")
        f.write("\nSee research_summary.csv for details.\n")

    print(f"Wrote {report_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
