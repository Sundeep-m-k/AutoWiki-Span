from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def prepare_retrieval_data(cfg: Dict[str, Any]) -> None:
    exp = cfg.get("experiment", {})
    retrieval = cfg.get("retrieval", {})
    data_cfg = retrieval.get("data", {})

    data_dir = Path(data_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    articles_src = Path(data_cfg["articles_src"])
    articles_dst = data_dir / "articles.jsonl"

    overwrite = bool(data_cfg.get("overwrite", False))

    if not articles_dst.exists() or overwrite:
        if not articles_src.is_file():
            raise FileNotFoundError(f"articles_src not found: {articles_src}")
        articles_dst.write_text(articles_src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {articles_dst}")
    else:
        print(f"Skipping articles.jsonl (exists): {articles_dst}")

    queries_src = Path(data_cfg["queries_src"])
    if not queries_src.is_file():
        raise FileNotFoundError(f"queries_src not found: {queries_src}")

    queries = list(_read_jsonl(queries_src))
    if not queries:
        raise ValueError(f"No queries found in {queries_src}")

    splits = data_cfg.get("splits", {})
    train_ratio = float(splits.get("train", 0.8))
    val_ratio = float(splits.get("val", 0.1))
    test_ratio = splits.get("test")
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    test_ratio = float(test_ratio)

    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    if (train_ratio + val_ratio + test_ratio) <= 0:
        raise ValueError("Split ratios sum to zero")

    seed = int(splits.get("seed", exp.get("seed", 0)))
    if splits.get("shuffle", True):
        rng = random.Random(seed)
        rng.shuffle(queries)

    n_total = len(queries)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train = queries[:n_train]
    val = queries[n_train:n_train + n_val]
    test = queries[n_train + n_val:]

    def _write(split_name: str, rows: Iterable[Dict[str, Any]]) -> None:
        out_path = data_dir / f"queries_{split_name}.jsonl"
        if out_path.exists() and not overwrite:
            print(f"Skipping {out_path} (exists)")
            return
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                if "article_id" not in row:
                    if "target_article_id" in row:
                        row = {**row, "article_id": int(row["target_article_id"])}
                    else:
                        raise KeyError("Missing article_id or target_article_id in query")
                else:
                    row = {**row, "article_id": int(row["article_id"])}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}")

    _write("train", train)
    _write("val", val)
    _write("test", test)

    print(
        "Split counts: "
        f"train={len(train)}, val={len(val)}, test={len(test)} "
        f"(total={n_total})"
    )
