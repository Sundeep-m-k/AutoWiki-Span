from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from fandom_span_id_retrieval.retrieval.paragraph_embeddings import _expand_placeholders
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_paragraph_links(csv_path: Path) -> Dict[str, List[Dict[str, object]]]:
    gold: Dict[str, List[Dict[str, object]]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("link_type") != "internal":
                continue
            pid = row.get("paragraph_id")
            if not pid:
                continue
            start = int(row.get("link_rel_start") or 0)
            end = int(row.get("link_rel_end") or 0)
            aid = row.get("article_id_of_internal_link")
            if not aid:
                continue
            gold.setdefault(pid, []).append({
                "start": start,
                "end": end,
                "target_article_id": int(aid),
            })
    return gold


def _read_page_links(jsonl_path: Path) -> Dict[str, List[Dict[str, object]]]:
    gold: Dict[str, List[Dict[str, object]]] = {}
    for rec in _read_jsonl(jsonl_path):
        pid = str(rec.get("page_id", ""))
        if not pid:
            continue
        links = rec.get("links") or []
        for lk in links:
            if lk.get("link_type") != "internal":
                continue
            start = int(lk.get("link_rel_start") or 0)
            end = int(lk.get("link_rel_end") or 0)
            aid = lk.get("article_id_of_internal_link")
            if aid is None:
                continue
            gold.setdefault(pid, []).append({
                "start": start,
                "end": end,
                "target_article_id": int(aid),
            })
    return gold


def _multiset_matches(a: List[Tuple], b: List[Tuple]) -> int:
    ca = Counter(a)
    cb = Counter(b)
    return sum(min(ca[k], cb[k]) for k in ca.keys() & cb.keys())


def _prf(match: int, pred: int, gold: int) -> Dict[str, float]:
    precision = match / pred if pred > 0 else 0.0
    recall = match / gold if gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _collect_error_samples(
    rec_id: str,
    pred_exact: List[Tuple[int, int, int]],
    gold_exact: List[Tuple[int, int, int]],
    max_samples: int,
    fp_samples: List[Dict[str, object]],
    fn_samples: List[Dict[str, object]],
) -> None:
    if max_samples <= 0:
        return

    pred_counter = Counter(pred_exact)
    gold_counter = Counter(gold_exact)

    for key, count in (pred_counter - gold_counter).items():
        if len(fp_samples) >= max_samples:
            break
        start, end, target = key
        fp_samples.append({
            "id": rec_id,
            "start": start,
            "end": end,
            "target_article_id": target,
        })

    for key, count in (gold_counter - pred_counter).items():
        if len(fn_samples) >= max_samples:
            break
        start, end, target = key
        fn_samples.append({
            "id": rec_id,
            "start": start,
            "end": end,
            "target_article_id": target,
        })


def evaluate_predictions(
    pred_path: Path,
    gold_map: Dict[str, List[Dict[str, object]]],
    max_error_samples: int = 0,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]], List[Dict[str, object]]]:
    exact_match = 0
    span_match = 0
    target_match = 0

    pred_exact = gold_exact = 0
    pred_span = gold_span = 0
    pred_target = gold_target = 0

    fp_samples: List[Dict[str, object]] = []
    fn_samples: List[Dict[str, object]] = []

    for rec in _read_jsonl(pred_path):
        rec_id = str(rec.get("id", ""))
        preds = rec.get("predicted_pairs") or []
        gold = gold_map.get(rec_id, [])

        pred_exact_list = [
            (int(p.get("start", 0)), int(p.get("end", 0)), int(p.get("target_article_id", -1)))
            for p in preds
        ]
        gold_exact_list = [
            (int(g.get("start", 0)), int(g.get("end", 0)), int(g.get("target_article_id", -1)))
            for g in gold
        ]

        pred_span_list = [(s, e) for s, e, _ in pred_exact_list]
        gold_span_list = [(s, e) for s, e, _ in gold_exact_list]

        pred_target_list = [(t,) for _, _, t in pred_exact_list]
        gold_target_list = [(t,) for _, _, t in gold_exact_list]

        _collect_error_samples(
            rec_id,
            pred_exact_list,
            gold_exact_list,
            max_error_samples,
            fp_samples,
            fn_samples,
        )

        exact_match += _multiset_matches(pred_exact_list, gold_exact_list)
        span_match += _multiset_matches(pred_span_list, gold_span_list)
        target_match += _multiset_matches(pred_target_list, gold_target_list)

        pred_exact += len(pred_exact_list)
        gold_exact += len(gold_exact_list)
        pred_span += len(pred_span_list)
        gold_span += len(gold_span_list)
        pred_target += len(pred_target_list)
        gold_target += len(gold_target_list)

    return (
        {
        "exact_span_target": {
            **_prf(exact_match, pred_exact, gold_exact),
            "pred_count": pred_exact,
            "gold_count": gold_exact,
            "match_count": exact_match,
        },
        "span_only": {
            **_prf(span_match, pred_span, gold_span),
            "pred_count": pred_span,
            "gold_count": gold_span,
            "match_count": span_match,
        },
        "target_only": {
            **_prf(target_match, pred_target, gold_target),
            "pred_count": pred_target,
            "gold_count": gold_target,
            "match_count": target_match,
        },
        },
        fp_samples,
        fn_samples,
    )


def run_linking_evaluation(config_path: Path) -> Path:
    cfg_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lp_cfg_raw = cfg_raw.get("linking_pipeline", {})
    domain = str(lp_cfg_raw.get("domain", ""))
    cfg = _expand_placeholders(cfg_raw, {"domain": domain})
    lp_cfg = cfg.get("linking_pipeline", {})

    levels = lp_cfg.get("levels", ["paragraph"])
    variants = lp_cfg.get("retrieval_variants", ["lead_paragraph", "full_text"])
    run_all_levels = bool(lp_cfg.get("run_all_levels", True))
    run_all_variants = bool(lp_cfg.get("run_all_variants", True))

    if not run_all_levels:
        levels = [str(lp_cfg.get("level", "paragraph"))]
    if not run_all_variants:
        variants = [str(lp_cfg.get("retrieval_variant", "lead_paragraph"))]

    # Point to variant pipeline outputs structure
    out_root = Path(lp_cfg.get("output_dir", "outputs")).parent / "linking_pipeline" / domain
    out_root.mkdir(parents=True, exist_ok=True)

    log_dir = out_root / "logs"
    logger, _ = create_logger(log_dir, script_name="linking_eval")
    logger.info(f"Evaluating domain={domain} levels={levels} variants={variants}")

    data_dir = Path(lp_cfg.get("data_dir", f"data/processed/{domain}"))
    eval_cfg = lp_cfg.get("evaluation", {})
    max_error_samples = int(eval_cfg.get("max_error_samples", 0))
    summary_path = out_root / "evaluation_summary.csv"
    summary_fields = [
        "domain",
        "level",
        "variant",
        "metric",
        "precision",
        "recall",
        "f1",
        "pred_count",
        "gold_count",
        "match_count",
    ]

    summary_exists = summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=summary_fields)
        if not summary_exists:
            writer.writeheader()

        for level in levels:
            if level == "page":
                gold_map = _read_page_links(data_dir / f"pages_{domain}.jsonl")
            else:
                gold_map = _read_paragraph_links(data_dir / f"paragraph_links_{domain}.csv")

            for variant in variants:
                variant_tag = variant.replace("+", "_plus_")
                pred_path = out_root / variant_tag / level / "predictions.jsonl"
                if not pred_path.exists():
                    logger.warning(f"Missing predictions: {pred_path}")
                    continue

                metrics, fp_samples, fn_samples = evaluate_predictions(
                    pred_path, gold_map, max_error_samples=max_error_samples
                )
                metrics_path = pred_path.parent / "metrics.json"
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                if max_error_samples > 0:
                    fp_path = pred_path.parent / "false_positives.jsonl"
                    fn_path = pred_path.parent / "false_negatives.jsonl"
                    with fp_path.open("w", encoding="utf-8") as f:
                        for row in fp_samples:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    with fn_path.open("w", encoding="utf-8") as f:
                        for row in fn_samples:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                for metric_name, vals in metrics.items():
                    writer.writerow({
                        "domain": domain,
                        "level": level,
                        "variant": variant,
                        "metric": metric_name,
                        "precision": vals["precision"],
                        "recall": vals["recall"],
                        "f1": vals["f1"],
                        "pred_count": vals["pred_count"],
                        "gold_count": vals["gold_count"],
                        "match_count": vals["match_count"],
                    })

                logger.info(f"Saved metrics: {metrics_path}")

    return summary_path
