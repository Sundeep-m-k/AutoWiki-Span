# src/fandom_span_id_retrieval/span_id/model.py

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from fandom_span_id_retrieval.eval.span_metrics import compute_span_metrics_for_trainer
from fandom_span_id_retrieval.utils.logging_utils import create_logger
from fandom_span_id_retrieval.pipeline.experiment import PROJECT_ROOT
from fandom_span_id_retrieval.span_id.preprocess import BILOU_LABELS, LABEL2ID, ID2LABEL
from fandom_span_id_retrieval.utils.model_registry import write_model_manifest


def _model_dir_for_cfg(span_cfg: Dict[str, Any]) -> Path:
    domain = span_cfg.get("domain", "unknown")
    raw_model_name = span_cfg.get("model_name", "bert-base-uncased")
    model_name = raw_model_name.split("/")[-1]
    level = span_cfg.get("level", "paragraph")
    normalize_punct = span_cfg.get("normalize_punctuation", False)
    punc_str = "punc" if not normalize_punct else "nopunc"

    base_dir = PROJECT_ROOT / span_cfg["token_dataset_dir"]
    return base_dir.parent / "models" / f"{domain}_{model_name}_{level}_{punc_str}"


def log_results_to_csv(span_cfg: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    # Base results directory
    results_dir_str = span_cfg.get("results_dir", "outputs/span_id")
    results_dir = PROJECT_ROOT / results_dir_str
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSV path (absolute)
    results_csv_str = span_cfg.get("results_csv", "outputs/span_id/all_experiments.csv")
    results_csv = PROJECT_ROOT / results_csv_str
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    domain = span_cfg.get("domain", "unknown")
    raw_model_name = span_cfg.get("model_name", "bert-base-uncased")
    model_name = raw_model_name.split("/")[-1]
    level = span_cfg.get("level", "paragraph")
    normalize_punct = span_cfg.get("normalize_punctuation", False)
    punc_str = "punc" if not normalize_punct else "nopunc"

    exp_name = f"{domain}_{model_name}_{level}_{punc_str}"

    row = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domain": domain,
        "model": model_name,
        "level": level,
        "normalize_punctuation": normalize_punct,
        "eval_f1": metrics.get("eval_f1_seqeval", metrics.get("eval_f1", 0.0)),
        "eval_precision": metrics.get("eval_precision_seqeval", metrics.get("eval_precision", 0.0)),
        "eval_recall": metrics.get("eval_recall_seqeval", metrics.get("eval_recall", 0.0)),
        "span_f1": metrics.get("eval_span_f1", metrics.get("span_f1", 0.0)),
        "span_precision": metrics.get("eval_span_precision", metrics.get("span_precision", 0.0)),
        "span_recall": metrics.get("eval_span_recall", metrics.get("span_recall", 0.0)),
        "token_f1": metrics.get("eval_token_f1", metrics.get("token_f1", 0.0)),
        "token_precision": metrics.get("eval_token_precision", metrics.get("token_precision", 0.0)),
        "token_recall": metrics.get("eval_token_recall", metrics.get("token_recall", 0.0)),
        "exact_span_f1": metrics.get("eval_exact_span_f1", metrics.get("exact_span_f1", 0.0)),
        "exact_span_precision": metrics.get("eval_exact_span_precision", metrics.get("exact_span_precision", 0.0)),
        "exact_span_recall": metrics.get("eval_exact_span_recall", metrics.get("exact_span_recall", 0.0)),
        "relaxed_span_f1": metrics.get("eval_relaxed_span_f1", metrics.get("relaxed_span_f1", 0.0)),
        "relaxed_span_precision": metrics.get("eval_relaxed_span_precision", metrics.get("relaxed_span_precision", 0.0)),
        "relaxed_span_recall": metrics.get("eval_relaxed_span_recall", metrics.get("relaxed_span_recall", 0.0)),
        "eval_loss": metrics.get("eval_loss", 0.0),
        "num_epochs": span_cfg.get("train", {}).get("num_epochs", 0),
        "learning_rate": span_cfg.get("train", {}).get("learning_rate", 0.0),
        "batch_size": span_cfg.get("train", {}).get("batch_size", 0),
    }

    file_exists = results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return exp_name


def save_experiment_summary(span_cfg: Dict[str, Any], metrics: Dict[str, Any], model_dir: Path) -> Path:
    results_dir_str = span_cfg.get("results_dir", "outputs/span_id")
    results_dir = PROJECT_ROOT / results_dir_str
    results_dir.mkdir(parents=True, exist_ok=True)


    domain = span_cfg.get("domain", "unknown")
    raw_model_name = span_cfg.get("model_name", "bert-base-uncased")
    model_name = raw_model_name.split("/")[-1]
    level = span_cfg.get("level", "paragraph")
    normalize_punct = span_cfg.get("normalize_punctuation", False)
    punc_str = "punc" if not normalize_punct else "nopunc"

    exp_name = f"{domain}_{model_name}_{level}_{punc_str}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "config": {
            "domain": domain,
            "model": model_name,
            "level": level,
            "normalize_punctuation": normalize_punct,
            "max_seq_length": span_cfg.get("max_seq_length", 512),
            "label_scheme": span_cfg.get("label_scheme", "BILOU"),
            "num_labels": span_cfg.get("num_labels", len(BILOU_LABELS)),
        },
        "training": span_cfg.get("train", {}),
        "metrics": metrics,
        "model_dir": str(model_dir),
    }

    summary_file = results_dir / f"{exp_name}_{timestamp}.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_file


def _load_jsonl(path: Path) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _build_hf_datasets(span_cfg: Dict[str, Any], tokenizer) -> DatasetDict:
    train_path = PROJECT_ROOT / span_cfg["train_file"]
    dev_path = PROJECT_ROOT / span_cfg["dev_file"]
    test_path = PROJECT_ROOT / span_cfg["test_file"]

    train_ds = _load_jsonl(train_path)
    dev_ds = _load_jsonl(dev_path)
    test_ds = _load_jsonl(test_path)

    max_seq_length = int(span_cfg.get("max_seq_length", 512))
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def pad_features(ex):
        input_ids = ex["input_ids"]
        attention_mask = ex["attention_mask"]
        labels = ex["label_ids"]

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            labels = labels[:max_seq_length]
        else:
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [LABEL2ID["O"]] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_ds = train_ds.map(pad_features, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(pad_features, remove_columns=dev_ds.column_names)
    test_ds = test_ds.map(pad_features, remove_columns=test_ds.column_names)

    return DatasetDict(train=train_ds, validation=dev_ds, test=test_ds)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.array(logits)
    labels = np.array(labels)
    preds = np.argmax(logits, axis=-1)

    preds_list = preds.tolist()
    labels_list = labels.tolist()

    pred_tags = [[ID2LABEL[id_] for id_ in seq] for seq in preds_list]
    true_tags = [[ID2LABEL[id_] for id_ in seq] for seq in labels_list]

    metrics = {
        "f1_seqeval": f1_score(true_tags, pred_tags),
        "precision_seqeval": precision_score(true_tags, pred_tags),
        "recall_seqeval": recall_score(true_tags, pred_tags),
    }

    span_metrics = compute_span_metrics_for_trainer((logits, labels), id2label=ID2LABEL)
    return {**metrics, **span_metrics}


def train_model_from_cfg(span_cfg: Dict[str, Any]) -> Path:
    log_dir = PROJECT_ROOT / span_cfg["log_dir"]
    logger, _ = create_logger(log_dir, script_name="03_train_span_identifier")
    logger.info("Step 3: Training span identification model")

    model_name = span_cfg["model_name"]
    num_labels = int(span_cfg.get("num_labels", len(BILOU_LABELS)))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = _build_hf_datasets(span_cfg, tokenizer)

    logger.info(
        f"Loaded datasets: train={len(datasets['train'])}, "
        f"dev={len(datasets['validation'])}, test={len(datasets['test'])}"
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    out_dir = _model_dir_for_cfg(span_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = PROJECT_ROOT / span_cfg.get(
        "tensorboard_dir", f"data/tensorboard/span_identification/{span_cfg['domain']}"
    )
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = span_cfg["train"]

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        logging_dir=str(tensorboard_dir),
        logging_strategy="steps",
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        per_device_eval_batch_size=int(train_cfg["batch_size"]),
        num_train_epochs=int(train_cfg["num_epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        seed=int(train_cfg["seed"]),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    write_model_manifest(
        out_dir,
        {
            "task": "span_id",
            "domain": span_cfg.get("domain"),
            "model_name": model_name,
            "level": span_cfg.get("level"),
            "normalize_punctuation": span_cfg.get("normalize_punctuation", False),
            "train": span_cfg.get("train", {}),
            "max_seq_length": span_cfg.get("max_seq_length", 512),
        },
    )
    logger.info(f"Training complete. Model saved to: {out_dir}")

    return out_dir


def evaluate_model_from_cfg(span_cfg: Dict[str, Any]) -> Dict[str, Any]:
    log_dir = PROJECT_ROOT / span_cfg["log_dir"]
    logger, _ = create_logger(log_dir, script_name="04_eval_span_identifier")
    logger.info("Step 4: Evaluating span identification model on test set")

    model_name = span_cfg["model_name"]
    num_labels = int(span_cfg.get("num_labels", len(BILOU_LABELS)))

    model_dir = _model_dir_for_cfg(span_cfg)

    tokenizer_source = str(model_dir) if model_dir.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    datasets = _build_hf_datasets(span_cfg, tokenizer)
    test_ds = datasets["test"]

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    logger.info(f"Test examples: {len(test_ds)}")
    metrics = trainer.evaluate(eval_dataset=test_ds)
    logger.info(f"Test metrics (token/BILOU-level): {metrics}")

    exp_name = log_results_to_csv(span_cfg, metrics)
    logger.info(f"Results saved to CSV as: {exp_name}")

    summary_file = save_experiment_summary(span_cfg, metrics, model_dir)
    logger.info(f"Detailed summary saved to: {summary_file}")

    return metrics
