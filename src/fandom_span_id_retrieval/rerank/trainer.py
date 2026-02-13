from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from fandom_span_id_retrieval.rerank.dataset import RerankDataset
from fandom_span_id_retrieval.utils.logging_utils import create_logger


def _compute_binary_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    logits = np.array(logits)
    labels = np.array(labels)
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean() if len(labels) else 0.0
    return {"accuracy": float(acc)}


def _load_datasets(data_dir: Path, tokenizer, max_length: int) -> Tuple[RerankDataset, RerankDataset]:
    train_ds = RerankDataset(data_dir / "train.jsonl", tokenizer, max_length=max_length)
    val_ds = RerankDataset(data_dir / "val.jsonl", tokenizer, max_length=max_length)
    return train_ds, val_ds


def train_rerank_from_config(config: Dict[str, Any]) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    out_cfg = config["output"]

    data_dir = Path(data_cfg["data_dir"])
    os.makedirs(out_cfg["dir"], exist_ok=True)

    log_dir = Path(out_cfg["dir"]) / "logs"
    logger, _ = create_logger(log_dir, script_name="rerank_train")
    logger.info(f"Device: {device}")
    logger.info(f"Data dir: {data_dir}")

    def _infer_num_labels(model_name: str, model_cfg: Dict[str, Any]) -> int:
        if "num_labels" in model_cfg:
            return int(model_cfg["num_labels"])
        if model_name.startswith("cross-encoder/"):
            return 1
        return 2

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["encoder_name"])
    train_ds, val_ds = _load_datasets(data_dir, tokenizer, data_cfg["max_length"])
    logger.info(f"Train examples: {len(train_ds)}")
    logger.info(f"Val examples: {len(val_ds)}")

    num_labels = _infer_num_labels(model_cfg["encoder_name"], model_cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["encoder_name"],
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    num_training_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = int(train_cfg["warmup_ratio"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_acc = 0.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
            if num_labels == 1:
                batch["labels"] = batch["labels"].float()
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc = _evaluate_accuracy(model, val_loader, device, num_labels)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_accuracy": val_acc})
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_accuracy={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = Path(out_cfg["dir"]) / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_name": model_cfg["encoder_name"],
                    "num_labels": num_labels,
                },
                ckpt_path,
            )
            logger.info(f"Saved best model: {ckpt_path} (val_accuracy={best_acc:.4f})")

    with open(Path(out_cfg["dir"]) / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved.")

    return Path(out_cfg["dir"]) / "best_model.pt"


def _evaluate_accuracy(model, dataloader, device, num_labels: int) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="val", unit="batch"):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
            outputs = model(**inputs)
            if num_labels == 1:
                preds = (outputs.logits.squeeze(-1) >= 0.0).long()
                correct += (preds == labels).sum().item()
            else:
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def load_rerank_model(ckpt_path: Path) -> Tuple[AutoModelForSequenceClassification, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_name = ckpt["encoder_name"]
    num_labels = int(ckpt.get("num_labels", 2))
    model = AutoModelForSequenceClassification.from_pretrained(encoder_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, encoder_name
