import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from fandom_span_id_retrieval.retrieval.dataset import RetrievalDataset
from fandom_span_id_retrieval.retrieval.model import ArticleRetriever
from fandom_span_id_retrieval.utils.model_registry import write_model_manifest


def _load_num_articles(articles_path: Path) -> int:
    max_id = -1
    with open(articles_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            aid = int(obj["article_id"])
            if aid > max_id:
                max_id = aid
    return max_id + 1


def _make_dataloaders(
    data_dir: Path,
    tokenizer,
    max_length: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = RetrievalDataset(
        data_dir / "queries_train.jsonl",
        tokenizer,
        max_length=max_length,
    )
    val_ds = RetrievalDataset(
        data_dir / "queries_val.jsonl",
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader


def _train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        if (step + 1) % log_interval == 0:
            print(f"step {step+1}, loss={total_loss / (step+1):.4f}")
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, k_list=(1, 5, 10)) -> Dict[str, float]:
    model.eval()
    total = 0
    correct_at_k = {k: 0 for k in k_list}

    for batch in tqdm(dataloader):
        labels = batch["labels"]
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs["logits"]  # [B, num_articles]
        topk = torch.topk(logits, k=max(k_list), dim=-1).indices

        labels = labels.to(device)
        total += labels.size(0)

        for k in k_list:
            in_topk = (topk[:, :k] == labels.unsqueeze(-1)).any(dim=-1)
            correct_at_k[k] += in_topk.sum().item()

    metrics = {f"recall@{k}": correct_at_k[k] / total for k in k_list}
    return metrics


def train_retrieval_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point called from pipeline/experiment.py

    config structure:
      task: retrieval
      dataset: {data_dir, max_length}
      model: {encoder_name, freeze_encoder}
      train: {batch_size, epochs, lr, weight_decay, warmup_ratio, log_interval}
      output: {dir}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    out_cfg = config["output"]

    data_dir = Path(data_cfg["data_dir"])
    articles_path = data_dir / "articles.jsonl"
    num_articles = _load_num_articles(articles_path)
    print("num_articles:", num_articles)

    os.makedirs(out_cfg["dir"], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["encoder_name"])
    train_loader, val_loader = _make_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=data_cfg["max_length"],
        batch_size=train_cfg["batch_size"],
    )

    model = ArticleRetriever(
        encoder_name=model_cfg["encoder_name"],
        num_articles=num_articles,
        freeze_encoder=model_cfg.get("freeze_encoder", False),
    ).to(device)

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

    best_val_recall = 0.0
    best_ckpt_path = os.path.join(out_cfg["dir"], "best_model.pt")
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"Epoch {epoch}/{train_cfg['epochs']}")
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_interval=train_cfg["log_interval"],
        )
        print(f"train_loss: {train_loss:.4f}")

        metrics = evaluate_retrieval(
            model=model,
            dataloader=val_loader,
            device=device,
            k_list=(1, 5, 10),
        )
        print("val_metrics:", metrics)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if metrics["recall@1"] > best_val_recall:
            best_val_recall = metrics["recall@1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_articles": num_articles,
                    "encoder_name": model_cfg["encoder_name"],
                },
                best_ckpt_path,
            )
            print("saved best model to", best_ckpt_path)

    if not os.path.exists(best_ckpt_path):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_articles": num_articles,
                "encoder_name": model_cfg["encoder_name"],
            },
            best_ckpt_path,
        )
        print("saved fallback model to", best_ckpt_path)

    # save training history
    with open(os.path.join(out_cfg["dir"], "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    write_model_manifest(
        Path(out_cfg["dir"]),
        {
            "task": "retrieval",
            "encoder_name": model_cfg.get("encoder_name"),
            "num_articles": num_articles,
            "train": train_cfg,
            "dataset": data_cfg,
            "best_val_recall@1": best_val_recall,
        },
    )

    return {"best_val_recall@1": best_val_recall, "history": history}
