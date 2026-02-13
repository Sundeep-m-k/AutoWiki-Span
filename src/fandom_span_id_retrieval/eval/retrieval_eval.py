import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fandom_span_id_retrieval.retrieval.dataset import RetrievalDataset
from fandom_span_id_retrieval.retrieval.model import ArticleRetriever
from fandom_span_id_retrieval.retrieval.trainer import evaluate_retrieval


def load_retrieval_model_for_eval(
    ckpt_path: str,
) -> Tuple[ArticleRetriever, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_name = ckpt["encoder_name"]
    num_articles = ckpt["num_articles"]

    model = ArticleRetriever(
        encoder_name=encoder_name,
        num_articles=num_articles,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, encoder_name


def eval_retrieval_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """
    config:
      dataset: {data_dir, max_length}
      output: {dir, ckpt_name}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["dataset"]
    out_cfg = config["output"]

    data_dir = Path(data_cfg["data_dir"])
    test_path = data_dir / "queries_test.jsonl"

    ckpt_path = Path(out_cfg["dir"]) / out_cfg.get("ckpt_name", "best_model.pt")

    model, encoder_name = load_retrieval_model_for_eval(str(ckpt_path))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    test_ds = RetrievalDataset(
        test_path,
        tokenizer,
        max_length=data_cfg["max_length"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    metrics = evaluate_retrieval(
        model=model,
        dataloader=test_loader,
        device=device,
        k_list=(1, 5, 10),
    )
    # save metrics
    with open(Path(out_cfg["dir"]) / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
