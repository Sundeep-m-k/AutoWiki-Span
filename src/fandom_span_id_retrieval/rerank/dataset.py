from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


class RerankDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        query_text = ex["query_text"]
        doc_text = ex["doc_text"]
        label = int(ex["label"])

        encoded = self.tokenizer(
            query_text,
            doc_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label,
            "query_id": ex.get("query_id"),
            "article_id": ex.get("article_id"),
        }
        return item
