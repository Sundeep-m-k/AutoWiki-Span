import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):
    def __init__(self, path, tokenizer, max_length: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict] = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append(obj)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        query_text = ex["query_text"]
        label = int(ex["article_id"])

        encoded = self.tokenizer(
            query_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item
