from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class ArticleRetriever(nn.Module):
    """
    Single-label classification over article IDs:
    query_text -> logits over article_id.
    """

    def __init__(
        self,
        encoder_name: str,
        num_articles: int,
        freeze_encoder: bool = False,
        init_article_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        d_model = self.encoder.config.hidden_size

        if init_article_emb is not None:
            assert init_article_emb.shape[0] == num_articles
            assert init_article_emb.shape[1] == d_model
            self.article_emb = nn.Parameter(init_article_emb)
        else:
            self.article_emb = nn.Parameter(
                torch.randn(num_articles, d_model) * 0.02
            )

    def encode_query(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return cls

    def forward(self, input_ids, attention_mask, labels=None):
        q = self.encode_query(input_ids, attention_mask)
        logits = torch.matmul(q, self.article_emb.t())
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}
