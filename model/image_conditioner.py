from __future__ import annotations

import torch
from torch import nn


class XENImageTextEncoder(nn.Module):
    """Trainable text conditioner for XEN-GEN1-I."""

    def __init__(self, vocab_size=260, dim=256, max_tokens=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(max_tokens, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids):
        if input_ids.shape[1] > self.position.num_embeddings:
            input_ids = input_ids[:, : self.position.num_embeddings]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        x = self.embedding(input_ids) + self.position(positions)
        mask = (input_ids != 0).float()[:, :, None]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(x)
