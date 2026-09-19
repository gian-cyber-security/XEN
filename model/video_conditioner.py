from __future__ import annotations

import torch
from torch import nn


class XENVideoTextEncoder(nn.Module):
    """Trainable byte-level text conditioner for XEN-GEN1-V.

    This intentionally starts lightweight so it can be trained from scratch on
    consumer hardware. It returns a global conditioning vector plus token
    features for future cross-attention extensions.
    """

    def __init__(self, vocab_size: int = 260, dim: int = 384, max_tokens: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Embedding(max_tokens, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids[:, : self.position.num_embeddings]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        x = self.embedding(input_ids) + self.position(positions)
        mask = (input_ids != 0).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)
