"""Compact latent visual context; the ACT decoder keeps every spatial token."""

import torch
from torch import nn


class VisualPooling(nn.Module):
    def __init__(self, dim, heads, cameras, queries=4, mode="multi_query_attention"):
        super().__init__()
        self.mode = mode
        self.camera = nn.Embedding(cameras, dim)
        if mode == "multi_query_attention":
            self.queries = nn.Embedding(queries, dim)
            self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)

    def forward(self, features, positions):
        tokens = torch.cat(
            [
                (feature + position + self.camera.weight[i]).transpose(0, 1)
                for i, (feature, position) in enumerate(zip(features, positions, strict=True))
            ],
            dim=1,
        )
        if self.mode == "gap":
            return tokens.mean(1, keepdim=True)
        queries = self.queries.weight.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled = self.attention(queries, tokens, tokens, need_weights=False)[0]
        return self.norm(queries + pooled)
