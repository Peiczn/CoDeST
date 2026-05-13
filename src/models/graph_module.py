from __future__ import annotations

import torch
from torch import nn


class GraphSAGELayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        src, dst = edge_index
        weight = edge_weight if edge_weight is not None else torch.ones_like(src, dtype=x.dtype)
        neigh = torch.zeros_like(x)
        neigh.index_add_(0, dst, x[src] * weight.unsqueeze(1))
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, weight)
        neigh = neigh / deg.clamp_min(1.0).unsqueeze(1)
        out = self.lin_self(x) + self.lin_neigh(neigh)
        out = self.dropout(self.act(self.norm(out)))
        return out


class GraphModule(nn.Module):
    def __init__(self, dim: int = 256, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GraphSAGELayer(dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = z
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_weight)
        return h
