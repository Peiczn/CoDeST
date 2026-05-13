from __future__ import annotations

import torch
from torch import nn

from src.models.model import MultiTaskSpatialModel


class FrozenTeacher(nn.Module):
    def __init__(self, model: MultiTaskSpatialModel) -> None:
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, use_graph=False, task="deconv")
        return out["deconv_prob"]
