from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import Encoder
from src.models.graph_module import GraphModule
from src.models.heads import DeathHead, DeconvHead, GeneDecoder


class MultiTaskSpatialModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4000,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        n_cell_types: int = 34,
        n_death_programs: int = 4,
        dropout: float = 0.1,
        graph_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.graph = GraphModule(dim=latent_dim, num_layers=graph_layers, dropout=dropout)
        self.deconv_head = DeconvHead(input_dim=latent_dim, output_dim=n_cell_types, hidden_dim=latent_dim, dropout=dropout)
        self.death_head = DeathHead(input_dim=latent_dim, output_dim=n_death_programs, hidden_dim=latent_dim, dropout=dropout)
        self.gene_decoder = GeneDecoder(input_dim=latent_dim, output_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward_repr(self, x: torch.Tensor, edge_index: torch.Tensor | None = None, edge_weight: torch.Tensor | None = None, use_graph: bool = True) -> torch.Tensor:
        z = self.encoder(x)
        if use_graph and edge_index is not None:
            return self.graph(z, edge_index, edge_weight)
        return z

    def predict_deconv(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.deconv_head(h)

    def predict_death(self, h: torch.Tensor) -> torch.Tensor:
        return self.death_head(h)

    def reconstruct_genes(self, h: torch.Tensor) -> torch.Tensor:
        return self.gene_decoder(h)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        use_graph: bool = True,
        task: str = "all",
    ) -> dict[str, torch.Tensor]:
        h = self.forward_repr(x, edge_index=edge_index, edge_weight=edge_weight, use_graph=use_graph)
        out = {"repr": h}
        if task in {"all", "deconv"}:
            logits, prob = self.predict_deconv(h)
            out["deconv_logits"] = logits
            out["deconv_prob"] = prob
        if task in {"all", "death"}:
            out["death_pred"] = self.predict_death(h)
        if task in {"all", "recon"}:
            out["x_recon"] = self.reconstruct_genes(h)
        return out
