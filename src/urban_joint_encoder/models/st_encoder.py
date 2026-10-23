"""Lightweight GCN + GRU encoder with a separate prediction head."""

from __future__ import annotations

import torch
from torch import nn


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Return D^-1/2 (A+I) D^-1/2."""
    adjacency = adjacency.float()
    a_hat = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
    degree = a_hat.sum(dim=1)
    inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    return inv_sqrt[:, None] * a_hat * inv_sqrt[None, :]


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, adjacency: torch.Tensor, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(adjacency.shape[0])
        self.register_buffer("normalized_adjacency", normalize_adjacency(adjacency))
        self.spatial_projection = nn.Linear(1, self.hidden_dim)
        self.spatial_activation = nn.ReLU()
        self.temporal_encoder = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        if x.ndim != 4 or x.shape[2:] != (self.num_nodes, 1):
            raise ValueError(f"Expected [B,T,{self.num_nodes},1], received {tuple(x.shape)}")
        # Each time slice: A_norm @ X, then feature projection.
        spatial = torch.einsum("nm,btmf->btnf", self.normalized_adjacency, x)
        spatial = self.spatial_activation(self.spatial_projection(spatial))
        batch, steps, nodes, channels = spatial.shape
        per_node_sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        temporal_sequence, _ = self.temporal_encoder(per_node_sequences)
        temporal_sequence = temporal_sequence.reshape(batch, nodes, steps, self.hidden_dim).permute(0, 2, 1, 3)
        embedding = temporal_sequence[:, -1]
        return (embedding, temporal_sequence) if return_sequence else embedding


class STForecastModel(nn.Module):
    def __init__(self, adjacency: torch.Tensor, hidden_dim: int = 32):
        super().__init__()
        self.encoder = SpatialTemporalEncoder(adjacency, hidden_dim)
        self.prediction_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        encoded = self.encoder(x, return_sequence=return_sequence)
        if return_sequence:
            embedding, hidden_sequence = encoded
            prediction = self.prediction_head(embedding).squeeze(-1)
            return prediction, embedding, hidden_sequence
        embedding = encoded
        return self.prediction_head(embedding).squeeze(-1), embedding

