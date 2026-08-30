from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


def gather_candidates(nodes: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Gather [batch, node, hidden] into [batch, ue, k, hidden]."""
    batch = nodes.shape[0]
    batch_index = torch.arange(batch, device=nodes.device)[:, None, None]
    return nodes[batch_index, candidate_ids]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeAwareBipartiteAttention(nn.Module):
    """Multi-head, edge-aware attention over sparse UE-to-cell candidates."""

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.ue_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bs_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bs_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(heads, self.head_dim))

        self.ue_update = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.ue_to_bs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bs_update = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.ue_norm = nn.LayerNorm(hidden_dim)
        self.bs_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attention_vector)

    def forward(
        self,
        ue: torch.Tensor,
        bs: torch.Tensor,
        edge: torch.Tensor,
        candidate_bs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_ues, k = candidate_bs.shape
        num_bs = bs.shape[1]
        candidate_embeddings = gather_candidates(bs, candidate_bs)

        query = self.ue_query(ue).view(batch, num_ues, self.heads, self.head_dim)
        key = self.bs_key(candidate_embeddings).view(batch, num_ues, k, self.heads, self.head_dim)
        value = self.bs_value(candidate_embeddings).view(batch, num_ues, k, self.heads, self.head_dim)
        edge_hidden = self.edge_projection(edge).view(batch, num_ues, k, self.heads, self.head_dim)

        logits = torch.tanh(query[:, :, None] + key + edge_hidden)
        logits = (logits * self.attention_vector[None, None, None]).sum(dim=-1)
        logits = logits.masked_fill(~mask[..., None], -1e9)
        attention = torch.softmax(logits, dim=2)
        attention = attention * mask[..., None].to(attention.dtype)
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1e-9)

        message = (attention[..., None] * (value + edge_hidden)).sum(dim=2)
        message = message.reshape(batch, num_ues, self.hidden_dim)
        ue_new = self.ue_norm(ue + self.dropout(self.ue_update(torch.cat([ue, message], dim=-1))))

        # Send attention-weighted UE messages back to candidate cells. This is
        # the second direction of the bipartite graph and exposes dynamic load.
        ue_message = self.ue_to_bs(ue_new).view(batch, num_ues, 1, self.heads, self.head_dim)
        back_message = (attention[..., None] * (ue_message + edge_hidden)).reshape(
            batch, num_ues * k, self.hidden_dim
        )
        back_weight = attention.mean(dim=-1).reshape(batch, num_ues * k, 1)
        flat_ids = candidate_bs.reshape(batch, num_ues * k)
        flat_mask = mask.reshape(batch, num_ues * k, 1).to(back_message.dtype)

        aggregate = torch.zeros(batch, num_bs, self.hidden_dim, device=bs.device, dtype=bs.dtype)
        denominator = torch.zeros(batch, num_bs, 1, device=bs.device, dtype=bs.dtype)
        aggregate.scatter_add_(1, flat_ids[..., None].expand(-1, -1, self.hidden_dim), back_message * flat_mask)
        denominator.scatter_add_(1, flat_ids[..., None], back_weight * flat_mask)
        aggregate = aggregate / denominator.clamp_min(1e-6)
        bs_new = self.bs_norm(bs + self.dropout(self.bs_update(torch.cat([bs, aggregate], dim=-1))))
        return ue_new, bs_new


class BipartiteDuelingQNetwork(nn.Module):
    """Parameter-shared Dueling Q-network for variable-size UE/cell graphs."""

    def __init__(
        self,
        ue_dim: int,
        bs_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ue_encoder = MLP(ue_dim, hidden_dim, hidden_dim, dropout)
        self.bs_encoder = MLP(bs_dim, hidden_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList(
            [EdgeAwareBipartiteAttention(hidden_dim, edge_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.edge_for_q = MLP(edge_dim, hidden_dim, hidden_dim, dropout)
        self.value_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.advantage_head = MLP(hidden_dim * 3, hidden_dim, 1, dropout)

    def forward(self, observation: TensorDict) -> torch.Tensor:
        ue = self.ue_encoder(observation["ue"])
        bs = self.bs_encoder(observation["bs"])
        edge = observation["edge"]
        candidate_bs = observation["candidate_bs"]
        mask = observation["mask"]

        for layer in self.layers:
            ue, bs = layer(ue, bs, edge, candidate_bs, mask)

        candidate_embeddings = gather_candidates(bs, candidate_bs)
        ue_expanded = ue[:, :, None, :].expand(-1, -1, candidate_bs.shape[-1], -1)
        edge_hidden = self.edge_for_q(edge)
        advantage = self.advantage_head(
            torch.cat([ue_expanded, candidate_embeddings, edge_hidden], dim=-1)
        ).squeeze(-1)
        value = self.value_head(ue).squeeze(-1)

        valid = mask.to(advantage.dtype)
        valid_mean = (advantage * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        q_values = value[..., None] + advantage - valid_mean[..., None]
        return q_values.masked_fill(~mask, -1e9)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
