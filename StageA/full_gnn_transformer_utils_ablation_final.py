#!/usr/bin/env python3
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def build_edge_index_from_pairs(
    src_names,
    dst_names,
    edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    weight_col: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    src_idx = {s: i for i, s in enumerate(src_names)}
    dst_idx = {s: i for i, s in enumerate(dst_names)}
    edge_rows = []

    for _, row in edges.iterrows():
        s = str(row[src_col])
        d = str(row[dst_col])
        if s in src_idx and d in dst_idx:
            w = (
                float(row[weight_col])
                if weight_col and weight_col in row and pd.notna(row[weight_col])
                else 1.0
            )
            edge_rows.append((src_idx[s], dst_idx[d], max(w, 1e-6)))

    if not edge_rows:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edge_index = np.array(
        [[x[0] for x in edge_rows], [x[1] for x in edge_rows]],
        dtype=np.int64,
    )
    edge_weight = np.array([x[2] for x in edge_rows], dtype=np.float32)
    return edge_index, edge_weight


def bipartite_to_intra(
    num_a: int,
    num_b: int,
    edge_index_ab: np.ndarray,
    edge_weight: np.ndarray,
    side: str,
) -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((num_a, num_b), dtype=np.float32)
    if edge_index_ab.shape[1] > 0:
        A[edge_index_ab[0], edge_index_ab[1]] = edge_weight

    M = A @ A.T if side == 'a' else A.T @ A
    np.fill_diagonal(M, 0.0)
    src, dst = np.where(M > 0)

    if len(src) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    return np.vstack([src, dst]).astype(np.int64), M[src, dst].astype(np.float32)


def make_undirected_with_self_loops(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    self_loop_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device if edge_index.numel() > 0 else edge_weight.device

    if edge_index.numel() == 0:
        loop_idx = torch.arange(num_nodes, device=device)
        edge_index = torch.stack([loop_idx, loop_idx], dim=0)
        edge_weight = torch.full((num_nodes,), float(self_loop_weight), dtype=torch.float32, device=device)
        return edge_index, edge_weight

    src, dst = edge_index[0], edge_index[1]
    rev_edge_index = torch.stack([dst, src], dim=0)
    rev_edge_weight = edge_weight.clone()

    loop_idx = torch.arange(num_nodes, device=device)
    loop_edge_index = torch.stack([loop_idx, loop_idx], dim=0)
    loop_edge_weight = torch.full((num_nodes,), float(self_loop_weight), dtype=edge_weight.dtype, device=device)

    new_edge_index = torch.cat([edge_index, rev_edge_index, loop_edge_index], dim=1)
    new_edge_weight = torch.cat([edge_weight, rev_edge_weight, loop_edge_weight], dim=0)
    return new_edge_index, new_edge_weight


class GCNLayer(nn.Module):
    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        _, num_nodes, _ = x.shape
        edge_index, edge_weight = make_undirected_with_self_loops(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            self_loop_weight=1.0,
        )

        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=x.device)
        deg.index_add_(0, dst, edge_weight)
        deg = torch.clamp(deg, min=1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

        xw = self.lin(x)
        msg = xw[:, src, :] * norm.view(1, -1, 1)
        out = torch.zeros_like(xw)
        out.index_add_(1, dst, msg)
        return out


class StackedGCN(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index, edge_weight)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)
        return h


class FullGNNTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 1,
        dropout: float = 0.2,
        use_gnn: bool = True,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_transformer = use_transformer

        self.feature_embed = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.value_proj = nn.Linear(1, d_model)
        self.input_dropout = nn.Dropout(dropout)

        if self.use_gnn:
            self.graph = StackedGCN(dim=d_model, num_layers=num_gnn_layers, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
        else:
            self.graph = None
            self.norm1 = None

        if self.use_transformer:
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(enc, num_layers=num_transformer_layers)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.transformer = None
            self.norm2 = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        tok = self.value_proj(x.unsqueeze(-1)) + self.feature_embed.unsqueeze(0)
        tok = self.input_dropout(tok)

        if self.use_gnn:
            tok = self.graph(tok, edge_index, edge_weight)
            tok = self.norm1(tok)

        if self.use_transformer:
            tok = self.transformer(tok)
            tok = self.norm2(tok)

        pooled = tok.mean(dim=1)
        return tok, pooled
