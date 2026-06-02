"""
Compatible baseline models that handle batched graph inputs.

These wrappers make GCN/GAT baselines compatible with the same
batched data format as MAGNet: (B, N, in_dim) node features and
(B, N, N) adjacency matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if adj.dim() == 3:
            h = torch.bmm(adj, h)
        else:
            h = adj @ h
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class GATConvLayer(nn.Module):
    """Graph Attention layer for batched inputs."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.W(x)  # (B, N, out)

        # Compute attention coefficients
        h_i = h.unsqueeze(2).expand(B, N, N, -1)  # (B, N, N, out)
        h_j = h.unsqueeze(1).expand(B, N, N, -1)  # (B, N, N, out)
        e = self.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (B, N, N)

        # Mask non-adjacent nodes (but keep self-loops via adj)
        zero_mask = (adj == 0)
        e = e.masked_fill(zero_mask, -1e9)

        # Softmax over neighbors
        alpha = F.softmax(e, dim=-1)  # (B, N, N)
        # Replace any remaining NaN with 0
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # Weighted aggregation
        out = torch.bmm(alpha, h)  # (B, N, out)
        out = self.norm(out)
        return F.relu(out)


class BatchedGCN(nn.Module):
    """
    GCN baseline for batched molecular graph classification.

    Args:
        in_dim: Node feature dimension.
        hidden_dim: Hidden dimension.
        out_dim: Output dimension.
        num_layers: Number of GCN layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphConvLayer(hidden_dim, hidden_dim, dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) node features.
            adj: (B, N, N) adjacency matrices.
            batch: unused, for API compatibility.

        Returns:
            (B,) or (B, out_dim) predictions.
        """
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h, adj)  # Residual
            h = norm(h)

        # Global mean pooling over nodes
        graph_repr = h.mean(dim=1)  # (B, hidden)
        out = self.classifier(graph_repr)  # (B, out_dim)

        if out.dim() > 1 and out.size(-1) == 1:
            out = out.squeeze(-1)
        return out


class BatchedGAT(nn.Module):
    """
    GAT baseline for batched molecular graph classification.

    Args:
        in_dim: Node feature dimension.
        hidden_dim: Hidden dimension.
        out_dim: Output dimension.
        num_layers: Number of GAT layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConvLayer(hidden_dim, hidden_dim, dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) node features.
            adj: (B, N, N) adjacency matrices.
            batch: unused, for API compatibility.

        Returns:
            (B,) or (B, out_dim) predictions.
        """
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h, adj)  # Residual
            h = norm(h)

        # Global mean pooling over nodes
        graph_repr = h.mean(dim=1)  # (B, hidden)
        out = self.classifier(graph_repr)  # (B, out_dim)

        if out.dim() > 1 and out.size(-1) == 1:
            out = out.squeeze(-1)
        return out


class BatchedMLP(nn.Module):
    """
    MLP baseline that operates on flattened graph representations.

    Uses node feature statistics (mean, max) as graph-level features.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input: concatenated mean + max pooling = 2 * in_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) node features.
            adj: unused.
            batch: unused.

        Returns:
            (B,) or (B, out_dim) predictions.
        """
        # Mean + Max pooling
        mean_pool = x.mean(dim=1)  # (B, in_dim)
        max_pool = x.max(dim=1)[0]  # (B, in_dim)
        graph_feat = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*in_dim)

        out = self.mlp(graph_feat)  # (B, out_dim)
        if out.dim() > 1 and out.size(-1) == 1:
            out = out.squeeze(-1)
        return out
