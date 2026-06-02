"""
MAGNet: Multi-scale Attention Graph Network for Molecular Property Prediction.

This module implements the proposed method for the CompBioChem paper.
Key components:
- MultiScaleGNN: Extracts features at multiple graph depths
- CrossScaleAttention: Fuses multi-scale features with learned attention
- AttentionPooling: Global pooling with learnable queries
- MAGNet: Complete model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class GraphConvLayer(nn.Module):
    """
    Basic graph convolution layer with residual connection.

    Implements: H' = ReLU(D^{-1/2} A D^{-1/2} H W) + H (if residual)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and (in_dim == out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_dim) or (B, N, in_dim)
            adj: Normalized adjacency matrix
        """
        h = self.linear(x)
        if adj.dim() == 2:
            h = adj @ h
        else:
            h = torch.bmm(adj, h)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        if self.residual:
            h = h + x
        return h


class MultiScaleGNN(nn.Module):
    """
    Multi-scale Graph Neural Network.

    Extracts node features at multiple scales (depths) by applying
    graph convolution repeatedly and capturing intermediate outputs.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden dimension for all scales.
        num_scales: Number of scales (depths) to extract features from.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_scales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_scales = num_scales

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Graph convolution layers for each scale
        self.conv_layers = nn.ModuleList()
        self.scale_norms = nn.ModuleList()

        for _ in range(num_scales):
            self.conv_layers.append(GraphConvLayer(hidden_dim, hidden_dim, dropout=dropout))
            self.scale_norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            x: Node features (N, in_dim) or (B, N, in_dim)
            adj: Adjacency matrix

        Returns:
            List of node features at each scale, each (N, hidden_dim) or (B, N, hidden_dim)
        """
        h = F.relu(self.input_proj(x))

        scale_features = []
        for i in range(self.num_scales):
            h = self.conv_layers[i](h, adj)
            h = self.scale_norms[i](h)
            scale_features.append(h)

        return scale_features


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention mechanism for fusing multi-scale features.

    Learns attention weights to combine features from different scales,
    allowing the model to focus on the most informative scale for each node.

    Args:
        hidden_dim: Feature dimension (same across all scales).
        num_scales: Number of input scales.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_scales: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Learnable scale queries
        self.scale_queries = nn.Parameter(torch.randn(num_scales, hidden_dim))

        # Multi-head attention for cross-scale fusion
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Scale-wise gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, num_scales),
            nn.Softmax(dim=-1),
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            scale_features: List of (B, N, H) tensors from different scales.

        Returns:
            Fused features (B, N, H).
        """
        # Stack scales: (B, N, S, H)
        stacked = torch.stack(scale_features, dim=2)
        B, N, S, H = stacked.shape

        # Flatten for attention: (B*N, S, H)
        flat = stacked.reshape(B * N, S, H)

        # Compute scale-wise gating weights
        concat_scales = stacked.reshape(B, N, S * H)  # (B, N, S*H)
        gate_weights = self.gate(concat_scales)  # (B, N, S)
        gate_weights = gate_weights.unsqueeze(-1)  # (B, N, S, 1)
        gated = stacked * gate_weights  # (B, N, S, H)
        fused_gated = gated.sum(dim=2)  # (B, N, H)

        # Multi-head cross-scale attention
        Q = self.q_proj(fused_gated).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(flat).reshape(B * N, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(flat).reshape(B * N, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Reshape Q for batched attention: (B*N, num_heads, 1, head_dim)
        Q = Q.reshape(B * N, self.num_heads, 1, self.head_dim)

        # Attention: (B*N, num_heads, 1, S)
        scale = self.head_dim ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum: (B*N, num_heads, 1, head_dim)
        out = torch.matmul(attn, V)
        out = out.reshape(B, N, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual + norm
        out = self.norm(out + fused_gated)

        return out


class AttentionPooling(nn.Module):
    """
    Global attention pooling with learnable queries.

    Uses learnable query vectors to attend to all nodes in the graph,
    producing a fixed-size graph-level representation.

    Args:
        hidden_dim: Node feature dimension.
        num_queries: Number of learnable query vectors.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Combine queries
        self.query_combine = nn.Sequential(
            nn.Linear(num_queries * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, H).

        Returns:
            Graph-level representation (B, H).
        """
        B, N, H = x.shape

        # Expand queries for batch: (B, num_queries, H)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Attend: queries attend to nodes
        out, _ = self.attn(queries, x, x)  # (B, num_queries, H)
        out = self.norm(out)

        # Flatten and combine queries: (B, num_queries * H) -> (B, H)
        out = out.reshape(B, self.num_queries * H)
        out = self.query_combine(out)

        return out


class MAGNet(nn.Module):
    """
    Multi-scale Attention Graph Network for molecular property prediction.

    Architecture:
    1. MultiScaleGNN extracts features at multiple graph depths
    2. CrossScaleAttention fuses multi-scale features
    3. AttentionPooling produces graph-level representation
    4. MLP classifier head for prediction

    Args:
        in_dim: Input node feature dimension.
        hidden_dim: Hidden dimension throughout the model.
        out_dim: Output dimension (1 for binary, num_classes for multi-class).
        num_scales: Number of graph convolution scales.
        num_heads: Number of attention heads.
        num_queries: Number of learnable pooling queries.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_scales: int = 3,
        num_heads: int = 4,
        num_queries: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-scale feature extraction
        self.multi_scale_gnn = MultiScaleGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_scales=num_scales,
            dropout=dropout,
        )

        # Cross-scale attention fusion
        self.cross_scale_attn = CrossScaleAttention(
            hidden_dim=hidden_dim,
            num_scales=num_scales,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Global attention pooling
        self.pooling = AttentionPooling(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            dropout=dropout,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_dim) or (total_nodes, in_dim) for batched.
            adj: Adjacency matrix (N, N) or (B, N, N).
            batch: Batch vector mapping nodes to graphs.

        Returns:
            Graph-level predictions. Shape (B,) for binary classification.
        """
        # Handle single graph vs batched
        if adj.dim() == 2:
            # Single graph: add batch dimension
            x = x.unsqueeze(0)  # (1, N, in_dim)
            adj = adj.unsqueeze(0)  # (1, N, N)
            single_graph = True
        else:
            single_graph = False

        # Multi-scale features: list of (B, N, H)
        scale_features = self.multi_scale_gnn(x, adj)

        # Cross-scale fusion: (B, N, H)
        fused = self.cross_scale_attn(scale_features)

        # Global pooling: (B, H)
        graph_repr = self.pooling(fused)

        # Classification: (B, out_dim)
        out = self.classifier(graph_repr)

        # Squeeze last dimension for binary classification
        if out.dim() > 1 and out.size(-1) == 1:
            out = out.squeeze(-1)  # (B,)

        if single_graph:
            out = out.squeeze(0)  # scalar

        return out


class MAGNetClassifier(nn.Module):
    """
    Convenience wrapper for binary molecular classification with MAGNet.

    Returns raw logits for use with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_scales: int = 3,
        num_heads: int = 4,
        num_queries: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.magnet = MAGNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_scales=num_scales,
            num_heads=num_heads,
            num_queries=num_queries,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns raw logits for BCEWithLogitsLoss."""
        return self.magnet(x, adj, batch).squeeze(-1)
