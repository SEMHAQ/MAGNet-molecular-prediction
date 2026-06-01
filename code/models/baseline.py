"""
Baseline models for CompBioChem project.

Includes:
- BaselineModel: MLP encoder-decoder (placeholder)
- BiologicalFeatureEncoder: Self-attention based feature encoder
- GNNModel: Graph Neural Network for molecular/biological graph data (GCN/GAT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BaselineModel(nn.Module):
    """
    Baseline MLP model for computational biology tasks.

    Simple encoder-decoder architecture. Replace with your actual model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class BiologicalFeatureEncoder(nn.Module):
    """
    Attention-based encoder for biological features.

    Uses multi-head self-attention with residual connections and layer normalization.
    Suitable for sequence-like biological data (e.g., protein sequences, genomic features).
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode biological features with self-attention.

        Args:
            x: Input features of shape (batch_size, seq_len, feature_dim).

        Returns:
            Encoded features of shape (batch_size, seq_len, embedding_dim).
        """
        embedded = self.embedding(x)
        attended, _ = self.attention(embedded, embedded, embedded)
        output = self.norm(embedded + self.dropout(attended))
        return output


# ---------------------------------------------------------------------------
# Graph Neural Network components
# ---------------------------------------------------------------------------


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).

    Computes: H' = σ(D^{-1/2} A D^{-1/2} H W + b)

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        dropout: Dropout rate.
        use_bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features, shape (N, in_features) or (B, N, in_features).
            adj: Adjacency matrix with self-loops and normalization,
                 shape (N, N) or (B, N, N).
        """
        x = self.dropout(x)
        # Support both batched and unbatched
        if adj.dim() == 2:
            support = self.linear(x)
            out = torch.spmm(adj, support) if support.is_sparse else adj @ support
        else:
            support = self.linear(x)  # (B, N, out)
            out = torch.bmm(adj, support)
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Veličković et al., 2018).

    Single-head GAT with LeakyReLU attention.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        dropout: Dropout rate.
        negative_slope: LeakyReLU negative slope.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features, shape (N, in_features).
            adj: Adjacency matrix (binary or weighted), shape (N, N).
        """
        N = x.size(0)
        h = self.W(x)  # (N, out)

        # Compute attention coefficients
        h_i = h.unsqueeze(1).expand(N, N, -1)  # (N, N, out)
        h_j = h.unsqueeze(0).expand(N, N, -1)  # (N, N, out)
        e = self.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (N, N)

        # Mask non-adjacent nodes
        zero_mask = (adj == 0)
        e = e.masked_fill(zero_mask, float("-inf"))

        # Softmax over neighbors
        alpha = F.softmax(e, dim=-1)  # (N, N)
        alpha = self.dropout(alpha)

        # Weighted aggregation
        out = torch.mm(alpha, h)  # (N, out)
        return out


class GNNModel(nn.Module):
    """
    Graph Neural Network for molecular/biological graph classification.

    Supports GCN and GAT backends. Takes node features and adjacency matrix,
    applies graph convolutions, global pooling, and a classifier head.

    Typical usage for molecular data:
        - Nodes = atoms, Edges = bonds
        - Node features = atom descriptors (element, charge, hybridization, etc.)
        - Graph-level task = property prediction, toxicity classification, etc.

    Args:
        in_dim: Node feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Number of output classes (1 for binary with sigmoid).
        num_layers: Number of graph convolution layers.
        conv_type: 'gcn' or 'gat'.
        dropout: Dropout rate.
        pooling: Global pooling type ('mean', 'max', 'sum').
        residual: Whether to use residual connections.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        pooling: str = "mean",
        residual: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == "gcn":
                self.convs.append(GCNLayer(hidden_dim, hidden_dim, dropout=dropout))
            elif conv_type == "gat":
                self.convs.append(GATLayer(hidden_dim, hidden_dim, dropout=dropout))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}. Use 'gcn' or 'gat'.")
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Global pooling: (N, H) -> (H,)."""
        if self.pooling == "mean":
            return x.mean(dim=0)
        elif self.pooling == "max":
            return x.max(dim=0)[0]
        elif self.pooling == "sum":
            return x.sum(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for graph classification.

        Args:
            x: Node features, shape (N, in_dim) for single graph or
               (total_nodes, in_dim) for batched graphs.
            adj: Adjacency matrix. Shape (N, N) or (B, N, N).
            batch: Batch vector mapping each node to its graph index,
                   shape (total_nodes,). Required for batched graphs.

        Returns:
            Graph-level output, shape (out_dim,) for single graph or
            (B, out_dim) for batched graphs.
        """
        # Input projection
        h = F.relu(self.input_proj(x))

        # Graph convolutions
        for i in range(self.num_layers):
            h_new = F.relu(self.convs[i](h, adj))
            h_new = self.norms[i](h_new)
            if self.residual:
                h = h + h_new
            else:
                h = h_new

        # Global pooling + classification
        if batch is not None:
            # Batched: pool per graph
            batch_size = batch.max().item() + 1
            pooled = []
            for g in range(batch_size):
                mask = batch == g
                pooled.append(self._pool(h[mask]))
            graph_emb = torch.stack(pooled)  # (B, H)
        else:
            graph_emb = self._pool(h).unsqueeze(0)  # (1, H)

        out = self.classifier(graph_emb)

        if out.size(0) == 1:
            return out.squeeze(0)
        return out


class MolecularGNNClassifier(nn.Module):
    """
    Convenience wrapper: GNNModel + sigmoid for binary molecular classification.

    Combines GNNModel with an output sigmoid for binary tasks (e.g., drug-likeness,
    toxicity prediction). Returns raw logits for use with BCEWithLogitsLoss.

    Args:
        in_dim: Node feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of GNN layers.
        conv_type: 'gcn' or 'gat'.
        dropout: Dropout rate.
        pooling: Global pooling type.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.gnn = GNNModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout,
            pooling=pooling,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns raw logits for BCEWithLogitsLoss."""
        return self.gnn(x, adj, batch).squeeze(-1)
