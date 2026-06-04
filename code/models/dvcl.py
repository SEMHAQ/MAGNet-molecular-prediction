"""
DVCL: Dual-View Contrastive Learning for Molecular Property Prediction.

Key innovations:
1. Dual-view contrastive learning: molecular graph (DMPNN) + Morgan fingerprint (MLP)
2. Chemistry-aware graph augmentations (functional group masking, scaffold perturbation)
3. Pre-train on unlabeled molecules, fine-tune on downstream tasks

Architecture:
- Graph encoder: DMPNN with enriched features (36-dim atom, 12-dim bond)
- Fingerprint encoder: MLP processing Morgan fingerprints (2048-dim)
- Projection head: maps both views to shared embedding space
- Fine-tune head: classifier on top of frozen/unfrozen encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import numpy as np


# ============================================================
# Morgan fingerprint extraction
# ============================================================
def morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Compute Morgan fingerprint as numpy array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def morgan_counts(smiles, radius=2, n_bits=2048):
    """Compute Morgan count fingerprint (more informative than binary)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx, count in fp.GetNonzeroElements().items():
        arr[idx % n_bits] = count
    return arr


# ============================================================
# Chemistry-aware graph augmentations
# ============================================================
def atom_mask_augment(data, mask_ratio=0.15):
    """
    Mask atom features (set to zero) for a random subset of atoms.
    Preserves molecular connectivity — only masks features.
    """
    import copy
    aug = copy.copy(data)
    num_nodes = aug.x.size(0)
    num_mask = max(1, int(num_nodes * mask_ratio))
    mask_idx = torch.randperm(num_nodes)[:num_mask]
    aug.x = aug.x.clone()
    aug.x[mask_idx] = 0.0
    return aug


def edge_drop_augment(data, drop_ratio=0.1):
    """
    Randomly drop edges (bonds) from the graph.
    Simulates bond breaking — chemically meaningful augmentation.
    """
    import copy
    aug = copy.copy(data)
    num_edges = aug.edge_index.size(1)
    num_drop = max(1, int(num_edges * drop_ratio))
    keep_idx = torch.randperm(num_edges)[num_drop:]
    keep_idx, _ = keep_idx.sort()
    aug.edge_index = aug.edge_index[:, keep_idx]
    if hasattr(aug, 'edge_attr') and aug.edge_attr is not None:
        aug.edge_attr = aug.edge_attr[keep_idx]
    return aug


def subgraph_augment(data, ratio=0.8):
    """
    Extract a connected subgraph by BFS from a random start node.
    Preserves chemical locality — atoms near each other stay together.
    """
    import copy
    aug = copy.copy(data)
    num_nodes = aug.x.size(0)
    if num_nodes <= 2:
        return aug
    
    target_size = max(2, int(num_nodes * ratio))
    
    # BFS from random start
    start = torch.randint(0, num_nodes, (1,)).item()
    edge_index = aug.edge_index
    
    # Build adjacency
    visited = set([start])
    queue = [start]
    while len(visited) < target_size and queue:
        current = queue.pop(0)
        neighbors = edge_index[1][edge_index[0] == current].tolist()
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                queue.append(n)
                if len(visited) >= target_size:
                    break
    
    node_idx = torch.tensor(sorted(visited), dtype=torch.long)
    
    # Remap edges
    node_map = {old.item(): new for new, old in enumerate(node_idx)}
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[node_idx] = True
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    
    new_edge_index = edge_index[:, edge_mask]
    new_edge_index = torch.tensor([
        [node_map[i.item()] for i in new_edge_index[0]],
        [node_map[i.item()] for i in new_edge_index[1]],
    ], dtype=torch.long)
    
    aug.x = aug.x[node_idx]
    aug.edge_index = new_edge_index
    if hasattr(aug, 'edge_attr') and aug.edge_attr is not None:
        aug.edge_attr = aug.edge_attr[edge_mask]
    aug.num_nodes = len(node_idx)
    return aug


def dual_augment(data):
    """
    Generate two augmented views of a molecular graph.
    View 1: atom masking + edge dropping
    View 2: subgraph extraction + atom masking
    """
    v1 = atom_mask_augment(data, mask_ratio=0.15)
    v1 = edge_drop_augment(v1, drop_ratio=0.1)
    
    v2 = subgraph_augment(data, ratio=0.8)
    v2 = atom_mask_augment(v2, mask_ratio=0.1)
    
    return v1, v2


# ============================================================
# Graph encoder (DMPNN)
# ============================================================
class DMPNNEncoder(nn.Module):
    """DMPNN encoder — same as in magnet_v2.py but returns graph embedding."""
    
    def __init__(self, node_dim=36, edge_dim=12, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        self.msg_fns = nn.ModuleList()
        self.update_fns = nn.ModuleList()
        for _ in range(num_layers):
            self.msg_fns.append(nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.update_fns.append(nn.GRUCell(hidden_dim, hidden_dim))
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = data.edge_attr.float() if hasattr(data, 'edge_attr') and data.edge_attr is not None \
            else torch.zeros(edge_index.size(1), 12, device=x.device)
        
        x = F.relu(self.node_proj(x))
        e = self.edge_proj(edge_attr)
        
        src, dst = edge_index[0], edge_index[1]
        
        for msg_fn, update_fn in zip(self.msg_fns, self.update_fns):
            msgs = F.relu(msg_fn(torch.cat([x[src], e], dim=-1)))
            agg = torch.zeros(x.size(0), msgs.size(1), device=x.device, dtype=msgs.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
            x = update_fn(agg, x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling
        graph_repr = global_mean_pool(x, batch)
        return graph_repr


# ============================================================
# Fingerprint encoder (MLP)
# ============================================================
class FingerprintEncoder(nn.Module):
    """MLP encoder for Morgan fingerprints."""
    
    def __init__(self, fp_dim=2048, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = fp_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, fp):
        return self.net(fp)


# ============================================================
# Projection head (shared)
# ============================================================
class ProjectionHead(nn.Module):
    """Projects embeddings to contrastive space."""
    
    def __init__(self, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# DVCL model
# ============================================================
class DVCL(nn.Module):
    """
    Dual-View Contrastive Learning model.
    
    Pre-training: contrastive loss between graph view and fingerprint view.
    Fine-tuning: classifier on top of concatenated graph + fingerprint embeddings.
    """
    
    def __init__(self, node_dim=36, edge_dim=12, fp_dim=2048, hidden_dim=128,
                 proj_dim=64, num_layers=3, dropout=0.1, out_dim=1):
        super().__init__()
        
        # Encoders
        self.graph_encoder = DMPNNEncoder(node_dim, edge_dim, hidden_dim, num_layers, dropout)
        self.fp_encoder = FingerprintEncoder(fp_dim, hidden_dim, num_layers=3, dropout=dropout)
        
        # Projection heads (for contrastive learning)
        self.graph_proj = ProjectionHead(hidden_dim, proj_dim)
        self.fp_proj = ProjectionHead(hidden_dim, proj_dim)
        
        # Classifier (for fine-tuning)
        # Concatenation of graph + fingerprint embeddings
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        
        self.hidden_dim = hidden_dim
    
    def encode_graph(self, data):
        """Encode molecular graph."""
        return self.graph_encoder(data)
    
    def encode_fp(self, fp):
        """Encode Morgan fingerprint."""
        return self.fp_encoder(fp)
    
    def project_graph(self, data):
        """Project graph embedding to contrastive space."""
        h = self.graph_encoder(data)
        return self.graph_proj(h)
    
    def project_fp(self, fp):
        """Project fingerprint embedding to contrastive space."""
        h = self.fp_encoder(fp)
        return self.fp_proj(h)
    
    def forward(self, data, fp):
        """
        Fine-tune forward: concatenate graph + fingerprint embeddings, classify.
        """
        h_graph = self.graph_encoder(data)
        h_fp = self.fp_encoder(fp)
        h = torch.cat([h_graph, h_fp], dim=-1)
        return self.classifier(h).squeeze(-1)
    
    def forward_pretrain(self, data_v1, data_v2, fp):
        """
        Pre-training forward: compute contrastive loss between graph views and fingerprint.
        
        Returns: contrastive loss
        """
        # Graph projections from two augmented views
        z_g1 = self.graph_proj(self.graph_encoder(data_v1))
        z_g2 = self.graph_proj(self.graph_encoder(data_v2))
        
        # Fingerprint projection
        z_fp = self.fp_proj(self.fp_encoder(fp))
        
        return z_g1, z_g2, z_fp


# ============================================================
# Contrastive loss: InfoNCE
# ============================================================
def info_nce_loss(z1, z2, temperature=0.1):
    """
    InfoNCE contrastive loss.
    z1, z2: (batch_size, proj_dim) — positive pairs
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    batch_size = z1.size(0)
    
    # Similarity matrix
    sim = torch.mm(z1, z2.t()) / temperature  # (B, B)
    
    # Labels: diagonal is positive
    labels = torch.arange(batch_size, device=z1.device)
    
    # Cross-entropy loss (symmetric)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)
    
    return (loss_12 + loss_21) / 2


def dvcl_pretrain_loss(z_g1, z_g2, z_fp, temperature=0.1):
    """
    DVCL pre-training loss: three-way contrastive alignment.
    
    Alignments:
    1. Graph view 1 <-> Graph view 2 (augmentation invariance)
    2. Graph view 1 <-> Fingerprint (cross-modal alignment)
    3. Graph view 2 <-> Fingerprint (cross-modal alignment)
    """
    loss_g1_g2 = info_nce_loss(z_g1, z_g2, temperature)
    loss_g1_fp = info_nce_loss(z_g1, z_fp, temperature)
    loss_g2_fp = info_nce_loss(z_g2, z_fp, temperature)
    
    return loss_g1_g2 + loss_g1_fp + loss_g2_fp


# ============================================================
# Fine-tune classifier (standalone, for frozen encoder)
# ============================================================
class DVCLClassifier(nn.Module):
    """
    Classifier head for fine-tuning with frozen DVCL encoder.
    Concatenates graph + fingerprint embeddings.
    """
    
    def __init__(self, hidden_dim=256, out_dim=1, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )
    
    def forward(self, h):
        return self.head(h).squeeze(-1)
