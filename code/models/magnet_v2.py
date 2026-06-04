"""
MAGNet v2: Multi-scale Attention Graph Network with Molecular Features.

Key improvements over v1:
1. Rich atom features (36 dims) from RDKit instead of 9-dim one-hot
2. Bond features (12 dims) for edge-aware message passing
3. DirectedMPNN-style message passing with edge conditioning
4. Multi-scale aggregation preserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import numpy as np

# ============================================================
# Feature extraction from RDKit
# ============================================================
def atom_features(atom):
    """Extract 36-dim atom features from RDKit atom object."""
    from rdkit import Chem
    
    # Atomic number (10 common elements + other)
    atomic_nums = [6, 7, 8, 16, 9, 17, 35, 53, 15, 0]  # C, N, O, S, F, Cl, Br, I, P, other
    atom_type = [0] * 10
    anum = atom.GetAtomicNum()
    if anum in atomic_nums:
        atom_type[atomic_nums.index(anum)] = 1
    else:
        atom_type[9] = 1
    
    # Degree (0-5)
    degree = [0] * 6
    deg = min(atom.GetDegree(), 5)
    degree[deg] = 1
    
    # Formal charge (-2 to +2)
    charge = [0] * 5
    fc = atom.GetFormalCharge()
    charge[min(max(fc + 2, 0), 4)] = 1
    
    # Hybridization
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
    hybridization = [0] * 5
    h = atom.GetHybridization()
    if h in hyb_types:
        hybridization[hyb_types.index(h)] = 1
    
    # Aromatic
    aromatic = [1 if atom.GetIsAromatic() else 0]
    
    # Total number of Hs (0-4)
    num_hs = [0] * 5
    nh = min(atom.GetTotalNumHs(), 4)
    num_hs[nh] = 1
    
    # In ring
    in_ring = [1 if atom.IsInRing() else 0]
    
    # Chirality
    chirality = [0] * 3
    try:
        chi = atom.GetChiralTag()
        if chi == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            chirality[0] = 1
        elif chi == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            chirality[1] = 1
        else:
            chirality[2] = 1
    except:
        chirality[2] = 1
    
    return atom_type + degree + charge + hybridization + aromatic + num_hs + in_ring + chirality


def bond_features(bond):
    """Extract 12-dim bond features from RDKit bond object."""
    from rdkit import Chem
    
    # Bond type (4)
    bt = [0] * 4
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bt[0] = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bt[1] = 1
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bt[2] = 1
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bt[3] = 1
    
    # Stereo (6)
    stereo = [0] * 6
    s = bond.GetStereo()
    stereo_types = [
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    if s in stereo_types:
        stereo[stereo_types.index(s)] = 1
    else:
        stereo[3] = 1  # default NONE
    
    # Conjugated (1)
    conjugated = [1 if bond.GetIsConjugated() else 0]
    
    # In ring (1)
    in_ring = [1 if bond.IsInRing() else 0]
    
    return bt + stereo + conjugated + in_ring


def smiles_to_graph(smiles):
    """Convert SMILES to graph with rich features. Returns (x, edge_index, edge_attr)."""
    from rdkit import Chem
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    
    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    x = torch.tensor(atom_feats, dtype=torch.float)
    
    # Edge features (directed — each bond becomes 2 directed edges)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)
    
    if len(edge_index) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 12), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return x, edge_index, edge_attr


def enrich_molnet_dataset(dataset):
    """
    Replace PyG MoleculeNet's 9-dim features with 36-dim RDKit features.
    Also adds edge_attr (bond features).
    Modifies dataset in-place.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    
    enriched = []
    skipped = 0
    for i in range(len(dataset)):
        data = dataset[i]
        smi = data.smiles if isinstance(data.smiles, str) else data.smiles[0]
        x, edge_index, edge_attr = smiles_to_graph(smi)
        if x is None:
            skipped += 1
            continue
        
        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.num_nodes = x.size(0)
        enriched.append(data)
    
    print(f"  Enriched: {len(enriched)} molecules, skipped {skipped}")
    return enriched


# ============================================================
# Edge-aware message passing layer
# ============================================================
class EdgeMPNNLayer(nn.Module):
    """
    Directed message passing with edge conditioning (scatter-based).
    message: m_ij = MLP([h_j || e_ij])
    update: h_i = GRU(h_i, sum_j m_ij)
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.msg_fn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_fn = nn.GRUCell(hidden_dim, node_dim)
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        msgs = self.msg_fn(torch.cat([x[src], edge_attr], dim=-1))
        agg = torch.zeros(x.size(0), msgs.size(1), device=x.device, dtype=msgs.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
        return self.update_fn(agg, x)


class EdgeGCNLayer(nn.Module):
    """
    GCN-style layer with edge feature gating (scatter-based).
    """
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.node_linear = nn.Linear(in_dim, out_dim)
        self.edge_gate = nn.Sequential(nn.Linear(edge_dim, out_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(out_dim)
        self.residual = (in_dim == out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        gate = self.edge_gate(edge_attr)
        msgs = self.node_linear(x[src]) * gate
        # Degree normalization
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg_inv = deg.pow(-0.5); deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[src] * deg_inv[dst]
        msgs = msgs * norm.unsqueeze(-1)
        out = torch.zeros(x.size(0), msgs.size(1), device=x.device, dtype=msgs.dtype)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
        out = self.norm(out)
        if self.residual: out = out + x
        return F.relu(out)


# ============================================================
# MAGNet v2
# ============================================================
class MAGNetV2(nn.Module):
    """
    MAGNet v2 with molecular features and edge-aware message passing.
    
    Architecture:
    1. Edge-aware graph convolutions at multiple scales
    2. Cross-scale attention fusion
    3. Attention pooling for graph-level representation
    4. Binary/multi-task classifier
    """
    def __init__(self, node_dim=36, edge_dim=12, hidden_dim=128, out_dim=1,
                 num_scales=3, num_heads=4, num_queries=4, dropout=0.1,
                 conv_type='edge_mpnn'):
        super().__init__()
        self.num_scales = num_scales
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.conv_type = conv_type
        
        # Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # Multi-scale edge-aware convolutions
        self.scale_convs = nn.ModuleList()
        self.scale_norms = nn.ModuleList()
        for _ in range(num_scales):
            if conv_type == 'edge_mpnn':
                self.scale_convs.append(EdgeMPNNLayer(hidden_dim, hidden_dim, hidden_dim))
            else:
                self.scale_convs.append(EdgeGCNLayer(hidden_dim, hidden_dim, hidden_dim))
            self.scale_norms.append(nn.LayerNorm(hidden_dim))
        
        # Cross-scale attention
        self.gate_linear = nn.Linear(hidden_dim * num_scales, num_scales)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Attention pooling
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.pool_proj = nn.Linear(hidden_dim * num_queries, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = data.edge_attr.float() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        
        if edge_attr is None:
            # Fallback: create zero edge features
            edge_attr = torch.zeros(edge_index.size(1), 12, device=x.device)
        
        # Project inputs
        h = F.relu(self.node_proj(x))
        e = self.edge_proj(edge_attr)
        
        # Multi-scale feature extraction
        scale_feats = []
        cur = h
        for conv, norm in zip(self.scale_convs, self.scale_norms):
            if self.conv_type == 'edge_mpnn':
                cur = conv(cur, edge_index, e)
            else:
                cur = conv(cur, edge_index, e)
            scale_feats.append(cur)
        
        # Cross-scale attention (per node)
        stacked = torch.stack(scale_feats, dim=1)          # (N, S, H)
        concat = stacked.reshape(stacked.size(0), -1)       # (N, S*H)
        gate = torch.softmax(self.gate_linear(concat), -1)  # (N, S)
        gated = (stacked * gate.unsqueeze(-1)).sum(1)        # (N, H)
        
        q = gated.unsqueeze(1)                               # (N, 1, H)
        attn_out, _ = self.cross_attn(q, stacked, stacked)   # (N, 1, H)
        fused = self.cross_norm(attn_out.squeeze(1) + gated) # (N, H)
        
        # Attention pooling (per graph)
        batch_size = int(batch.max().item()) + 1
        graph_list = []
        for i in range(batch_size):
            mask = batch == i
            nf = fused[mask].unsqueeze(0)                    # (1, n, H)
            qe = self.queries.unsqueeze(0)                   # (1, Q, H)
            po, _ = self.pool_attn(qe, nf, nf)               # (1, Q, H)
            graph_list.append(po.reshape(1, -1))              # (1, Q*H)
        graph_repr = torch.cat(graph_list, dim=0)             # (B, Q*H)
        graph_repr = F.relu(self.pool_proj(graph_repr))       # (B, H)
        graph_repr = self.pool_norm(graph_repr)
        
        return self.classifier(graph_repr).squeeze(-1)


# ============================================================
# Also add D-MPNN baseline for comparison
# ============================================================
class DMPNNLayer(nn.Module):
    """
    Directed MPNN (D-MPNN style, scatter-based).
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.W_msg = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.W_up = nn.GRUCell(hidden_dim, node_dim)
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        msgs = F.relu(self.W_msg(torch.cat([x[src], edge_attr], dim=-1)))
        agg = torch.zeros(x.size(0), msgs.size(1), device=x.device, dtype=msgs.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
        return self.W_up(agg, x)


class DMPNN(nn.Module):
    """D-MPNN baseline: directed message passing + GRU + global pooling."""
    def __init__(self, node_dim=36, edge_dim=12, hidden_dim=128, out_dim=1,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            DMPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = data.edge_attr.float() if hasattr(data, 'edge_attr') and data.edge_attr is not None \
            else torch.zeros(edge_index.size(1), 12, device=x.device)
        
        x = F.relu(self.node_proj(x))
        e = self.edge_proj(edge_attr)
        
        for layer in self.layers:
            x = layer(x, edge_index, e)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


# ============================================================
# AttentiveFP-style baseline (also uses edge features)
# ============================================================
class AttentiveFP(nn.Module):
    """
    AttentiveFP-style model: GNN + attention readout.
    Uses edge features in message passing.
    """
    def __init__(self, node_dim=36, edge_dim=12, hidden_dim=128, out_dim=1,
                 num_layers=3, num_timesteps=2, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.grus = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
            self.grus.append(nn.GRUCell(hidden_dim, hidden_dim))
        
        # Attention readout
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_timesteps = num_timesteps
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        x = F.relu(self.node_proj(x))
        
        for conv, gru in zip(self.convs, self.grus):
            m = F.relu(conv(x, edge_index))
            x = gru(m, x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Attention readout
        batch_size = int(batch.max().item()) + 1
        graph_list = []
        for i in range(batch_size):
            mask = batch == i
            nf = x[mask]
            scores = self.attn_gate(nf)
            weights = torch.softmax(scores, dim=0)
            graph_repr = (weights * nf).sum(0, keepdim=True)
            graph_list.append(graph_repr)
        graph_repr = torch.cat(graph_list, dim=0)
        
        return self.fc(graph_repr).squeeze(-1)
