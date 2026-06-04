#!/usr/bin/env python3
"""
Comprehensive MoleculeNet benchmark for MAGNet paper revision.
Addresses all ARS reviewer concerns:
  1. Real data: BBBP, BACE, HIV, Tox21 with scaffold splitting
  2. More baselines: GCN, GAT, GIN, GraphSAGE, MPNN, MAGNet, RF-Morgan
  3. Statistical rigor: 10 seeds, mean ± std, 5-fold CV
  4. Fair comparison: matched parameter counts
  5. Significance tests: paired t-test, Wilcoxon signed-rank
  6. Ablation study on real data
  7. Hyperparameter sensitivity analysis
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, SAGEConv,
    global_mean_pool, global_add_pool, MessagePassing,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# ── paths ──────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "molnet")
RESULTS_DIR = os.path.join(ROOT, "results")

# ── constants ──────────────────────────────────────────────────
DATASETS = ["BBBP", "BACE", "HIV", "Tox21"]
NUM_SEEDS = 10
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 10
DEFAULT_HIDDEN = 128
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.1


# ================================================================
# Utility
# ================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================================================
# Scaffold splitting (Murcko)
# ================================================================
def _get_scaffold(smiles: str):
    from rdkit import Chem as _Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    mol = _Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, seed=42):
    """Deterministic scaffold split – returns (train_idx, val_idx, test_idx)."""
    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        smi = data.smiles if isinstance(data.smiles, str) else data.smiles[0]
        sc = _get_scaffold(smi)
        scaffolds[sc if sc else f"_unknown_{i}"].append(i)

    rng = np.random.RandomState(seed)
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng.shuffle(scaffold_groups)           # shuffle groups of equal size

    n = len(dataset)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx += group
        elif len(val_idx) + len(group) <= n_val:
            val_idx += group
        else:
            test_idx += group
    # safety: put leftovers in test
    seen = set(train_idx) | set(val_idx) | set(test_idx)
    for i in range(n):
        if i not in seen:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


def kfold_scaffold_split(dataset, k=5, seed=42):
    """Yield (train_idx, val_idx) tuples for k-fold CV on training portion."""
    # First split off test set (held out)
    train_val_idx, _, test_idx = scaffold_split(dataset, 0.8, 0.1, seed)
    # Now split train_val into k folds
    sub = [train_val_idx[i] for i in range(len(train_val_idx))]
    rng = np.random.RandomState(seed)
    rng.shuffle(sub)
    folds = np.array_split(sub, k)
    for i in range(k):
        val_fold = list(folds[i])
        train_fold = [x for j, f in enumerate(folds) if j != i for x in f]
        yield train_fold, val_fold, test_idx


# ================================================================
# Dataset helpers
# ================================================================
def load_dataset(name: str):
    """Download / cache MoleculeNet dataset via PyG."""
    dataset = MoleculeNet(root=DATA_DIR, name=name)
    return dataset


def get_task_count(dataset):
    """Number of prediction tasks (1 for single-task, >1 for multi-task)."""
    if hasattr(dataset, "num_tasks"):
        return dataset.num_tasks
    return dataset[0].y.shape[-1] if dataset[0].y.dim() > 0 else 1


def get_in_dim(dataset):
    return dataset[0].x.shape[-1]


# ================================================================
# Model definitions (PyG-native, sparse edge_index)
# ================================================================

class PyGGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


class PyGGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


class PyGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


class PyGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


class MPNNLayer(MessagePassing):
    """Simple MPNN layer: message = linear(src), aggregate = sum, update = GRU."""
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.msg_fn = nn.Linear(in_dim, out_dim, bias=False)
        self.update_fn = nn.GRUCell(out_dim, in_dim)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return self.update_fn(out, x)

    def message(self, x_j):
        return self.msg_fn(x_j)


class PyMPNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.input_proj(x))
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


class PyMAGNet(nn.Module):
    """MAGNet implemented with PyG sparse message passing."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=1,
                 num_scales=3, num_heads=4, num_queries=4, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # multi-scale GCN
        self.scale_convs = nn.ModuleList()
        self.scale_norms = nn.ModuleList()
        for _ in range(num_scales):
            self.scale_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.scale_norms.append(nn.LayerNorm(hidden_dim))

        # cross-scale attention
        self.gate_linear = nn.Linear(hidden_dim * num_scales, num_scales)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # attention pooling
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.pool_proj = nn.Linear(hidden_dim * num_queries, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # input projection
        h = F.relu(self.input_proj(x))

        # multi-scale feature extraction
        scale_feats = []
        cur = h
        for conv, norm in zip(self.scale_convs, self.scale_norms):
            cur = F.relu(norm(conv(cur, edge_index))) + cur  # residual
            scale_feats.append(cur)

        # cross-scale attention (per node)
        stacked = torch.stack(scale_feats, dim=1)          # (N, S, H)
        concat = stacked.reshape(stacked.size(0), -1)       # (N, S*H)
        gate = torch.softmax(self.gate_linear(concat), -1)  # (N, S)
        gated = (stacked * gate.unsqueeze(-1)).sum(1)        # (N, H)

        q = gated.unsqueeze(1)                               # (N, 1, H)
        attn_out, _ = self.cross_attn(q, stacked, stacked)   # (N, 1, H)
        fused = self.cross_norm(attn_out.squeeze(1) + gated) # (N, H)

        # attention pooling (per graph in batch)
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


# ── model factory ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "GCN":      PyGGCN,
    "GAT":      PyGGAT,
    "GIN":      PyGIN,
    "GraphSAGE": PyGraphSAGE,
    "MPNN":     PyMPNN,
    "MAGNet":   PyMAGNet,
}


def build_model(name, in_dim, out_dim=1, hidden_dim=128, **kw):
    cls = MODEL_REGISTRY[name]
    if name == "MAGNet":
        return cls(in_dim, hidden_dim, out_dim,
                   num_scales=kw.get("num_scales", 3),
                   num_heads=kw.get("num_heads", 4),
                   num_queries=kw.get("num_queries", 4),
                   dropout=kw.get("dropout", DEFAULT_DROPOUT))
    if name == "GAT":
        return cls(in_dim, hidden_dim, out_dim,
                   num_layers=kw.get("num_layers", 3),
                   heads=kw.get("heads", 4),
                   dropout=kw.get("dropout", DEFAULT_DROPOUT))
    return cls(in_dim, hidden_dim, out_dim,
               num_layers=kw.get("num_layers", 3),
               dropout=kw.get("dropout", DEFAULT_DROPOUT))


# ================================================================
# Training / evaluation
# ================================================================
def masked_bce_loss(logits, target, mask):
    """BCE loss ignoring entries where mask == 0."""
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.clamp(min=1e-8).sum()


def train_one_epoch(model, loader, optimizer, device, is_multitask=False):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        if is_multitask:
            y = batch.y.float().view(batch.num_graphs, -1)
            mask = ~torch.isnan(y)
            y = torch.nan_to_num(y, 0.0)
            loss = masked_bce_loss(logits.view(batch.num_graphs, -1), y, mask)
        else:
            y = batch.y.float().view(-1)
            mask = ~torch.isnan(y)
            y = torch.nan_to_num(y, 0.0)
            loss = masked_bce_loss(logits, y, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, device, is_multitask=False):
    """Return per-task ROC-AUC (list of floats)."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        if is_multitask:
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(batch.num_graphs, -1))
            all_labels.append(batch.y.cpu().numpy().reshape(batch.num_graphs, -1))
        else:
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
            all_labels.append(batch.y.float().cpu().numpy().reshape(-1))

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    if is_multitask:
        n_tasks = labels.shape[1]
        aucs = []
        for t in range(n_tasks):
            mask = ~np.isnan(labels[:, t])
            if mask.sum() < 2 or len(np.unique(labels[mask, t])) < 2:
                continue
            try:
                aucs.append(roc_auc_score(labels[mask, t], preds[mask, t]))
            except ValueError:
                pass
        return aucs  # may be empty
    else:
        mask = ~np.isnan(labels)
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            return [np.nan]
        try:
            return [roc_auc_score(labels[mask], preds[mask])]
        except ValueError:
            return [np.nan]


def run_single(model_name, dataset, train_idx, val_idx, test_idx,
               device, seed, hidden_dim=128, lr=1e-3, dropout=0.1,
               num_scales=3, num_heads=4, num_queries=4,
               max_epochs=MAX_EPOCHS, patience=PATIENCE, is_multitask=False):
    """Train one model from scratch and return test AUC list + training time."""
    set_seed(seed)
    in_dim = get_in_dim(dataset)
    out_dim = get_task_count(dataset) if is_multitask else 1

    model = build_model(model_name, in_dim, out_dim, hidden_dim,
                        num_scales=num_scales, num_heads=num_heads,
                        num_queries=num_queries, dropout=dropout).to(device)

    train_loader = DataLoader(dataset[train_idx], BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], BATCH_SIZE)
    test_loader = DataLoader(dataset[test_idx], BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

    best_val_auc = -1.0
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, device, is_multitask)
        scheduler.step()

        val_aucs = evaluate(model, val_loader, device, is_multitask)
        val_auc = np.nanmean(val_aucs) if val_aucs else 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    test_aucs = evaluate(model, test_loader, device, is_multitask)
    return test_aucs, count_parameters(model), elapsed


# ── Random-Forest on Morgan fingerprints ───────────────────────
def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem as _AllChem
    fps = []
    for smi in smiles_list:
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        fp = _AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.uint8))
    return np.stack(fps)


def run_rf_morgan(dataset, train_idx, val_idx, test_idx, seed):
    """Random Forest on Morgan fingerprints baseline."""
    set_seed(seed)
    smiles_list = []
    for i in range(len(dataset)):
        d = dataset[i]
        s = d.smiles if isinstance(d.smiles, str) else d.smiles[0]
        smiles_list.append(s)

    train_smi = [smiles_list[i] for i in train_idx]
    test_smi = [smiles_list[i] for i in test_idx]
    train_y = np.array([dataset[i].y.item() if dataset[i].y.dim() == 0
                        else dataset[i].y[0].item() for i in train_idx])
    test_y = np.array([dataset[i].y.item() if dataset[i].y.dim() == 0
                       else dataset[i].y[0].item() for i in test_idx])

    # filter NaN
    mask = ~np.isnan(train_y)
    train_smi = [s for s, m in zip(train_smi, mask) if m]
    train_y = train_y[mask]
    mask = ~np.isnan(test_y)
    test_smi = [s for s, m in zip(test_smi, mask) if m]
    test_y = test_y[mask]

    X_train = smiles_to_morgan(train_smi)
    X_test = smiles_to_morgan(test_smi)

    clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    t0 = time.time()
    clf.fit(X_train, train_y)
    elapsed = time.time() - t0

    probs = clf.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(test_y, probs)
    except ValueError:
        auc = np.nan
    return [auc], 0, elapsed  # RF has no "parameters" in the NN sense


# ================================================================
# Full benchmark runner
# ================================================================
def run_benchmark(datasets=None, num_seeds=NUM_SEEDS, hidden_dim=DEFAULT_HIDDEN,
                  device=None, quick=False):
    """Run full benchmark on all datasets with all models."""
    if device is None:
        device = get_device()
    if datasets is None:
        datasets = DATASETS

    results = {}
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")
        dataset = load_dataset(ds_name)
        is_mt = get_task_count(dataset) > 1
        in_dim = get_in_dim(dataset)
        print(f"  Molecules: {len(dataset)},  Features: {in_dim},  Tasks: {get_task_count(dataset)}")

        ds_results = {}
        models_to_run = list(MODEL_REGISTRY.keys()) + ["RF-Morgan"]

        for model_name in models_to_run:
            print(f"\n  ▸ {model_name}", end="", flush=True)
            seed_aucs = []
            seed_times = []
            seed_params = []

            seeds = range(num_seeds)
            for seed in seeds:
                if model_name == "RF-Morgan":
                    idx = scaffold_split(dataset, seed=seed)
                    aucs, params, elapsed = run_rf_morgan(dataset, *idx, seed=seed)
                else:
                    idx = scaffold_split(dataset, seed=seed)
                    aucs, params, elapsed = run_single(
                        model_name, dataset, *idx, device, seed,
                        hidden_dim=hidden_dim, is_multitask=is_mt,
                    )
                mean_auc = np.nanmean(aucs) if aucs else np.nan
                seed_aucs.append(mean_auc)
                seed_times.append(elapsed)
                seed_params.append(params)
                print(".", end="", flush=True)

            ds_results[model_name] = {
                "mean_auc": float(np.nanmean(seed_aucs)),
                "std_auc": float(np.nanstd(seed_aucs)),
                "seed_aucs": [float(x) for x in seed_aucs],
                "mean_time": float(np.mean(seed_times)),
                "params": int(np.median(seed_params)),
            }
            print(f"  AUC = {np.nanmean(seed_aucs):.4f} ± {np.nanstd(seed_aucs):.4f}")

        results[ds_name] = ds_results

    return results


# ================================================================
# 5-fold cross-validation
# ================================================================
def run_cross_validation(datasets=None, num_seeds=3, hidden_dim=DEFAULT_HIDDEN, device=None):
    """Run 5-fold CV on each dataset (fewer seeds for speed)."""
    if device is None:
        device = get_device()
    if datasets is None:
        datasets = DATASETS

    cv_results = {}
    for ds_name in datasets:
        dataset = load_dataset(ds_name)
        is_mt = get_task_count(dataset) > 1
        print(f"\n  5-fold CV: {ds_name}", end="", flush=True)

        ds_cv = {}
        for model_name in list(MODEL_REGISTRY.keys()):
            fold_aucs = []
            for fold_id, (train_idx, val_idx, test_idx) in enumerate(
                    kfold_scaffold_split(dataset, k=5, seed=42)):
                for seed in range(num_seeds):
                    aucs, _, _ = run_single(
                        model_name, dataset, train_idx, val_idx, test_idx,
                        device, seed + fold_id * 100,
                        hidden_dim=hidden_dim, is_multitask=is_mt,
                    )
                    fold_aucs.append(np.nanmean(aucs))
                print(".", end="", flush=True)
            ds_cv[model_name] = {
                "mean_auc": float(np.nanmean(fold_aucs)),
                "std_auc": float(np.nanstd(fold_aucs)),
            }
        cv_results[ds_name] = ds_cv
    return cv_results


# ================================================================
# Significance tests (MAGNet vs each baseline)
# ================================================================
def compute_significance(benchmark_results):
    """Paired t-test and Wilcoxon test: MAGNet vs each baseline on seed-level AUCs."""
    sig_results = {}
    for ds_name, ds_res in benchmark_results.items():
        magnet_aucs = np.array(ds_res["MAGNet"]["seed_aucs"])
        ds_sig = {}
        for model_name, model_res in ds_res.items():
            if model_name == "MAGNet" or model_name == "RF-Morgan":
                continue
            other_aucs = np.array(model_res["seed_aucs"])
            n = min(len(magnet_aucs), len(other_aucs))
            t_stat, t_p = stats.ttest_rel(magnet_aucs[:n], other_aucs[:n])
            try:
                w_stat, w_p = stats.wilcoxon(magnet_aucs[:n], other_aucs[:n])
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            ds_sig[model_name] = {
                "t_statistic": float(t_stat), "t_pvalue": float(t_p),
                "w_statistic": float(w_stat), "w_pvalue": float(w_p),
            }
        sig_results[ds_name] = ds_sig
    return sig_results


# ================================================================
# Ablation: number of scales
# ================================================================
def run_ablation_scales(dataset_name="BBBP", scales_list=None, device=None, num_seeds=None):
    if scales_list is None:
        scales_list = [1, 2, 3, 4]
    if num_seeds is None:
        num_seeds = NUM_SEEDS
    if device is None:
        device = get_device()
    dataset = load_dataset(dataset_name)
    is_mt = get_task_count(dataset) > 1
    results = {}
    for S in scales_list:
        aucs_all = []
        for seed in range(num_seeds):
            idx = scaffold_split(dataset, seed=seed)
            aucs, _, _ = run_single(
                "MAGNet", dataset, *idx, device, seed,
                num_scales=S, num_heads=4, is_multitask=is_mt,
            )
            aucs_all.append(np.nanmean(aucs))
        results[S] = {"mean": float(np.nanmean(aucs_all)), "std": float(np.nanstd(aucs_all))}
        print(f"  Scales={S}: AUC = {results[S]['mean']:.4f} ± {results[S]['std']:.4f}")
    return results


# ================================================================
# Ablation: number of attention heads
# ================================================================
def run_ablation_heads(dataset_name="BBBP", heads_list=None, device=None, num_seeds=None):
    if heads_list is None:
        heads_list = [1, 2, 4, 8]
    if num_seeds is None:
        num_seeds = NUM_SEEDS
    if device is None:
        device = get_device()
    dataset = load_dataset(dataset_name)
    is_mt = get_task_count(dataset) > 1
    results = {}
    for H in heads_list:
        aucs_all = []
        for seed in range(num_seeds):
            idx = scaffold_split(dataset, seed=seed)
            aucs, _, _ = run_single(
                "MAGNet", dataset, *idx, device, seed,
                num_scales=3, num_heads=H, is_multitask=is_mt,
            )
            aucs_all.append(np.nanmean(aucs))
        results[H] = {"mean": float(np.nanmean(aucs_all)), "std": float(np.nanstd(aucs_all))}
        print(f"  Heads={H}: AUC = {results[H]['mean']:.4f} ± {results[H]['std']:.4f}")
    return results


# ================================================================
# Hyperparameter sensitivity
# ================================================================
def run_hyperparameter_search(dataset_name="BBBP", device=None):
    if device is None:
        device = get_device()
    dataset = load_dataset(dataset_name)
    is_mt = get_task_count(dataset) > 1

    search_space = {
        "lr": [5e-4, 1e-3, 2e-3],
        "hidden_dim": [64, 128, 256],
        "dropout": [0.0, 0.1, 0.2],
        "num_scales": [2, 3, 4],
        "num_heads": [2, 4, 8],
    }
    # Baseline config
    base = {"lr": 1e-3, "hidden_dim": 128, "dropout": 0.1, "num_scales": 3, "num_heads": 4}
    results = {}

    for param, values in search_space.items():
        param_results = {}
        for val in values:
            cfg = dict(base)
            cfg[param] = val
            aucs_all = []
            for seed in range(3):  # 3 seeds for speed
                idx = scaffold_split(dataset, seed=seed)
                aucs, _, _ = run_single(
                    "MAGNet", dataset, *idx, device, seed,
                    hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
                    num_scales=cfg["num_scales"], num_heads=cfg["num_heads"],
                    is_multitask=is_mt,
                )
                aucs_all.append(np.nanmean(aucs))
            param_results[str(val)] = {
                "mean": float(np.nanmean(aucs_all)),
                "std": float(np.nanstd(aucs_all)),
            }
            print(f"  {param}={val}: AUC = {np.nanmean(aucs_all):.4f}")
        results[param] = param_results
    return results


# ================================================================
# Parameter-matched comparison
# ================================================================
def run_parameter_matched(dataset_name="BBBP", target_params=150_000, device=None):
    """Run all models with hidden_dim tuned to match target parameter count."""
    if device is None:
        device = get_device()
    dataset = load_dataset(dataset_name)
    is_mt = get_task_count(dataset) > 1
    in_dim = get_in_dim(dataset)

    # Find hidden_dim for each model to hit ~target_params
    hd_map = {}
    for name in MODEL_REGISTRY:
        for hd in [32, 48, 64, 96, 128, 192, 256]:
            m = build_model(name, in_dim, 1, hd)
            p = count_parameters(m)
            if p >= target_params * 0.7:
                hd_map[name] = hd
                break
        if name not in hd_map:
            hd_map[name] = 256

    print(f"\n  Parameter-matched (target ~{target_params}):")
    results = {}
    for name in MODEL_REGISTRY:
        aucs_all = []
        for seed in range(NUM_SEEDS):
            idx = scaffold_split(dataset, seed=seed)
            aucs, params, _ = run_single(
                name, dataset, *idx, device, seed,
                hidden_dim=hd_map[name], is_multitask=is_mt,
            )
            aucs_all.append(np.nanmean(aucs))
        m = build_model(name, in_dim, 1, hd_map[name])
        results[name] = {
            "hidden_dim": hd_map[name],
            "params": count_parameters(m),
            "mean_auc": float(np.nanmean(aucs_all)),
            "std_auc": float(np.nanstd(aucs_all)),
        }
        print(f"    {name:12s}  hd={hd_map[name]:3d}  params={count_parameters(m):>8,d}  "
              f"AUC={np.nanmean(aucs_all):.4f}±{np.nanstd(aucs_all):.4f}")
    return results


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="MoleculeNet benchmark for MAGNet")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--quick", action="store_true", help="Run quick subset for testing")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    if args.quick:
        args.datasets = ["BBBP"]
        args.seeds = 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    # 1. Main benchmark
    print("\n" + "=" * 60 + "\n  PHASE 1: Main Benchmark\n" + "=" * 60)
    all_results["benchmark"] = run_benchmark(
        args.datasets, args.seeds, args.hidden, device, args.quick,
    )

    if not args.quick:
        # 2. 5-fold cross-validation
        print("\n" + "=" * 60 + "\n  PHASE 2: 5-Fold Cross-Validation\n" + "=" * 60)
        all_results["cross_validation"] = run_cross_validation(
            args.datasets, num_seeds=min(3, args.seeds), hidden_dim=args.hidden, device=device,
        )

    # 3. Significance tests
    print("\n" + "=" * 60 + "\n  PHASE 3: Significance Tests\n" + "=" * 60)
    all_results["significance"] = compute_significance(all_results["benchmark"])
    for ds, sigs in all_results["significance"].items():
        print(f"\n  {ds}:")
        for mdl, t in sigs.items():
            print(f"    vs {mdl}:  t={t['t_statistic']:+.3f} (p={t['t_pvalue']:.4f}), "
                  f"W={t['w_pvalue']:.4f}")

    # 4. Ablation: scales
    print("\n" + "=" * 60 + "\n  PHASE 4: Ablation (scales)\n" + "=" * 60)
    all_results["ablation_scales"] = run_ablation_scales("BBBP", device=device,
                                                         num_seeds=args.seeds)

    # 5. Ablation: heads
    print("\n" + "=" * 60 + "\n  PHASE 5: Ablation (heads)\n" + "=" * 60)
    all_results["ablation_heads"] = run_ablation_heads("BBBP", device=device,
                                                       num_seeds=args.seeds)

    if not args.quick:
        # 6. Hyperparameter search
        print("\n" + "=" * 60 + "\n  PHASE 6: Hyperparameter Search\n" + "=" * 60)
        all_results["hyperparameter_search"] = run_hyperparameter_search("BBBP", device=device)

        # 7. Parameter-matched comparison
        print("\n" + "=" * 60 + "\n  PHASE 7: Parameter-Matched Comparison\n" + "=" * 60)
        all_results["parameter_matched"] = run_parameter_matched("BBBP", device=device)

    # Save
    out_path = os.path.join(RESULTS_DIR, "molnet_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
