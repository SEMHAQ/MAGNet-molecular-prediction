#!/usr/bin/env python3
"""
DVCL: Dual-View Contrastive Learning for Molecular Property Prediction.

Full pipeline:
1. Load MoleculeNet dataset with enriched features
2. Pre-train with contrastive loss (graph-fingerprint alignment)
3. Fine-tune on downstream task
4. Compare with baselines (DMPNN, GCN, GAT, etc.)

Usage:
    python -m code.experiments.dvcl_pipeline --dataset BBBP --seeds 3
    python -m code.experiments.dvcl_pipeline --dataset BACE --seeds 3
    python -m code.experiments.dvcl_pipeline --dataset HIV --seeds 3
    python -m code.experiments.dvcl_pipeline --dataset Tox21 --seeds 3
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, mean_absolute_error
from collections import defaultdict

from code.models.dvcl import (
    DVCL, DMPNNEncoder, FingerprintEncoder, ProjectionHead,
    morgan_counts, dual_augment, dvcl_pretrain_loss, info_nce_loss,
    atom_mask_augment, edge_drop_augment,
)
from code.models.magnet_v2 import enrich_molnet_dataset, DMPNN

# Baselines
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool


# ============================================================
# Baseline models (same as bench_fast.py)
# ============================================================
class GCN(nn.Module):
    def __init__(self, in_dim=36, hidden=128, out_dim=1, dropout=0.1):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden); self.c2 = GCNConv(hidden, hidden); self.c3 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden, out_dim); self.dp = dropout
    def forward(self, d):
        x, ei, b = d.x.float(), d.edge_index, d.batch
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, self.dp, training=self.training)
        x = F.relu(self.c2(x, ei)); x = F.dropout(x, self.dp, training=self.training)
        x = F.relu(self.c3(x, ei))
        return self.fc(global_mean_pool(x, b)).squeeze(-1)

class GAT(nn.Module):
    def __init__(self, in_dim=36, hidden=128, out_dim=1, dropout=0.1):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden//4, heads=4); self.c2 = GATConv(hidden, hidden//4, heads=4)
        self.fc = nn.Linear(hidden, out_dim); self.dp = dropout
    def forward(self, d):
        x, ei, b = d.x.float(), d.edge_index, d.batch
        x = F.elu(self.c1(x, ei)); x = F.dropout(x, self.dp, training=self.training)
        x = F.elu(self.c2(x, ei))
        return self.fc(global_mean_pool(x, b)).squeeze(-1)

class GIN(nn.Module):
    def __init__(self, in_dim=36, hidden=128, out_dim=1, dropout=0.1):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.c1 = GINConv(nn1); self.c2 = GINConv(nn2)
        self.fc = nn.Linear(hidden, out_dim); self.dp = dropout
    def forward(self, d):
        x, ei, b = d.x.float(), d.edge_index, d.batch
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, self.dp, training=self.training)
        x = F.relu(self.c2(x, ei))
        return self.fc(global_mean_pool(x, b)).squeeze(-1)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim=36, hidden=128, out_dim=1, dropout=0.1):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden); self.c2 = SAGEConv(hidden, hidden)
        self.fc = nn.Linear(hidden, out_dim); self.dp = dropout
    def forward(self, d):
        x, ei, b = d.x.float(), d.edge_index, d.batch
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, self.dp, training=self.training)
        x = F.relu(self.c2(x, ei))
        return self.fc(global_mean_pool(x, b)).squeeze(-1)


# ============================================================
# Data loading
# ============================================================
def load_dataset(name, root='/tmp/molnet'):
    """Load MoleculeNet dataset with enriched features."""
    from torch_geometric.datasets import MoleculeNet
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    
    ds = MoleculeNet(root=root, name=name)
    print(f"  Raw: {len(ds)} molecules, features: {ds.num_features}")
    
    # Enrich features
    data_list = enrich_molnet_dataset(ds)
    print(f"  Enriched: {len(data_list)} molecules")
    
    return data_list


def extract_fingerprints(data_list, radius=2, n_bits=2048):
    """Extract Morgan fingerprints for all molecules."""
    fps = []
    for d in data_list:
        smi = d.smiles if isinstance(d.smiles, str) else d.smiles[0]
        fp = morgan_counts(smi, radius, n_bits)
        fps.append(fp)
    return np.array(fps, dtype=np.float32)


def make_batch(data_list, indices, device):
    """Create a batch from data_list at given indices."""
    subset = [data_list[i] for i in indices]
    batch = Batch.from_data_list(subset)
    return batch.to(device)


# ============================================================
# Pre-training
# ============================================================
def pretrain_dvcl(model, data_list, fps_tensor, epochs=50, batch_size=128,
                  lr=1e-3, temperature=0.1, device='cuda'):
    """
    Pre-train DVCL with dual-view contrastive learning.
    Uses ALL molecules (ignoring labels) for self-supervised learning.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(data_list)
    indices = np.arange(n)
    
    print(f"\n  === DVCL Pre-training ({epochs} epochs, {n} molecules) ===")
    
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0
        n_batches = 0
        
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:
                continue
            
            # Create augmented views
            subset = [data_list[i] for i in idx]
            v1_list, v2_list = [], []
            for d in subset:
                a1, a2 = dual_augment(d)
                v1_list.append(a1)
                v2_list.append(a2)
            
            batch_v1 = Batch.from_data_list(v1_list).to(device)
            batch_v2 = Batch.from_data_list(v2_list).to(device)
            fp_batch = fps_tensor[idx].to(device)
            
            # Forward
            z_g1, z_g2, z_fp = model.forward_pretrain(batch_v1, batch_v2, fp_batch)
            loss = dvcl_pretrain_loss(z_g1, z_g2, z_fp, temperature)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
    
    return model


# ============================================================
# Fine-tuning
# ============================================================
def finetune_and_eval(model, data_list, fps_np, task_type, n_tasks,
                      seed=42, epochs=100, batch_size=64, lr=1e-3, device='cuda',
                      freeze_encoder=False):
    """
    Fine-tune DVCL on downstream task and evaluate.
    
    Returns: dict with train/val/test metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n = len(data_list)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Split: 80/10/10
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    fps_tensor = torch.tensor(fps_np, dtype=torch.float32)
    
    # Get labels
    all_y = []
    for d in data_list:
        y = d.y
        if y.dim() == 0:
            all_y.append(y.item())
        else:
            all_y.append(y.numpy())
    all_y = np.array(all_y)
    
    # Determine if binary classification or regression
    is_binary = (task_type == 'classification')
    
    # Freeze encoder if requested
    if freeze_encoder:
        for param in model.graph_encoder.parameters():
            param.requires_grad = False
        for param in model.fp_encoder.parameters():
            param.requires_grad = False
    
    # Only train classifier (and optionally encoders)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if is_binary else 'min',
                                                            patience=10, factor=0.5)
    
    best_val_metric = -float('inf') if is_binary else float('inf')
    best_test_metric = None
    patience_counter = 0
    max_patience = 20
    
    model.train()
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        np.random.shuffle(train_idx)
        train_loss = 0
        n_train_batches = 0
        
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start:start + batch_size]
            if len(idx) == 0:
                continue
            
            batch = make_batch(data_list, idx, device)
            fp_batch = fps_tensor[idx].to(device)
            
            # Get labels
            if is_binary:
                labels = []
                for i in idx:
                    y = data_list[i].y
                    if y.dim() == 0:
                        labels.append(y.item())
                    else:
                        # Multi-task: use first task
                        labels.append(y[0].item() if y.numel() > 0 else 0)
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            else:
                labels = []
                for i in idx:
                    y = data_list[i].y
                    if y.dim() == 0:
                        labels.append(y.item())
                    else:
                        labels.append(y[0].item() if y.numel() > 0 else 0)
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            
            # Handle NaN labels
            valid_mask = ~torch.isnan(labels)
            if valid_mask.sum() == 0:
                continue
            
            out = model(batch, fp_batch)
            
            if is_binary:
                # Binary cross-entropy with class weighting
                pos_weight = (1 - labels[valid_mask]).sum() / (labels[valid_mask].sum() + 1e-6)
                loss = F.binary_cross_entropy_with_logits(
                    out[valid_mask], labels[valid_mask],
                    pos_weight=pos_weight.clamp(0.1, 10.0)
                )
            else:
                loss = F.mse_loss(out[valid_mask], labels[valid_mask])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
        
        # Validation
        val_metric = evaluate(model, data_list, fps_tensor, val_idx, is_binary, device)
        test_metric = evaluate(model, data_list, fps_tensor, test_idx, is_binary, device)
        
        scheduler.step(val_metric)
        
        # Early stopping
        improved = (val_metric > best_val_metric + 1e-4) if is_binary else (val_metric < best_val_metric - 1e-4)
        if improved:
            best_val_metric = val_metric
            best_test_metric = test_metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            break
    
    return {
        'val': best_val_metric,
        'test': best_test_metric,
        'epochs': epoch,
    }


def evaluate(model, data_list, fps_tensor, indices, is_binary, device, batch_size=128):
    """Evaluate model on a set of indices."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) == 0:
                continue
            
            batch = make_batch(data_list, idx, device)
            fp_batch = fps_tensor[idx].to(device)
            
            out = model(batch, fp_batch)
            
            # Get labels
            labels = []
            for i in idx:
                y = data_list[i].y
                if y.dim() == 0:
                    labels.append(y.item())
                else:
                    labels.append(y[0].item() if y.numel() > 0 else 0)
            
            all_preds.extend(out.cpu().numpy().tolist())
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Handle NaN
    valid = ~np.isnan(all_labels)
    if valid.sum() == 0:
        return 0.0 if is_binary else float('inf')
    
    preds = all_preds[valid]
    labels = all_labels[valid]
    
    if is_binary:
        try:
            probs = 1 / (1 + np.exp(-preds))  # sigmoid
            return roc_auc_score(labels, probs)
        except:
            return 0.5
    else:
        return mean_absolute_error(labels, preds)


# ============================================================
# Baseline training (DMPNN only, for comparison)
# ============================================================
def train_baseline(model_class, data_list, task_type, n_tasks,
                   seed=42, epochs=100, batch_size=64, lr=1e-3, device='cuda',
                   model_kwargs=None):
    """Train a baseline model and return test metric."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if model_kwargs is None:
        model_kwargs = {}
    
    model = model_class(**model_kwargs).to(device)
    
    n = len(data_list)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    is_binary = (task_type == 'classification')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if is_binary else 'min',
                                                            patience=10, factor=0.5)
    
    best_val_metric = -float('inf') if is_binary else float('inf')
    best_test_metric = None
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_idx)
        
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start:start + batch_size]
            if len(idx) == 0:
                continue
            
            batch = make_batch(data_list, idx, device)
            
            labels = []
            for i in idx:
                y = data_list[i].y
                if y.dim() == 0:
                    labels.append(y.item())
                else:
                    labels.append(y[0].item() if y.numel() > 0 else 0)
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
            
            valid_mask = ~torch.isnan(labels)
            if valid_mask.sum() == 0:
                continue
            
            out = model(batch)
            
            if is_binary:
                pos_weight = (1 - labels[valid_mask]).sum() / (labels[valid_mask].sum() + 1e-6)
                loss = F.binary_cross_entropy_with_logits(
                    out[valid_mask], labels[valid_mask],
                    pos_weight=pos_weight.clamp(0.1, 10.0)
                )
            else:
                loss = F.mse_loss(out[valid_mask], labels[valid_mask])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate
        val_metric = eval_baseline(model, data_list, val_idx, is_binary, device)
        test_metric = eval_baseline(model, data_list, test_idx, is_binary, device)
        
        scheduler.step(val_metric)
        
        improved = (val_metric > best_val_metric + 1e-4) if is_binary else (val_metric < best_val_metric - 1e-4)
        if improved:
            best_val_metric = val_metric
            best_test_metric = test_metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            break
    
    return best_test_metric


def eval_baseline(model, data_list, indices, is_binary, device, batch_size=128):
    """Evaluate baseline model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) == 0:
                continue
            
            batch = make_batch(data_list, idx, device)
            out = model(batch)
            
            labels = []
            for i in idx:
                y = data_list[i].y
                if y.dim() == 0:
                    labels.append(y.item())
                else:
                    labels.append(y[0].item() if y.numel() > 0 else 0)
            
            all_preds.extend(out.cpu().numpy().tolist())
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    valid = ~np.isnan(all_labels)
    if valid.sum() == 0:
        return 0.0 if is_binary else float('inf')
    
    preds = all_preds[valid]
    labels = all_labels[valid]
    
    if is_binary:
        try:
            probs = 1 / (1 + np.exp(-preds))
            return roc_auc_score(labels, probs)
        except:
            return 0.5
    else:
        return mean_absolute_error(labels, preds)


# ============================================================
# Main pipeline
# ============================================================
DATASET_CONFIG = {
    'BBBP':   {'task_type': 'classification', 'n_tasks': 1},
    'BACE':   {'task_type': 'classification', 'n_tasks': 1},
    'HIV':    {'task_type': 'classification', 'n_tasks': 1},
    'Tox21':  {'task_type': 'classification', 'n_tasks': 12},
    'ToxCast':{'task_type': 'classification', 'n_tasks': 617},
    'SIDER':  {'task_type': 'classification', 'n_tasks': 27},
    'ClinTox':{'task_type': 'classification', 'n_tasks': 2},
    'ESOL':   {'task_type': 'regression',     'n_tasks': 1},
    'FreeSolv':{'task_type': 'regression',    'n_tasks': 1},
    'Lipophilicity':{'task_type': 'regression','n_tasks': 1},
}


def main():
    parser = argparse.ArgumentParser(description='DVCL Pipeline')
    parser.add_argument('--dataset', type=str, default='BBBP', choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pre-training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=100, help='Fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--no_pretrain', action='store_true', help='Skip pre-training (ablation)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    config = DATASET_CONFIG[args.dataset]
    task_type = config['task_type']
    n_tasks = config['n_tasks']
    
    print(f"Device: {device}, Dataset: {args.dataset}, Seeds: {args.seeds}")
    print(f"Task: {task_type}, Tasks: {n_tasks}")
    print(f"Pre-train epochs: {args.pretrain_epochs}, Fine-tune epochs: {args.finetune_epochs}")
    
    # Load data
    data_list = load_dataset(args.dataset)
    fps_np = extract_fingerprints(data_list, radius=2, n_bits=2048)
    print(f"  Fingerprints shape: {fps_np.shape}")
    
    # ============================================================
    # Run DVCL with pre-training
    # ============================================================
    dvcl_results = []
    dvcl_no_pretrain_results = []
    baseline_results = defaultdict(list)
    
    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed+1}/{args.seeds}")
        print(f"{'='*60}")
        
        # --- DVCL with pre-training ---
        if not args.no_pretrain:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = DVCL(
                node_dim=36, edge_dim=12, fp_dim=2048,
                hidden_dim=args.hidden_dim, out_dim=n_tasks,
            ).to(device)
            
            t0 = time.time()
            model = pretrain_dvcl(
                model, data_list, torch.tensor(fps_np, dtype=torch.float32),
                epochs=args.pretrain_epochs, batch_size=args.batch_size,
                lr=args.lr, device=device,
            )
            pretrain_time = time.time() - t0
            
            # Fine-tune (unfreeze encoder)
            result = finetune_and_eval(
                model, data_list, fps_np, task_type, n_tasks,
                seed=seed, epochs=args.finetune_epochs, batch_size=args.batch_size,
                lr=args.lr, device=device, freeze_encoder=False,
            )
            dvcl_results.append(result['test'])
            print(f"  DVCL (pretrained): test={result['test']:.4f}, pretrain_time={pretrain_time:.0f}s")
        
        # --- DVCL without pre-training (ablation) ---
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model_no_pt = DVCL(
            node_dim=36, edge_dim=12, fp_dim=2048,
            hidden_dim=args.hidden_dim, out_dim=n_tasks,
        ).to(device)
        
        result_no_pt = finetune_and_eval(
            model_no_pt, data_list, fps_np, task_type, n_tasks,
            seed=seed, epochs=args.finetune_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, freeze_encoder=False,
        )
        dvcl_no_pretrain_results.append(result_no_pt['test'])
        print(f"  DVCL (no pretrain): test={result_no_pt['test']:.4f}")
        
        # --- Baselines ---
        baselines = {
            'DMPNN': (DMPNN, {'node_dim': 36, 'edge_dim': 12, 'hidden_dim': args.hidden_dim, 'out_dim': n_tasks}),
            'GCN': (GCN, {'in_dim': 36, 'hidden': args.hidden_dim, 'out_dim': n_tasks}),
            'GAT': (GAT, {'in_dim': 36, 'hidden': args.hidden_dim, 'out_dim': n_tasks}),
            'GIN': (GIN, {'in_dim': 36, 'hidden': args.hidden_dim, 'out_dim': n_tasks}),
            'GraphSAGE': (GraphSAGE, {'in_dim': 36, 'hidden': args.hidden_dim, 'out_dim': n_tasks}),
        }
        
        for name, (cls, kwargs) in baselines.items():
            t0 = time.time()
            metric = train_baseline(
                cls, data_list, task_type, n_tasks,
                seed=seed, epochs=args.finetune_epochs, batch_size=args.batch_size,
                lr=args.lr, device=device, model_kwargs=kwargs,
            )
            baseline_results[name].append(metric)
            elapsed = time.time() - t0
            print(f"  {name:12s}: test={metric:.4f} ({elapsed:.0f}s)")
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {args.dataset}")
    print(f"{'='*60}")
    
    metric_name = 'AUROC' if task_type == 'classification' else 'MAE'
    
    if not args.no_pretrain:
        arr = np.array(dvcl_results)
        print(f"  DVCL (pretrained):     {arr.mean():.4f} ± {arr.std():.4f}")
    
    arr = np.array(dvcl_no_pretrain_results)
    print(f"  DVCL (no pretrain):    {arr.mean():.4f} ± {arr.std():.4f}")
    
    for name in baselines:
        arr = np.array(baseline_results[name])
        print(f"  {name:20s}  {arr.mean():.4f} ± {arr.std():.4f}")
    
    # Save results
    results = {
        'dataset': args.dataset,
        'task_type': task_type,
        'metric': metric_name,
        'seeds': args.seeds,
        'dvcl_pretrained': dvcl_results if not args.no_pretrain else None,
        'dvcl_no_pretrain': dvcl_no_pretrain_results,
        'baselines': dict(baseline_results),
    }
    
    out_path = f'/mnt/e/Project/CompBioChem/tmp/dvcl_{args.dataset.lower()}.json'
    import json
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
