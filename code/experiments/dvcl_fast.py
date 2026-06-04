#!/usr/bin/env python3
"""
DVCL fast benchmark: DVCL (pretrained) vs DVCL (no pretrain) vs DMPNN.
Only runs the key comparison — other baselines from bench_fast results.
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, mean_absolute_error

from code.models.dvcl import DVCL, dual_augment, dvcl_pretrain_loss, morgan_counts
from code.models.magnet_v2 import enrich_molnet_dataset, DMPNN
from torch_geometric.nn import global_mean_pool


# ============================================================
# Data helpers
# ============================================================
def load_dataset(name, root='/tmp/molnet'):
    from torch_geometric.datasets import MoleculeNet
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    ds = MoleculeNet(root=root, name=name)
    data_list = enrich_molnet_dataset(ds)
    return data_list

def extract_fingerprints(data_list, radius=2, n_bits=2048):
    fps = []
    for d in data_list:
        smi = d.smiles if isinstance(d.smiles, str) else d.smiles[0]
        fps.append(morgan_counts(smi, radius, n_bits))
    return np.array(fps, dtype=np.float32)

def make_batch(data_list, indices, device):
    subset = [data_list[i] for i in indices]
    return Batch.from_data_list(subset).to(device)


# ============================================================
# Pre-training
# ============================================================
def pretrain_dvcl(model, data_list, fps_tensor, epochs=30, batch_size=128,
                  lr=1e-3, temperature=0.1, device='cuda'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = len(data_list)
    indices = np.arange(n)
    
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0; n_batches = 0
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2: continue
            subset = [data_list[i] for i in idx]
            v1_list, v2_list = [], []
            for d in subset:
                a1, a2 = dual_augment(d)
                v1_list.append(a1); v2_list.append(a2)
            batch_v1 = Batch.from_data_list(v1_list).to(device)
            batch_v2 = Batch.from_data_list(v2_list).to(device)
            fp_batch = fps_tensor[idx].to(device)
            z_g1, z_g2, z_fp = model.forward_pretrain(batch_v1, batch_v2, fp_batch)
            loss = dvcl_pretrain_loss(z_g1, z_g2, z_fp, temperature)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"    PT {epoch:3d}/{epochs}: loss={total_loss/max(n_batches,1):.4f}")
    return model


# ============================================================
# Fine-tuning + eval (shared for DVCL and DMPNN)
# ============================================================
def get_labels(data_list, indices, n_tasks=1):
    labels = []
    for i in indices:
        y = data_list[i].y
        if y.dim() == 0:
            labels.append(y.item())
        else:
            row = [y[t].item() if t < y.numel() else float('nan') for t in range(n_tasks)]
            labels.append(row if n_tasks > 1 else row[0])
    return np.array(labels, dtype=np.float32)


def finetune_dvcl(model, data_list, fps_np, task_type, n_tasks,
                  seed=42, epochs=80, batch_size=64, lr=1e-3, device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(data_list); idx = np.arange(n); np.random.seed(seed); np.random.shuffle(idx)
    n_train = int(0.8*n); n_val = int(0.1*n)
    tr_idx, va_idx, te_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    fps_tensor = torch.tensor(fps_np, dtype=torch.float32)
    is_binary = (task_type == 'classification')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if is_binary else 'min', patience=8, factor=0.5)
    best_val = -1e9 if is_binary else 1e9; best_test = None; pat = 0
    
    for epoch in range(1, epochs+1):
        model.train(); np.random.shuffle(tr_idx)
        for start in range(0, len(tr_idx), batch_size):
            idx_b = tr_idx[start:start+batch_size]
            if len(idx_b)==0: continue
            batch = make_batch(data_list, idx_b, device)
            fp_b = fps_tensor[idx_b].to(device)
            out = model(batch, fp_b)
            y = get_labels(data_list, idx_b, n_tasks)
            if n_tasks > 1:
                # Multi-task: average loss over valid tasks
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                valid = ~torch.isnan(y_t)
                if valid.sum()==0: continue
                loss = F.binary_cross_entropy_with_logits(out[valid], y_t[valid])
            else:
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                valid = ~torch.isnan(y_t)
                if valid.sum()==0: continue
                if is_binary:
                    pw = (1-y_t[valid]).sum()/(y_t[valid].sum()+1e-6)
                    loss = F.binary_cross_entropy_with_logits(out[valid], y_t[valid], pos_weight=pw.clamp(0.1,10))
                else:
                    loss = F.mse_loss(out[valid], y_t[valid])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_m = eval_dvcl(model, data_list, fps_tensor, va_idx, is_binary, n_tasks, device)
        test_m = eval_dvcl(model, data_list, fps_tensor, te_idx, is_binary, n_tasks, device)
        scheduler.step(val_m)
        improved = (val_m > best_val + 1e-4) if is_binary else (val_m < best_val - 1e-4)
        if improved: best_val = val_m; best_test = test_m; pat = 0
        else: pat += 1
        if pat >= 15: break
    return best_test


def eval_dvcl(model, data_list, fps_tensor, indices, is_binary, n_tasks, device, batch_size=256):
    model.eval(); all_p = []; all_y = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx_b = indices[start:start+batch_size]
            if len(idx_b)==0: continue
            batch = make_batch(data_list, idx_b, device)
            fp_b = fps_tensor[idx_b].to(device)
            out = model(batch, fp_b)
            all_p.extend(out.cpu().numpy().tolist())
            y = get_labels(data_list, idx_b, n_tasks)
            if n_tasks > 1:
                all_y.extend(y.tolist())
            else:
                all_y.extend(y.tolist() if hasattr(y, 'tolist') else [y])
    preds = np.array(all_p); labels = np.array(all_y)
    valid = ~np.isnan(labels)
    if valid.sum()==0: return 0.5 if is_binary else 1e9
    if is_binary:
        try: return roc_auc_score(labels[valid], 1/(1+np.exp(-preds[valid])))
        except: return 0.5
    else:
        return mean_absolute_error(labels[valid], preds[valid])


# ============================================================
# DMPNN baseline
# ============================================================
def train_dmpnn(data_list, task_type, n_tasks, seed=42, epochs=80, batch_size=64, lr=1e-3, device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(data_list); idx = np.arange(n); np.random.seed(seed); np.random.shuffle(idx)
    n_train = int(0.8*n); n_val = int(0.1*n)
    tr_idx, va_idx, te_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    is_binary = (task_type == 'classification')
    model = DMPNN(node_dim=36, edge_dim=12, hidden_dim=128, out_dim=n_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if is_binary else 'min', patience=8, factor=0.5)
    best_val = -1e9 if is_binary else 1e9; best_test = None; pat = 0
    
    for epoch in range(1, epochs+1):
        model.train(); np.random.shuffle(tr_idx)
        for start in range(0, len(tr_idx), batch_size):
            idx_b = tr_idx[start:start+batch_size]
            if len(idx_b)==0: continue
            batch = make_batch(data_list, idx_b, device)
            out = model(batch)
            y = get_labels(data_list, idx_b, n_tasks)
            if n_tasks > 1:
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                valid = ~torch.isnan(y_t)
                if valid.sum()==0: continue
                loss = F.binary_cross_entropy_with_logits(out[valid], y_t[valid])
            else:
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                valid = ~torch.isnan(y_t)
                if valid.sum()==0: continue
                if is_binary:
                    pw = (1-y_t[valid]).sum()/(y_t[valid].sum()+1e-6)
                    loss = F.binary_cross_entropy_with_logits(out[valid], y_t[valid], pos_weight=pw.clamp(0.1,10))
                else:
                    loss = F.mse_loss(out[valid], y_t[valid])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_m = eval_dmpnn(model, data_list, va_idx, is_binary, n_tasks, device)
        test_m = eval_dmpnn(model, data_list, te_idx, is_binary, n_tasks, device)
        scheduler.step(val_m)
        improved = (val_m > best_val + 1e-4) if is_binary else (val_m < best_val - 1e-4)
        if improved: best_val = val_m; best_test = test_m; pat = 0
        else: pat += 1
        if pat >= 15: break
    return best_test


def eval_dmpnn(model, data_list, indices, is_binary, n_tasks, device, batch_size=256):
    model.eval(); all_p = []; all_y = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx_b = indices[start:start+batch_size]
            if len(idx_b)==0: continue
            batch = make_batch(data_list, idx_b, device)
            out = model(batch)
            all_p.extend(out.cpu().numpy().tolist())
            y = get_labels(data_list, idx_b, n_tasks)
            if n_tasks > 1:
                all_y.extend(y.tolist())
            else:
                all_y.extend(y.tolist() if hasattr(y, 'tolist') else [y])
    preds = np.array(all_p); labels = np.array(all_y)
    valid = ~np.isnan(labels)
    if valid.sum()==0: return 0.5 if is_binary else 1e9
    if is_binary:
        try: return roc_auc_score(labels[valid], 1/(1+np.exp(-preds[valid])))
        except: return 0.5
    else:
        return mean_absolute_error(labels[valid], preds[valid])


# ============================================================
# Main
# ============================================================
DATASET_CONFIG = {
    'BBBP':   {'task_type': 'classification', 'n_tasks': 1},
    'BACE':   {'task_type': 'classification', 'n_tasks': 1},
    'HIV':    {'task_type': 'classification', 'n_tasks': 1},
    'Tox21':  {'task_type': 'classification', 'n_tasks': 12},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['BBBP', 'BACE', 'HIV', 'Tox21'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--pretrain_epochs', type=int, default=30)
    parser.add_argument('--finetune_epochs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    all_results = {}
    
    for dataset in args.datasets:
        config = DATASET_CONFIG[dataset]
        task_type = config['task_type']
        n_tasks = config['n_tasks']
        
        print(f"\n{'#'*60}")
        print(f"  {dataset} (task={task_type}, n_tasks={n_tasks})")
        print(f"{'#'*60}")
        
        data_list = load_dataset(dataset)
        fps_np = extract_fingerprints(data_list)
        
        dvcl_pt_results = []
        dvcl_nopt_results = []
        dmpnn_results = []
        
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed+1}/{args.seeds} ---")
            
            # 1. DVCL with pre-training
            torch.manual_seed(seed); np.random.seed(seed)
            model_pt = DVCL(node_dim=36, edge_dim=12, fp_dim=2048, hidden_dim=128, out_dim=n_tasks).to(device)
            t0 = time.time()
            model_pt = pretrain_dvcl(model_pt, data_list,
                                     torch.tensor(fps_np, dtype=torch.float32),
                                     epochs=args.pretrain_epochs, device=device)
            pt_time = time.time() - t0
            r = finetune_dvcl(model_pt, data_list, fps_np, task_type, n_tasks,
                              seed=seed, epochs=args.finetune_epochs, device=device)
            dvcl_pt_results.append(r)
            print(f"  DVCL+PT: {r:.4f} (pretrain {pt_time:.0f}s)")
            
            # 2. DVCL without pre-training
            torch.manual_seed(seed); np.random.seed(seed)
            model_nopt = DVCL(node_dim=36, edge_dim=12, fp_dim=2048, hidden_dim=128, out_dim=n_tasks).to(device)
            r2 = finetune_dvcl(model_nopt, data_list, fps_np, task_type, n_tasks,
                               seed=seed, epochs=args.finetune_epochs, device=device)
            dvcl_nopt_results.append(r2)
            print(f"  DVCL:    {r2:.4f}")
            
            # 3. DMPNN baseline
            t0 = time.time()
            r3 = train_dmpnn(data_list, task_type, n_tasks,
                             seed=seed, epochs=args.finetune_epochs, device=device)
            dmpnn_time = time.time() - t0
            dmpnn_results.append(r3)
            print(f"  DMPNN:   {r3:.4f} ({dmpnn_time:.0f}s)")
        
        # Summary
        metric = 'AUROC' if task_type == 'classification' else 'MAE'
        pt_arr = np.array(dvcl_pt_results)
        nopt_arr = np.array(dvcl_nopt_results)
        dm_arr = np.array(dmpnn_results)
        
        print(f"\n  === {dataset} Summary ({metric}) ===")
        print(f"  DVCL+PT:  {pt_arr.mean():.4f} ± {pt_arr.std():.4f}")
        print(f"  DVCL:     {nopt_arr.mean():.4f} ± {nopt_arr.std():.4f}")
        print(f"  DMPNN:    {dm_arr.mean():.4f} ± {dm_arr.std():.4f}")
        delta = pt_arr.mean() - dm_arr.mean()
        print(f"  Δ(DVCL+PT vs DMPNN): {delta:+.4f}")
        
        all_results[dataset] = {
            'dvcl_pretrained': dvcl_pt_results,
            'dvcl_no_pretrain': dvcl_nopt_results,
            'dmpnn': dmpnn_results,
        }
    
    # Final summary
    print(f"\n{'#'*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*60}")
    for ds, r in all_results.items():
        metric = 'AUROC' if DATASET_CONFIG[ds]['task_type'] == 'classification' else 'MALE'
        pt = np.array(r['dvcl_pretrained'])
        nopt = np.array(r['dvcl_no_pretrain'])
        dm = np.array(r['dmpnn'])
        print(f"  {ds:8s}  DVCL+PT={pt.mean():.4f}±{pt.std():.4f}  DVCL={nopt.mean():.4f}±{nopt.std():.4f}  DMPNN={dm.mean():.4f}±{dm.std():.4f}  Δ={pt.mean()-dm.mean():+.4f}")
    
    out_path = '/mnt/e/Project/CompBioChem/tmp/dvcl_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
