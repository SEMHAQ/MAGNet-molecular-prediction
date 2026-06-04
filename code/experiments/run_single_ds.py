#!/usr/bin/env python3
"""HIV benchmark only, 1 seed, to test if it completes."""
import os, sys, time, random, json, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset as TDataset
from collections import defaultdict
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

DATA_DIR = os.path.join(ROOT, "data", "molnet")
RESULTS_DIR = os.path.join(ROOT, "results")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64; EPOCHS = 20; PAT = 5; HIDDEN = 64

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def scaffold_split(dataset, seed=42):
    from rdkit import Chem as _C
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        d = dataset[i]
        smi = d.smiles if isinstance(d.smiles, str) else d.smiles[0]
        mol = _C.MolFromSmiles(smi)
        try: sc = MurckoScaffoldSmiles(mol=mol) if mol else None
        except: sc = None
        scaffolds[sc if sc else f"_unk_{i}"].append(i)
    rng = np.random.RandomState(seed)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng.shuffle(groups)
    n = len(dataset); n_tr = int(n*0.8); n_va = int(n*0.1)
    tr,va,te = [],[],[]
    for g in groups:
        if len(tr)+len(g)<=n_tr: tr+=g
        elif len(va)+len(g)<=n_va: va+=g
        else: te+=g
    seen=set(tr)|set(va)|set(te)
    for i in range(n):
        if i not in seen: te.append(i)
    return tr,va,te

def _af(atom):
    from rdkit import Chem
    nums=[6,7,8,16,9,17,35,53,15,0]
    at=[0]*10; at[nums.index(atom.GetAtomicNum()) if atom.GetAtomicNum() in nums else 9]=1
    deg=[0]*6; deg[min(atom.GetDegree(),5)]=1
    ch=[0]*5; ch[min(max(atom.GetFormalCharge()+2,0),4)]=1
    hts=[Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,
         Chem.rdchem.HybridizationType.SP3,Chem.rdchem.HybridizationType.SP3D,
         Chem.rdchem.HybridizationType.SP3D2]
    hy=[0]*5; h=atom.GetHybridization()
    if h in hts: hy[hts.index(h)]=1
    ar=[1 if atom.GetIsAromatic() else 0]
    nh=[0]*5; nh[min(atom.GetTotalNumHs(),4)]=1
    ir=[1 if atom.IsInRing() else 0]
    chi=[0]*3
    try:
        c=atom.GetChiralTag()
        if c==Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: chi[0]=1
        elif c==Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: chi[1]=1
        else: chi[2]=1
    except: chi[2]=1
    return at+deg+ch+hy+ar+nh+ir+chi

def _bf(bond):
    from rdkit import Chem
    bt=[0]*4; t=bond.GetBondType()
    if t==Chem.rdchem.BondType.SINGLE: bt[0]=1
    elif t==Chem.rdchem.BondType.DOUBLE: bt[1]=1
    elif t==Chem.rdchem.BondType.TRIPLE: bt[2]=1
    elif t==Chem.rdchem.BondType.AROMATIC: bt[3]=1
    st=[0]*6; st[3]=1
    cj=[1 if bond.GetIsConjugated() else 0]
    ir=[1 if bond.IsInRing() else 0]
    return bt+st+cj+ir

def enrich(dataset):
    from rdkit import Chem
    out=[]
    for i in range(len(dataset)):
        d=dataset[i]; smi=d.smiles if isinstance(d.smiles,str) else d.smiles[0]
        mol=Chem.MolFromSmiles(smi)
        if mol is None: continue
        d.x=torch.tensor([_af(a) for a in mol.GetAtoms()],dtype=torch.float)
        ei,ea=[],[]
        for bond in mol.GetBonds():
            a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            bf=_bf(bond); ei+=[[a,b],[b,a]]; ea+=[bf,bf]
        if ei:
            d.edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous()
            d.edge_attr=torch.tensor(ea,dtype=torch.float)
        else:
            d.edge_index=torch.zeros(2,0,dtype=torch.long)
            d.edge_attr=torch.zeros(0,12,dtype=torch.float)
        d.num_nodes=d.x.size(0); out.append(d)
    return out

class ListDS(TDataset):
    def __init__(self,L): self.L=L
    def __len__(self): return len(self.L)
    def __getitem__(self,i):
        if isinstance(i,(list,np.ndarray)): return [self.L[j] for j in i]
        return self.L[i]

class PyGCN(nn.Module):
    def __init__(s,i,h=64,o=1,nl=3,d=0.1):
        super().__init__(); s.c=nn.ModuleList([GCNConv(i,h)]+[GCNConv(h,h) for _ in range(nl-1)]); s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        for c in s.c: x=F.relu(c(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

class PyGAT(nn.Module):
    def __init__(s,i,h=64,o=1,nl=3,hd=4,d=0.1):
        super().__init__(); s.c=nn.ModuleList([GATConv(i,h//hd,heads=hd,dropout=d)]+[GATConv(h,h//hd,heads=hd,dropout=d) for _ in range(nl-1)]); s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        for c in s.c: x=F.relu(c(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

class PyGIN(nn.Module):
    def __init__(s,i,h=64,o=1,nl=3,d=0.1):
        super().__init__(); s.c=nn.ModuleList()
        for j in range(nl):
            m=nn.Sequential(nn.Linear(i if j==0 else h,h),nn.ReLU(),nn.Linear(h,h))
            s.c.append(GINConv(m,train_eps=True))
        s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        for c in s.c: x=F.relu(c(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

class PySAGE(nn.Module):
    def __init__(s,i,h=64,o=1,nl=3,d=0.1):
        super().__init__(); s.c=nn.ModuleList([SAGEConv(i,h)]+[SAGEConv(h,h) for _ in range(nl-1)]); s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        for c in s.c: x=F.relu(c(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

class MPNNLayer(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.m=nn.Linear(i,o,bias=False); s.u=nn.GRUCell(o,i)
    def forward(s,x,ei):
        src,dst=ei[0],ei[1]
        msgs=s.m(x[src])
        agg=torch.zeros(x.size(0),msgs.size(1),device=x.device,dtype=msgs.dtype)
        agg.scatter_add_(0,dst.unsqueeze(1).expand_as(msgs),msgs)
        return s.u(agg,x)

class PyMPNN(nn.Module):
    def __init__(s,i,h=64,o=1,nl=3,d=0.1):
        super().__init__(); s.ip=nn.Linear(i,h); s.l=nn.ModuleList([MPNNLayer(h,h) for _ in range(nl)]); s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        x=F.relu(s.ip(x))
        for l in s.l: x=F.relu(l(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

class DMPNNL(nn.Module):
    def __init__(s,nd,ed,hd):
        super().__init__()
        s.msg=nn.Sequential(nn.Linear(nd+ed,hd),nn.ReLU(),nn.Linear(hd,hd))
        s.upd=nn.GRUCell(hd,nd)
    def forward(s,x,ei,ea):
        src,dst=ei[0],ei[1]
        msgs=s.msg(torch.cat([x[src],ea],-1))
        agg=torch.zeros(x.size(0),msgs.size(1),device=x.device,dtype=msgs.dtype)
        agg.scatter_add_(0,dst.unsqueeze(1).expand_as(msgs),msgs)
        return s.upd(agg,x)

class DMPNN(nn.Module):
    def __init__(s,nd=36,ed=12,hd=64,o=1,nl=3,d=0.1):
        super().__init__(); s.np=nn.Linear(nd,hd); s.ep=nn.Linear(ed,hd)
        s.l=nn.ModuleList([DMPNNL(hd,hd,hd) for _ in range(nl)]); s.fc=nn.Linear(hd,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        ea=data.edge_attr.float() if hasattr(data,'edge_attr') and data.edge_attr is not None else torch.zeros(ei.size(1),12,device=x.device)
        x=F.relu(s.np(x)); e=s.ep(ea)
        for l in s.l: x=l(x,ei,e); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

def train_eval(model,dataset,tr,va,te,device,seed,is_mt=False):
    set_seed(seed); model=model.to(device)
    tl=DataLoader(dataset[tr],BATCH,shuffle=True)
    vl=DataLoader(dataset[va],BATCH)
    tstl=DataLoader(dataset[te],BATCH)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    best_val,best_st,noimp=-1,None,0
    t0=time.time()
    for ep in range(EPOCHS):
        model.train()
        for batch in tl:
            batch=batch.to(device); opt.zero_grad(True)
            logits=model(batch)
            if is_mt:
                y=batch.y.float(); mask=~torch.isnan(y); y=torch.nan_to_num(y,0)
                # logits shape: (B, n_tasks) -> need to match
                if logits.dim()==1: logits=logits.unsqueeze(1).expand_as(y)
                loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*mask).sum()/mask.clamp(1e-8).sum()
            else:
                y=batch.y.float().view(-1); mask=~torch.isnan(y); y=torch.nan_to_num(y,0)
                loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*mask).sum()/mask.clamp(1e-8).sum()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step()
        model.eval(); preds,labs=[],[]
        with torch.no_grad():
            for batch in vl:
                batch=batch.to(device)
                logits=model(batch)
                if is_mt:
                    y=batch.y.float()
                    if logits.dim()==1: logits=logits.unsqueeze(1).expand_as(y)
                    preds.append(torch.sigmoid(logits).cpu().numpy())
                    labs.append(y.cpu().numpy())
                else:
                    preds.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
                    labs.append(batch.y.float().cpu().numpy().reshape(-1))
        p,l=np.concatenate(preds),np.concatenate(labs)
        if is_mt:
            aucs=[]
            for t in range(l.shape[1]):
                m=~np.isnan(l[:,t])
                if m.sum()>1 and len(np.unique(l[m,t]))>1:
                    try: aucs.append(roc_auc_score(l[m,t],p[m,t]))
                    except: pass
            va_=np.nanmean(aucs) if aucs else 0
        else:
            m=~np.isnan(l)
            va_=roc_auc_score(l[m],p[m]) if m.sum()>1 and len(np.unique(l[m]))>1 else 0
        if va_>best_val: best_val=va_; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; noimp=0
        else: noimp+=1
        if noimp>=PAT: break
    elapsed=time.time()-t0
    model.load_state_dict(best_st); model.eval()
    preds,labs=[],[]
    with torch.no_grad():
        for batch in tstl:
            batch=batch.to(device)
            logits=model(batch)
            if is_mt:
                y=batch.y.float()
                if logits.dim()==1: logits=logits.unsqueeze(1).expand_as(y)
                preds.append(torch.sigmoid(logits).cpu().numpy())
                labs.append(y.cpu().numpy())
            else:
                preds.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
                labs.append(batch.y.float().cpu().numpy().reshape(-1))
    p,l=np.concatenate(preds),np.concatenate(labs)
    if is_mt:
        aucs=[]
        for t in range(l.shape[1]):
            m=~np.isnan(l[:,t])
            if m.sum()>1 and len(np.unique(l[m,t]))>1:
                try:
                    aucs.append(roc_auc_score(l[m,t],p[m,t]))
                except:
                    pass
        auc=np.nanmean(aucs) if aucs else np.nan
    else:
        m=~np.isnan(l)
        auc=roc_auc_score(l[m],p[m]) if m.sum()>1 and len(np.unique(l[m]))>1 else np.nan
    params=sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    return [auc],params,elapsed

def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "HIV"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print(f"Device: {DEV}, Dataset: {ds_name}, Seeds: {n_seeds}")
    raw=MoleculeNet(root=DATA_DIR,name=ds_name)
    in_dim=raw[0].x.shape[-1]
    # Detect multi-task
    y0=raw[0].y
    n_tasks=y0.shape[-1] if y0.dim()>0 else 1
    is_mt=n_tasks>1
    out_dim=n_tasks if is_mt else 1
    print(f"Molecules: {len(raw)}, Features: {in_dim}, Tasks: {n_tasks}, Multi-task: {is_mt}")
    
    print("Enriching features...")
    enriched=ListDS(enrich(raw))
    nd=enriched[0].x.shape[1]; ed=enriched[0].edge_attr.shape[1]
    gc.collect()
    print(f"Enriched: {nd} atom, {ed} edge")
    
    v1_models={
        "GCN": lambda: PyGCN(in_dim,HIDDEN,out_dim),
        "GAT": lambda: PyGAT(in_dim,HIDDEN,out_dim),
        "GIN": lambda: PyGIN(in_dim,HIDDEN,out_dim),
        "GraphSAGE": lambda: PySAGE(in_dim,HIDDEN,out_dim),
        "MPNN": lambda: PyMPNN(in_dim,HIDDEN,out_dim),
    }
    
    ds_results={}
    for name,mk in v1_models.items():
        print(f">> {name}",end="",flush=True)
        aucs=[]
        for seed in range(n_seeds):
            tr,va,te=scaffold_split(raw,seed=seed)
            auc,params,elapsed=train_eval(mk(),raw,tr,va,te,DEV,seed,is_mt)
            aucs.append(np.nanmean(auc)); print(f".({elapsed:.0f}s)",end="",flush=True)
            gc.collect()
        ds_results[name]={"mean":float(np.nanmean(aucs)),"std":float(np.nanstd(aucs)),
                          "params":params,"aucs":[float(a) for a in aucs]}
        print(f"  AUC={np.nanmean(aucs):.4f}+/-{np.nanstd(aucs):.4f}")
    
    # DMPNN
    print(f">> DMPNN",end="",flush=True)
    aucs=[]
    for seed in range(n_seeds):
        tr,va,te=scaffold_split(raw,seed=seed)
        auc,params,elapsed=train_eval(DMPNN(nd,ed,HIDDEN,out_dim),enriched,tr,va,te,DEV,seed,is_mt)
        aucs.append(np.nanmean(auc)); print(f".({elapsed:.0f}s)",end="",flush=True)
        gc.collect()
    ds_results["DMPNN"]={"mean":float(np.nanmean(aucs)),"std":float(np.nanstd(aucs)),
                         "params":params,"aucs":[float(a) for a in aucs]}
    print(f"  AUC={np.nanmean(aucs):.4f}+/-{np.nanstd(aucs):.4f}")
    
    # RF-Morgan
    from rdkit import Chem as _C; from rdkit.Chem import AllChem as _AC
    def morgan(smis,rad=2,nb=2048):
        fps=[]
        for s in smis:
            mol=_C.MolFromSmiles(s)
            if mol is None: fps.append(np.zeros(nb,dtype=np.uint8)); continue
            fps.append(np.array(_AC.GetMorganFingerprintAsBitVect(mol,rad,nBits=nb),dtype=np.uint8))
        return np.stack(fps)
    
    print(f">> RF-Morgan",end="",flush=True)
    rf_aucs=[]
    smis=[(raw[i].smiles if isinstance(raw[i].smiles,str) else raw[i].smiles[0]) for i in range(len(raw))]
    fps_all=morgan(smis)
    for seed in range(n_seeds):
        tr,va,te=scaffold_split(raw,seed=seed)
        set_seed(seed)
        X_tr,X_te=fps_all[tr],fps_all[te]
        if is_mt:
            # Multi-task: train separate RF per task
            task_aucs=[]
            for t in range(n_tasks):
                tr_y=np.array([raw[i].y[t].item() if raw[i].y.dim()>0 else raw[i].y.item() for i in tr])
                te_y=np.array([raw[i].y[t].item() if raw[i].y.dim()>0 else raw[i].y.item() for i in te])
                m=~np.isnan(tr_y)
                if m.sum()<10 or len(np.unique(tr_y[m]))<2: continue
                clf=RandomForestClassifier(n_estimators=500,random_state=seed,n_jobs=-1)
                clf.fit(X_tr[m],tr_y[m])
                m2=~np.isnan(te_y)
                if m2.sum()<2 or len(np.unique(te_y[m2]))<2: continue
                probs=clf.predict_proba(X_te[m2])[:,1]
                try: task_aucs.append(roc_auc_score(te_y[m2],probs))
                except: pass
            rf_aucs.append(np.nanmean(task_aucs) if task_aucs else np.nan)
        else:
            tr_y=np.array([raw[i].y.item() if raw[i].y.dim()==0 else raw[i].y[0].item() for i in tr])
            te_y=np.array([raw[i].y.item() if raw[i].y.dim()==0 else raw[i].y[0].item() for i in te])
            m=~np.isnan(tr_y); clf=RandomForestClassifier(n_estimators=500,random_state=seed,n_jobs=-1)
            clf.fit(X_tr[m],tr_y[m])
            m2=~np.isnan(te_y)
            try: rf_aucs.append(roc_auc_score(te_y[m2],clf.predict_proba(X_te[m2])[:,1]))
            except: rf_aucs.append(np.nan)
        print(".",end="",flush=True)
    ds_results["RF-Morgan"]={"mean":float(np.nanmean(rf_aucs)),"std":float(np.nanstd(rf_aucs)),
                             "params":0,"aucs":[float(a) for a in rf_aucs]}
    print(f"  AUC={np.nanmean(rf_aucs):.4f}+/-{np.nanstd(rf_aucs):.4f}")
    
    print(f"\n{'Model':15s}  {'AUC':>14s}")
    print(f"{'-'*15}  {'-'*14}")
    for n,r in sorted(ds_results.items(),key=lambda x:-x[1]["mean"]):
        print(f"{n:15s}  {r['mean']:.4f}+/-{r['std']:.4f}")
    
    # Save
    os.makedirs(RESULTS_DIR,exist_ok=True)
    out_path=os.path.join(RESULTS_DIR,f"{ds_name.lower()}_results.json")
    with open(out_path,"w") as f:
        json.dump(ds_results,f,indent=2)
    print(f"\nSaved to {out_path}")

if __name__=="__main__":
    main()
