#!/usr/bin/env python3
"""
Full MoleculeNet benchmark: MAGNet-v2 + DMPNN + all baselines.
4 datasets × 8 models × 10 seeds.
"""
import os, sys, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GCNConv, GATConv, GINConv, SAGEConv,
                                 global_mean_pool)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
from torch.utils.data import Dataset as TDataset
from collections import defaultdict
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

DATA_DIR = os.path.join(ROOT, "data", "molnet")
RESULTS_DIR = os.path.join(ROOT, "results")
DATASETS = ["BBBP", "BACE", "HIV", "Tox21"]
NUM_SEEDS = 5
BATCH = 64; MAX_EPOCHS = 30; PATIENCE = 7; HIDDEN = 128; LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Utils
# ============================================================
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# Scaffold split
# ============================================================
def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, seed=42):
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
    n = len(dataset); n_tr = int(n*frac_train); n_va = int(n*frac_val)
    tr,va,te = [],[],[]
    for g in groups:
        if len(tr)+len(g)<=n_tr: tr+=g
        elif len(va)+len(g)<=n_va: va+=g
        else: te+=g
    seen=set(tr)|set(va)|set(te)
    for i in range(n):
        if i not in seen: te.append(i)
    return tr,va,te

# ============================================================
# RDKit feature enrichment
# ============================================================
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

# ============================================================
# Standard GNN baselines (v1 features, 9-dim)
# ============================================================
class PyGCN(nn.Module):
    def __init__(s,i,h=128,o=1,nl=3,d=0.1):
        super().__init__(); s.c=nn.ModuleList([GCNConv(i,h)]+[GCNConv(h,h) for _ in range(nl-1)]); s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        for c in s.c: x=F.relu(c(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

class PyGAT(nn.Module):
    def __init__(s,i,h=128,o=1,nl=3,hd=4,d=0.1):
        super().__init__(); s.c=nn.ModuleList([GATConv(i,h//hd,heads=hd,dropout=d)]+[GATConv(h,h//hd,heads=hd,dropout=d) for _ in range(nl-1)]); s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        for c in s.c: x=F.relu(c(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

class PyGIN(nn.Module):
    def __init__(s,i,h=128,o=1,nl=3,d=0.1):
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
    def __init__(s,i,h=128,o=1,nl=3,d=0.1):
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
    def __init__(s,i,h=128,o=1,nl=3,d=0.1):
        super().__init__(); s.ip=nn.Linear(i,h); s.l=nn.ModuleList([MPNNLayer(h,h) for _ in range(nl)]); s.fc=nn.Linear(h,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        x=F.relu(s.ip(x))
        for l in s.l: x=F.relu(l(x,ei)); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

# ============================================================
# MAGNet v1 (multi-scale GCN)
# ============================================================
class MAGNetV1(nn.Module):
    def __init__(s,i,h=128,o=1,ns=3,nh=4,nq=4,d=0.1):
        super().__init__(); s.ns=ns; s.nq=nq
        s.ip=nn.Linear(i,h)
        s.sc=nn.ModuleList([GCNConv(h,h) for _ in range(ns)])
        s.sn=nn.ModuleList([nn.LayerNorm(h) for _ in range(ns)])
        s.gl=nn.Linear(h*ns,ns)
        s.ca=nn.MultiheadAttention(h,nh,dropout=d,batch_first=True)
        s.cn=nn.LayerNorm(h)
        s.q=nn.Parameter(torch.randn(nq,h)*0.02)
        s.pa=nn.MultiheadAttention(h,nh,dropout=d,batch_first=True)
        s.pp=nn.Linear(h*nq,h); s.pn=nn.LayerNorm(h)
        s.clf=nn.Sequential(nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(d),nn.Linear(h//2,o))
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        h=F.relu(s.ip(x)); sf=[]; cur=h
        for c,n in zip(s.sc,s.sn): cur=F.relu(n(c(cur,ei)))+cur; sf.append(cur)
        st=torch.stack(sf,1); ct=st.reshape(st.size(0),-1)
        g=torch.softmax(s.gl(ct),-1); ga=(st*g.unsqueeze(-1)).sum(1)
        q=ga.unsqueeze(1); ao,_=s.ca(q,st,st); fu=s.cn(ao.squeeze(1)+ga)
        bs=int(b.max().item())+1; gl=[]
        for i in range(bs):
            m=b==i; nf=fu[m].unsqueeze(0); qe=s.q.unsqueeze(0)
            po,_=s.pa(qe,nf,nf); gl.append(po.reshape(1,-1))
        gr=torch.cat(gl,0); gr=F.relu(s.pp(gr)); gr=s.pn(gr)
        return s.clf(gr).squeeze(-1)

# ============================================================
# MAGNet v2 (edge-aware multi-scale)
# ============================================================
class EdgeMPNN(nn.Module):
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

class MAGNetV2(nn.Module):
    def __init__(s,nd=36,ed=12,hd=128,o=1,ns=3,nh=4,nq=4,d=0.1):
        super().__init__(); s.ns=ns; s.nq=nq
        s.np=nn.Linear(nd,hd); s.ep=nn.Linear(ed,hd)
        s.convs=nn.ModuleList([EdgeMPNN(hd,hd,hd) for _ in range(ns)])
        s.norms=nn.ModuleList([nn.LayerNorm(hd) for _ in range(ns)])
        s.gl=nn.Linear(hd*ns,ns)
        s.ca=nn.MultiheadAttention(hd,nh,dropout=d,batch_first=True)
        s.cn=nn.LayerNorm(hd)
        s.q=nn.Parameter(torch.randn(nq,hd)*0.02)
        s.pa=nn.MultiheadAttention(hd,nh,dropout=d,batch_first=True)
        s.pp=nn.Linear(hd*nq,hd); s.pn=nn.LayerNorm(hd)
        s.clf=nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(d),nn.Linear(hd//2,o))
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        ea=data.edge_attr.float() if hasattr(data,'edge_attr') and data.edge_attr is not None else torch.zeros(ei.size(1),12,device=x.device)
        h=F.relu(s.np(x)); e=s.ep(ea)
        sf=[]; cur=h
        for cv,nm in zip(s.convs,s.norms): cur=cv(cur,ei,e); sf.append(cur)
        st=torch.stack(sf,1); ct=st.reshape(st.size(0),-1)
        g=torch.softmax(s.gl(ct),-1); ga=(st*g.unsqueeze(-1)).sum(1)
        q=ga.unsqueeze(1); ao,_=s.ca(q,st,st); fu=s.cn(ao.squeeze(1)+ga)
        bs=int(b.max().item())+1; gl=[]
        for i in range(bs):
            m=b==i; nf=fu[m].unsqueeze(0); qe=s.q.unsqueeze(0)
            po,_=s.pa(qe,nf,nf); gl.append(po.reshape(1,-1))
        gr=torch.cat(gl,0); gr=F.relu(s.pp(gr)); gr=s.pn(gr)
        return s.clf(gr).squeeze(-1)

# ============================================================
# DMPNN (edge-aware baseline)
# ============================================================
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
    def __init__(s,nd=36,ed=12,hd=128,o=1,nl=3,d=0.1):
        super().__init__(); s.np=nn.Linear(nd,hd); s.ep=nn.Linear(ed,hd)
        s.l=nn.ModuleList([DMPNNL(hd,hd,hd) for _ in range(nl)]); s.fc=nn.Linear(hd,o); s.d=d
    def forward(s,data):
        x,ei,b=data.x.float(),data.edge_index,data.batch
        ea=data.edge_attr.float() if hasattr(data,'edge_attr') and data.edge_attr is not None else torch.zeros(ei.size(1),12,device=x.device)
        x=F.relu(s.np(x)); e=s.ep(ea)
        for l in s.l: x=l(x,ei,e); x=F.dropout(x,s.d,training=s.training)
        return s.fc(global_mean_pool(x,b)).squeeze(-1)

# ============================================================
# RF-Morgan
# ============================================================
def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    from rdkit import Chem as _C; from rdkit.Chem import AllChem as _AC
    fps=[]
    for smi in smiles_list:
        mol=_C.MolFromSmiles(smi)
        if mol is None: fps.append(np.zeros(n_bits,dtype=np.uint8)); continue
        fp=_AC.GetMorganFingerprintAsBitVect(mol,radius,nBits=n_bits)
        fps.append(np.array(fp,dtype=np.uint8))
    return np.stack(fps)

def run_rf(dataset,tr,va,te,seed):
    set_seed(seed)
    smis=[(dataset[i].smiles if isinstance(dataset[i].smiles,str) else dataset[i].smiles[0]) for i in range(len(dataset))]
    ys=np.array([dataset[i].y.item() if dataset[i].y.dim()==0 else dataset[i].y[0].item() for i in range(len(dataset))])
    tr_s=[smis[i] for i in tr]; te_s=[smis[i] for i in te]
    tr_y=ys[tr]; te_y=ys[te]
    m=~np.isnan(tr_y); tr_s=[s for s,m2 in zip(tr_s,m) if m2]; tr_y=tr_y[m]
    m=~np.isnan(te_y); te_s=[s for s,m2 in zip(te_s,m) if m2]; te_y=te_y[m]
    clf=RandomForestClassifier(n_estimators=500,random_state=seed,n_jobs=-1)
    clf.fit(smiles_to_morgan(tr_s),tr_y)
    probs=clf.predict_proba(smiles_to_morgan(te_s))[:,1]
    try: return [roc_auc_score(te_y,probs)]
    except: return [np.nan]

# ============================================================
# Train / eval
# ============================================================
def masked_bce(logits,target,mask):
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    return (loss*mask).sum()/mask.clamp(1e-8).sum()

def train_eval(model,dataset,tr,va,te,device,seed,is_mt=False):
    set_seed(seed); model=model.to(device)
    tl=DataLoader(dataset[tr],BATCH,shuffle=True)
    vl=DataLoader(dataset[va],BATCH)
    tstl=DataLoader(dataset[te],BATCH)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,MAX_EPOCHS)
    best_val,best_st,noimp=-1,None,0
    t0=time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        for batch in tl:
            batch=batch.to(device); opt.zero_grad(True)
            logits=model(batch)
            if is_mt:
                y=batch.y.float().view(batch.num_graphs,-1); mask=~torch.isnan(y); y=torch.nan_to_num(y,0)
                loss=masked_bce(logits.view(batch.num_graphs,-1),y,mask)
            else:
                y=batch.y.float().view(-1); mask=~torch.isnan(y); y=torch.nan_to_num(y,0)
                loss=masked_bce(logits,y,mask)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step()
        model.eval(); preds,labs=[],[]
        with torch.no_grad():
            for batch in vl:
                batch=batch.to(device); logits=model(batch)
                if is_mt:
                    preds.append(torch.sigmoid(logits).cpu().numpy().reshape(batch.num_graphs,-1))
                    labs.append(batch.y.cpu().numpy().reshape(batch.num_graphs,-1))
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
        if noimp>=PATIENCE: break
    elapsed=time.time()-t0
    model.load_state_dict(best_st); model.eval()
    preds,labs=[],[]
    with torch.no_grad():
        for batch in tstl:
            batch=batch.to(device); logits=model(batch)
            if is_mt:
                preds.append(torch.sigmoid(logits).cpu().numpy().reshape(batch.num_graphs,-1))
                labs.append(batch.y.cpu().numpy().reshape(batch.num_graphs,-1))
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
        test_auc=np.nanmean(aucs) if aucs else np.nan
    else:
        m=~np.isnan(l)
        test_auc=roc_auc_score(l[m],p[m]) if m.sum()>1 and len(np.unique(l[m]))>1 else np.nan
    params=sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    return [test_auc],params,elapsed

# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",default=DATASETS)
    parser.add_argument("--seeds",type=int,default=NUM_SEEDS)
    parser.add_argument("--quick",action="store_true")
    args=parser.parse_args()
    
    if args.quick:
        args.datasets=["BBBP"]; args.seeds=3
    
    print(f"Device: {DEVICE}")
    print(f"Datasets: {args.datasets}, Seeds: {args.seeds}")
    
    # Models that use original 9-dim features
    v1_models={
        "GCN": lambda i: PyGCN(i,HIDDEN,1),
        "GAT": lambda i: PyGAT(i,HIDDEN,1),
        "GIN": lambda i: PyGIN(i,HIDDEN,1),
        "GraphSAGE": lambda i: PySAGE(i,HIDDEN,1),
        "MPNN": lambda i: PyMPNN(i,HIDDEN,1),
        "MAGNet-v1": lambda i: MAGNetV1(i,HIDDEN,1),
    }
    # Models that use enriched features (36 atom + 12 edge)
    v2_models={
        "MAGNet-v2": lambda nd,ed: MAGNetV2(nd,ed,HIDDEN,1),
        "DMPNN": lambda nd,ed: DMPNN(nd,ed,HIDDEN,1),
    }
    
    all_results={}
    
    for ds_name in args.datasets:
        print(f"\n{'='*60}\n  Dataset: {ds_name}\n{'='*60}")
        raw=MoleculeNet(root=DATA_DIR,name=ds_name)
        is_mt=raw[0].y.dim()>0 and raw[0].y.shape[-1]>1
        in_dim=raw[0].x.shape[-1]
        print(f"  Molecules: {len(raw)}, Features: {in_dim}, Multi-task: {is_mt}")
        
        # Enrich
        print("  Enriching features...")
        enriched=ListDS(enrich(raw))
        nd=enriched[0].x.shape[1]; ed=enriched[0].edge_attr.shape[1]
        
        ds_results={}
        
        # v1 models (original features)
        for name,mk in v1_models.items():
            print(f"\n  >> {name}",end="",flush=True)
            aucs=[]
            for seed in range(args.seeds):
                tr,va,te=scaffold_split(raw,seed=seed)
                auc,params,elapsed=train_eval(mk(in_dim),raw,tr,va,te,DEVICE,seed,is_mt)
                aucs.append(np.nanmean(auc)); print(".",end="",flush=True)
            ds_results[name]={"mean":float(np.nanmean(aucs)),"std":float(np.nanstd(aucs)),
                              "params":params,"aucs":[float(a) for a in aucs]}
            print(f"  AUC={np.nanmean(aucs):.4f}+/-{np.nanstd(aucs):.4f}")
        
        # v2 models (enriched features)
        for name,mk in v2_models.items():
            print(f"\n  >> {name}",end="",flush=True)
            aucs=[]
            for seed in range(args.seeds):
                tr,va,te=scaffold_split(raw,seed=seed)
                auc,params,elapsed=train_eval(mk(nd,ed),enriched,tr,va,te,DEVICE,seed,is_mt)
                aucs.append(np.nanmean(auc)); print(".",end="",flush=True)
            ds_results[name]={"mean":float(np.nanmean(aucs)),"std":float(np.nanstd(aucs)),
                              "params":params,"aucs":[float(a) for a in aucs]}
            print(f"  AUC={np.nanmean(aucs):.4f}+/-{np.nanstd(aucs):.4f}")
        
        # RF-Morgan
        print(f"\n  >> RF-Morgan",end="",flush=True)
        rf_aucs=[]
        for seed in range(args.seeds):
            tr,va,te=scaffold_split(raw,seed=seed)
            auc=run_rf(raw,tr,va,te,seed)
            rf_aucs.append(np.nanmean(auc)); print(".",end="",flush=True)
        ds_results["RF-Morgan"]={"mean":float(np.nanmean(rf_aucs)),"std":float(np.nanstd(rf_aucs)),
                                 "params":0,"aucs":[float(a) for a in rf_aucs]}
        print(f"  AUC={np.nanmean(rf_aucs):.4f}+/-{np.nanstd(rf_aucs):.4f}")
        
        all_results[ds_name]=ds_results
        
        # Summary for this dataset
        print(f"\n  {'Model':20s}  {'AUC':>12s}  {'Params':>10s}")
        print(f"  {'-'*20}  {'-'*12}  {'-'*10}")
        for n,r in sorted(ds_results.items(),key=lambda x:-x[1]["mean"]):
            print(f"  {n:20s}  {r['mean']:.4f}+/-{r['std']:.4f}  {r['params']:>10,}")
    
    # Save
    os.makedirs(RESULTS_DIR,exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"molnet_v2_benchmark.json"),"w") as f:
        json.dump(all_results,f,indent=2)
    print(f"\nSaved to {RESULTS_DIR}/molnet_v2_benchmark.json")

if __name__=="__main__":
    main()
