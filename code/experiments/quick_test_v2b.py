#!/usr/bin/env python3
"""Quick test v2 only — skip MAGNet-v1 (already done: 0.8521 ± 0.0011)."""
import os, sys, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset as TDataset
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

DATA_DIR = os.path.join(ROOT, "data", "molnet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 128; MAX_EPOCHS = 30; PATIENCE = 7; BATCH = 64

# scaffold split
from collections import defaultdict
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
    n = len(dataset)
    n_tr = int(n*0.8); n_va = int(n*0.1)
    tr,va,te = [],[],[]
    for g in groups:
        if len(tr)+len(g)<=n_tr: tr+=g
        elif len(va)+len(g)<=n_va: va+=g
        else: te+=g
    seen=set(tr)|set(va)|set(te)
    for i in range(n):
        if i not in seen: te.append(i)
    return tr,va,te

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# RDKit features
def _atom_feat(atom):
    from rdkit import Chem
    nums=[6,7,8,16,9,17,35,53,15,0]
    at=[0]*10; at[nums.index(atom.GetAtomicNum()) if atom.GetAtomicNum() in nums else 9]=1
    deg=[0]*6; deg[min(atom.GetDegree(),5)]=1
    ch=[0]*5; ch[min(max(atom.GetFormalCharge()+2,0),4)]=1
    hyb_types=[Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3,Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2]
    hy=[0]*5; h=atom.GetHybridization()
    if h in hyb_types: hy[hyb_types.index(h)]=1
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

def _bond_feat(bond):
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

def enrich_dataset(dataset):
    from rdkit import Chem
    enriched=[]
    for i in range(len(dataset)):
        d=dataset[i]; smi=d.smiles if isinstance(d.smiles,str) else d.smiles[0]
        mol=Chem.MolFromSmiles(smi)
        if mol is None: continue
        d.x=torch.tensor([_atom_feat(a) for a in mol.GetAtoms()],dtype=torch.float)
        ei,ea=[],[]
        for bond in mol.GetBonds():
            a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            bf=_bond_feat(bond); ei+=[[a,b],[b,a]]; ea+=[bf,bf]
        if ei:
            d.edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous()
            d.edge_attr=torch.tensor(ea,dtype=torch.float)
        else:
            d.edge_index=torch.zeros(2,0,dtype=torch.long)
            d.edge_attr=torch.zeros(0,12,dtype=torch.float)
        d.num_nodes=d.x.size(0); enriched.append(d)
    return enriched

class ListDS(torch.utils.data.Dataset):
    def __init__(self,L): self.L=L
    def __len__(self): return len(self.L)
    def __getitem__(self,i):
        if isinstance(i,(list,np.ndarray)): return [self.L[j] for j in i]
        return self.L[i]

# EdgeMPNN
class EdgeMPNN(MessagePassing):
    def __init__(self,nd,ed,hd):
        super().__init__(aggr='add')
        self.msg=nn.Sequential(nn.Linear(nd+ed,hd),nn.ReLU(),nn.Linear(hd,hd))
        self.upd=nn.GRUCell(hd,nd)
    def forward(self,x,ei,ea):
        return self.upd(self.propagate(ei,x=x,edge_attr=ea),x)
    def message(self,x_j,edge_attr):
        return self.msg(torch.cat([x_j,edge_attr],-1))

# MAGNet v2
class MAGNetV2(nn.Module):
    def __init__(self,nd=36,ed=12,hd=128,out=1,ns=3,nh=4,nq=4,dp=0.1):
        super().__init__()
        self.ns=ns; self.nq=nq
        self.np=nn.Linear(nd,hd); self.ep=nn.Linear(ed,hd)
        self.convs=nn.ModuleList([EdgeMPNN(hd,hd,hd) for _ in range(ns)])
        self.norms=nn.ModuleList([nn.LayerNorm(hd) for _ in range(ns)])
        self.gl=nn.Linear(hd*ns,ns)
        self.ca=nn.MultiheadAttention(hd,nh,dropout=dp,batch_first=True)
        self.cn=nn.LayerNorm(hd)
        self.q=nn.Parameter(torch.randn(nq,hd)*0.02)
        self.pa=nn.MultiheadAttention(hd,nh,dropout=dp,batch_first=True)
        self.pp=nn.Linear(hd*nq,hd); self.pn=nn.LayerNorm(hd)
        self.clf=nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(dp),nn.Linear(hd//2,out))
    def forward(self,data):
        x,ei,batch=data.x.float(),data.edge_index,data.batch
        ea=data.edge_attr.float() if hasattr(data,'edge_attr') and data.edge_attr is not None else torch.zeros(ei.size(1),12,device=x.device)
        h=F.relu(self.np(x)); e=self.ep(ea)
        sf=[]; cur=h
        for conv,norm in zip(self.convs,self.norms):
            cur=conv(cur,ei,e); sf.append(cur)
        stacked=torch.stack(sf,1); concat=stacked.reshape(stacked.size(0),-1)
        gate=torch.softmax(self.gl(concat),-1)
        gated=(stacked*gate.unsqueeze(-1)).sum(1)
        q=gated.unsqueeze(1); ao,_=self.ca(q,stacked,stacked)
        fused=self.cn(ao.squeeze(1)+gated)
        bs=int(batch.max().item())+1; gl=[]
        for i in range(bs):
            m=batch==i; nf=fused[m].unsqueeze(0); qe=self.q.unsqueeze(0)
            po,_=self.pa(qe,nf,nf); gl.append(po.reshape(1,-1))
        gr=torch.cat(gl,0); gr=F.relu(self.pp(gr)); gr=self.pn(gr)
        return self.clf(gr).squeeze(-1)

# DMPNN
class DMPNNLayer(MessagePassing):
    def __init__(self,nd,ed,hd):
        super().__init__(aggr='add')
        self.msg=nn.Sequential(nn.Linear(nd+ed,hd),nn.ReLU(),nn.Linear(hd,hd))
        self.upd=nn.GRUCell(hd,nd)
    def forward(self,x,ei,ea):
        return self.upd(self.propagate(ei,x=x,edge_attr=ea),x)
    def message(self,x_j,edge_attr):
        return self.msg(torch.cat([x_j,edge_attr],-1))

class DMPNN(nn.Module):
    def __init__(self,nd=36,ed=12,hd=128,out=1,nl=3,dp=0.1):
        super().__init__()
        self.np=nn.Linear(nd,hd); self.ep=nn.Linear(ed,hd)
        self.layers=nn.ModuleList([DMPNNLayer(hd,hd,hd) for _ in range(nl)])
        self.fc=nn.Linear(hd,out); self.dp=dp
    def forward(self,data):
        x,ei,batch=data.x.float(),data.edge_index,data.batch
        ea=data.edge_attr.float() if hasattr(data,'edge_attr') and data.edge_attr is not None else torch.zeros(ei.size(1),12,device=x.device)
        x=F.relu(self.np(x)); e=self.ep(ea)
        for l in self.layers:
            x=l(x,ei,e); x=F.dropout(x,self.dp,training=self.training)
        x=global_mean_pool(x,batch)
        return self.fc(x).squeeze(-1)

def train_eval(model,dataset,tr,va,te,device,seed):
    set_seed(seed); model=model.to(device)
    tl=DataLoader(dataset[tr],BATCH,shuffle=True)
    vl=DataLoader(dataset[va],BATCH)
    tstl=DataLoader(dataset[te],BATCH)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,MAX_EPOCHS)
    best_val,best_st,noimp=-1,None,0
    t0=time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        for batch in tl:
            batch=batch.to(device); opt.zero_grad(True)
            logits=model(batch)
            y=batch.y.float().view(-1); mask=~torch.isnan(y); y=torch.nan_to_num(y,0)
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*mask).sum()/mask.clamp(1e-8).sum()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step()
        model.eval(); preds,labs=[],[]
        with torch.no_grad():
            for batch in vl:
                batch=batch.to(device)
                preds.append(torch.sigmoid(model(batch)).cpu().numpy().reshape(-1))
                labs.append(batch.y.float().cpu().numpy().reshape(-1))
        p,l=np.concatenate(preds),np.concatenate(labs); m=~np.isnan(l)
        va_=roc_auc_score(l[m],p[m]) if m.sum()>1 and len(np.unique(l[m]))>1 else 0
        if va_>best_val: best_val=va_; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; noimp=0
        else: noimp+=1
        if noimp>=PATIENCE: break
    elapsed=time.time()-t0
    model.load_state_dict(best_st); model.eval()
    preds,labs=[],[]
    with torch.no_grad():
        for batch in tstl:
            batch=batch.to(device)
            preds.append(torch.sigmoid(model(batch)).cpu().numpy().reshape(-1))
            labs.append(batch.y.float().cpu().numpy().reshape(-1))
    p,l=np.concatenate(preds),np.concatenate(labs); m=~np.isnan(l)
    auc=roc_auc_score(l[m],p[m]) if m.sum()>1 and len(np.unique(l[m]))>1 else 0
    params=sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    return auc,params,elapsed

def main():
    print(f"Device: {DEVICE}")
    ds=MoleculeNet(root=DATA_DIR,name="BBBP")
    print(f"BBBP: {len(ds)} mols")
    print("Enriching features...")
    enriched=enrich_dataset(ds)
    ed=ListDS(enriched)
    n2=enriched[0].x.shape[1]; e2=enriched[0].edge_attr.shape[1]
    print(f"Enriched: {n2} atom, {e2} edge")

    # Load previous MAGNet-v1 results
    results = {"MAGNet-v1": {"mean": 0.8521, "std": 0.0011, "params": 259844, "aucs": [0.8532, 0.8510]}}
    print("MAGNet-v1 (cached): 0.8521 ± 0.0011")

    models={
        "MAGNet-v2": (lambda: MAGNetV2(n2,e2,HIDDEN,1), ed),
        "DMPNN":     (lambda: DMPNN(n2,e2,HIDDEN,1), ed),
    }

    for name,(fn,use_ds) in models.items():
        print(f"\n>>> {name}")
        aucs=[]
        for seed in range(2):
            tr,va,te=scaffold_split(ds,seed=seed)
            auc,params,t=train_eval(fn(),use_ds,tr,va,te,DEVICE,seed)
            aucs.append(auc)
            print(f"  Seed {seed}: AUC={auc:.4f}  {t:.0f}s  params={params:,}")
        results[name]={"mean":float(np.mean(aucs)),"std":float(np.std(aucs)),"params":params,"aucs":[float(a) for a in aucs]}
        print(f"  -> {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    print(f"\n{'='*50}")
    print("  BBBP Results (2 seeds)")
    print(f"{'='*50}")
    for n,r in sorted(results.items(),key=lambda x:-x[1]["mean"]):
        print(f"  {n:20s}  {r['mean']:.4f} +/- {r['std']:.4f}  ({r['params']:,} params)")

    os.makedirs(os.path.join(ROOT,"results"),exist_ok=True)
    with open(os.path.join(ROOT,"results","v2_quicktest.json"),"w") as f:
        json.dump(results,f,indent=2)
    print("\nDone")

if __name__=="__main__":
    main()
