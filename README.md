# MAGNet

**Multi-scale Attention Graph Network for Molecular Property Prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

## Introduction

MAGNet is a graph neural network for predicting molecular properties (e.g. toxicity, solubility, binding affinity). It addresses the limitation of single-scale message passing in existing GNNs by extracting and fusing features at multiple graph depths.

**Key features:**
- Multi-scale GNN backbone (1-hop, 2-hop, 3-hop neighborhoods)
- Cross-scale attention fusion with learned weighting
- Global attention pooling via learnable query vectors
- O(N) complexity per scale, significantly more efficient than full-attention Graph Transformers

## Installation

```bash
git clone https://github.com/SEMHAQ/MAGNet-molecular-prediction.git
cd MAGNet-molecular-prediction
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, CUDA (optional but recommended)

## Usage

### Run Experiments

```bash
cd code
python -m experiments.run_experiments
```

### Configuration

All hyperparameters are in `config.yaml`:

```yaml
model:
  num_scales: 3        # number of graph convolution scales
  num_heads: 4         # attention heads in cross-scale fusion
  hidden_dim: 128      # hidden feature dimension
  num_queries: 4       # learnable pooling queries
  dropout: 0.1

training:
  epochs: 50
  learning_rate: 0.001
  scheduler: cosine
  early_stopping: 10
```

## Architecture

```
Input: Molecular Graph (atoms, bonds)
  │
  ├─► Scale 1: 1-hop GNN  ─┐
  ├─► Scale 2: 2-hop GNN  ─┼─► Cross-Scale Attention Fusion
  └─► Scale 3: 3-hop GNN  ─┘
                                │
                                ▼
                         Attention Pooling (learnable queries)
                                │
                                ▼
                         MLP Head → Prediction
```

## Baselines

| Model | Type | Description |
|-------|------|-------------|
| MLP | Traditional | Molecular fingerprint + MLP |
| GCN | GNN | Graph Convolutional Network |
| GAT | GNN | Graph Attention Network |
| GraphSAGE | GNN | Sample-and-aggregate framework |

**Metrics:** AUROC, Accuracy, F1-score, Precision, Recall, MCC

## Project Structure

```
├── code/
│   ├── models/
│   │   ├── magnet.py           # MAGNet (proposed method)
│   │   ├── baseline.py         # Traditional ML baselines
│   │   └── baselines_compat.py # GNN baselines (GCN, GAT, SAGE)
│   ├── data/
│   │   └── molecular_dataset.py # Molecular graph dataset
│   ├── experiments/
│   │   ├── run_experiments.py   # Main experiment runner
│   │   └── trainer.py           # Training loop
│   └── utils/
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Plotting utilities
├── paper/
│   ├── main.tex                 # LaTeX manuscript
│   └── references.bib           # Bibliography
├── config.yaml                  # Hyperparameters
└── requirements.txt             # Dependencies
```

## Citation

```bibtex
@article{yu2026magnet,
  title   = {MAGNet: Multi-scale Attention Graph Network for Molecular Property Prediction},
  author  = {Yu, Huanjie},
  journal = {Computational Biology and Chemistry},
  year    = {2026}
}
```

## License

Academic use only.
