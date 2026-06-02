# MAGNet: Multi-scale Attention Graph Network for Molecular Property Prediction

A PyTorch implementation of MAGNet for molecular property prediction, targeting submission to **Computational Biology and Chemistry** (Elsevier, IF 3.29).

## Overview

MAGNet addresses the limitation of single-scale message passing in existing GNN-based molecular predictors by combining:

- **Multi-scale GNN backbone** — extracts features at 1-hop, 2-hop, and 3-hop neighborhoods
- **Cross-scale attention fusion** — learns to weight and combine features from different scales
- **Global attention pooling** — uses learnable query vectors for graph-level representation
- **Efficient design** — O(N) complexity per scale vs O(N²) for full attention

## Project Structure

```
CompBioChem/
├── paper/                      # LaTeX manuscript (Elsevier CAS template)
│   ├── main.tex
│   └── references.bib
├── code/
│   ├── models/
│   │   ├── magnet.py           # MAGNet model
│   │   ├── baseline.py         # Traditional ML baselines
│   │   └── baselines_compat.py # GNN baselines (GCN, GAT, GraphSAGE)
│   ├── data/
│   │   ├── dataset.py          # Base dataset
│   │   └── molecular_dataset.py # Molecular graph dataset
│   ├── experiments/
│   │   ├── run_experiments.py  # Main experiment runner
│   │   └── trainer.py          # Training loop
│   └── utils/
│       ├── metrics.py          # AUROC, F1, accuracy, etc.
│       └── visualization.py    # Plotting utilities
├── results/                    # Experiment outputs
│   ├── experiment_results.json
│   └── figures/                # Generated plots
├── els-cas-templates/          # Elsevier CAS LaTeX template (DO NOT modify)
├── docs/
│   ├── research_plan.md        # Research design and contributions
│   ├── notes.md                # Journal info and notes
│   └── timeline.md             # Project timeline
└── config.yaml                 # Experiment configuration
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
cd code
python -m experiments.run_experiments

# Run with specific device
python -m experiments.run_experiments --device cuda
```

## Configuration

Edit `config.yaml` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.num_scales` | 3 | Number of graph convolution scales |
| `model.num_heads` | 4 | Attention heads in cross-scale fusion |
| `model.hidden_dim` | 128 | Hidden feature dimension |
| `training.epochs` | 50 | Training epochs |
| `data.n_train` | 2000 | Training molecules |

## Baselines

- **Traditional ML**: Random Forest, SVM, XGBoost, MLP
- **GNN**: GCN, GAT, GraphSAGE
- **Metrics**: AUROC, Accuracy, F1-score, Precision, Recall

## Target Journal

- **Journal**: Computational Biology and Chemistry
- **Publisher**: Elsevier
- **Impact Factor**: 3.29 (2026)
- **ISSN**: 1476-9271
- **Open Access**: No (no page charges)

## Citation

```bibtex
@article{magnet2026,
  title={MAGNet: Multi-scale Attention Graph Network for Molecular Property Prediction},
  author={Yu, Huanjie},
  journal={Computational Biology and Chemistry},
  year={2026}
}
```

## License

For academic research use only.
