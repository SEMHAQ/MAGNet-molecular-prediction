"""
Comprehensive experiment runner for CompBioChem paper.

Runs experiments comparing MAGNet against baselines (GCN, GAT, MLP)
on molecular property prediction tasks.

Target journal: Computational Biology and Chemistry (Elsevier, IF 3.29)
"""

import os
import sys
import yaml
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from code.models.baseline import GNNModel, BaselineModel
from code.models.magnet import MAGNet, MAGNetClassifier
from code.models.baselines_compat import BatchedGCN, BatchedGAT, BatchedMLP
from code.data.molecular_dataset import SyntheticMolecularDataset, create_molecular_datasets
from code.data.dataset import create_dataloaders
from code.utils.metrics import compute_classification_metrics, format_metrics_table
from code.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_comparison_bar,
    plot_ablation_study,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MolecularTrainer:
    """
    Trainer for molecular property prediction models.

    Handles training, validation, and evaluation with proper logging.
    Uses AUROC for early stopping and class-weighted loss.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 10,
        pos_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        self.best_val_metric = 0.0
        self.best_model_state = None
        self.patience = patience
        self.patience_counter = 0

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            node_features, adj, labels = batch
            node_features = node_features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(node_features, adj)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            correct += (predictions == labels.long()).sum().item()
            total += labels.size(0)

        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
        }

    def evaluate(self, data_loader) -> Dict[str, Any]:
        """Evaluate model on data."""
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in data_loader:
                node_features, adj, labels = batch
                node_features = node_features.to(self.device)
                adj = adj.to(self.device)

                outputs = self.model(node_features, adj)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).long()

                all_labels.extend(labels.numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        metrics = compute_classification_metrics(
            all_labels, all_predictions, all_probabilities
        )

        return metrics

    def validate(self, val_loader) -> float:
        """Quick validation returning AUROC for early stopping."""
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                node_features, adj, labels = batch
                node_features = node_features.to(self.device)
                adj = adj.to(self.device)
                outputs = self.model(node_features, adj)
                probs = torch.sigmoid(outputs)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        try:
            return roc_auc_score(all_labels, all_probs)
        except ValueError:
            return 0.5

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping based on AUROC."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_auroc': [],
        }

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])

            # Validate with AUROC
            val_auroc = self.validate(val_loader)
            history['val_auroc'].append(val_auroc)

            # Early stopping based on AUROC
            if val_auroc > self.best_val_metric:
                self.best_val_metric = val_auroc
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val AUROC: {val_auroc:.4f}"
                )

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        return history


def create_model(
    model_name: str,
    in_dim: int,
    hidden_dim: int = 128,
    **kwargs,
) -> nn.Module:
    """Create model by name."""
    if model_name == 'magnet':
        return MAGNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_scales=kwargs.get('num_scales', 3),
            num_heads=kwargs.get('num_heads', 4),
            num_queries=kwargs.get('num_queries', 4),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_name == 'gcn':
        return BatchedGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=kwargs.get('num_layers', 3),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_name == 'gat':
        return BatchedGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=kwargs.get('num_layers', 3),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_name == 'mlp':
        return BatchedMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            dropout=kwargs.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(
    model_name: str,
    train_dataset,
    val_dataset,
    test_dataset,
    device: torch.device,
    config: Dict,
) -> Dict[str, Any]:
    """Run a single experiment with one model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Get input dimension
    sample = train_dataset[0]
    in_dim = sample[0].shape[1]  # node feature dimension

    # Compute pos_weight from training set
    labels_np = np.array(train_dataset.labels)
    n_pos = labels_np.sum()
    n_neg = len(labels_np) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"Class balance: {n_pos} positive, {n_neg} negative, pos_weight={pos_weight:.3f}")

    # Create model
    model = create_model(
        model_name=model_name,
        in_dim=in_dim,
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        max_nodes=train_dataset.max_nodes,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
    )

    # Create trainer with class weighting
    trainer = MolecularTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['early_stopping'],
        pos_weight=pos_weight,
    )

    # Train
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
    )
    train_time = time.time() - start_time

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)

    logger.info(f"\n{model_name.upper()} Results:")
    logger.info(f"Training time: {train_time:.2f}s")
    logger.info(format_metrics_table(test_metrics, title=f"{model_name.upper()} Test Results"))

    return {
        'model_name': model_name,
        'history': history,
        'test_metrics': test_metrics,
        'train_time': train_time,
        'best_val_auroc': trainer.best_val_metric,
    }


def run_ablation_study(
    train_dataset,
    val_dataset,
    test_dataset,
    device: torch.device,
    config: Dict,
) -> Dict[str, Any]:
    """Run ablation study on MAGNet components."""
    logger.info("\n" + "="*60)
    logger.info("Running Ablation Study")
    logger.info("="*60)

    # Compute pos_weight
    labels_np = np.array(train_dataset.labels)
    n_pos = labels_np.sum()
    n_neg = len(labels_np) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    results = {}

    # Ablation 1: Number of scales
    for num_scales in [1, 2, 3, 4]:
        logger.info(f"\nAblation: num_scales={num_scales}")

        model = MAGNet(
            in_dim=train_dataset[0][0].shape[1],
            hidden_dim=config['model']['hidden_dim'],
            out_dim=1,
            num_scales=num_scales,
            num_heads=4,
            num_queries=4,
            dropout=config['model']['dropout'],
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
        )

        trainer = MolecularTrainer(
            model=model,
            device=device,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            patience=config['training']['early_stopping'],
            pos_weight=pos_weight,
        )

        history = trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])
        test_metrics = trainer.evaluate(test_loader)

        results[f'scales_{num_scales}'] = {
            'num_scales': num_scales,
            'test_metrics': test_metrics,
            'best_val_auroc': trainer.best_val_metric,
        }

    # Ablation 2: Number of attention heads
    for num_heads in [1, 2, 4, 8]:
        logger.info(f"\nAblation: num_heads={num_heads}")
        model = MAGNet(
            in_dim=train_dataset[0][0].shape[1],
            hidden_dim=config['model']['hidden_dim'],
            out_dim=1,
            num_scales=3,
            num_heads=num_heads,
            num_queries=4,
            dropout=config['model']['dropout'],
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
        )

        trainer = MolecularTrainer(
            model=model,
            device=device,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            patience=config['training']['early_stopping'],
            pos_weight=pos_weight,
        )

        history = trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])
        test_metrics = trainer.evaluate(test_loader)

        results[f'heads_{num_heads}'] = {
            'num_heads': num_heads,
            'test_metrics': test_metrics,
            'best_val_auroc': trainer.best_val_metric,
        }

    return results


def generate_figures(
    baseline_results: List[Dict],
    ablation_results: Dict,
    output_dir: str,
):
    """Generate publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Publication style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })
    colors = {'magnet': '#E63946', 'gcn': '#457B9D', 'gat': '#2A9D8F', 'mlp': '#264653'}

    # Figure 1: Training curves comparison (loss + AUROC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for result in baseline_results:
        name = result['model_name']
        c = colors.get(name, '#333333')
        axes[0].plot(result['history']['train_loss'], label=name.upper(), color=c, linewidth=2)
        axes[1].plot(result['history']['val_auroc'], label=name.upper(), color=c, linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('(a) Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation AUROC')
    axes[1].set_title('(b) Validation AUROC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), bbox_inches='tight')
    plt.close()

    # Figure 2: Model comparison bar chart (multi-metric)
    metrics_to_plot = ['accuracy', 'auc', 'f1', 'mcc']
    metric_labels = ['Accuracy', 'AUROC', 'F1', 'MCC']
    model_names = [r['model_name'] for r in baseline_results]
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, result in enumerate(baseline_results):
        name = result['model_name']
        c = colors.get(name, '#333333')
        values = [result['test_metrics'].get(m, 0.0) for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name.upper(), color=c, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'model_comparison.pdf'), bbox_inches='tight')
    plt.close()

    # Figure 3: Ablation -- scales
    if ablation_results:
        scales_results = {k: v for k, v in ablation_results.items() if k.startswith('scales_')}
        if scales_results:
            fig, ax = plt.subplots(figsize=(7, 5))
            scales = [v['num_scales'] for v in scales_results.values()]
            aurocs = [v['test_metrics'].get('auc', 0.0) for v in scales_results.values()]
            accs = [v['test_metrics']['accuracy'] for v in scales_results.values()]
            ax.plot(scales, aurocs, 'o-', linewidth=2, markersize=8, color='#E63946', label='AUROC')
            ax.plot(scales, accs, 's--', linewidth=2, markersize=8, color='#457B9D', label='Accuracy')
            for s, a in zip(scales, aurocs):
                ax.annotate(f'{a:.3f}', (s, a), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10)
            ax.set_xlabel('Number of Scales')
            ax.set_ylabel('Score')
            ax.set_title('Ablation: Effect of Multi-scale Feature Extraction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(scales)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ablation_scales.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'ablation_scales.pdf'), bbox_inches='tight')
            plt.close()

        # Figure 4: Ablation -- heads
        heads_results = {k: v for k, v in ablation_results.items() if k.startswith('heads_')}
        if heads_results:
            fig, ax = plt.subplots(figsize=(7, 5))
            heads = [v['num_heads'] for v in heads_results.values()]
            aurocs = [v['test_metrics'].get('auc', 0.0) for v in heads_results.values()]
            accs = [v['test_metrics']['accuracy'] for v in heads_results.values()]
            ax.plot(heads, aurocs, 'o-', linewidth=2, markersize=8, color='#E63946', label='AUROC')
            ax.plot(heads, accs, 's--', linewidth=2, markersize=8, color='#457B9D', label='Accuracy')
            for h, a in zip(heads, aurocs):
                ax.annotate(f'{a:.3f}', (h, a), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10)
            ax.set_xlabel('Number of Attention Heads')
            ax.set_ylabel('Score')
            ax.set_title('Ablation: Effect of Attention Heads')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(heads)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ablation_heads.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'ablation_heads.pdf'), bbox_inches='tight')
            plt.close()


def main():
    """Main experiment runner."""
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['experiment']['seed'])

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create output directories
    result_dir = config['output']['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    figure_dir = os.path.join(result_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    # Create datasets
    logger.info("Creating synthetic molecular datasets...")
    train_dataset, val_dataset, test_dataset = create_molecular_datasets(
        n_train=2000,
        n_val=400,
        n_test=400,
        property_type='toxicity',
        difficulty='medium',
        seed=config['experiment']['seed'],
    )

    logger.info(f"Train: {len(train_dataset)} molecules")
    logger.info(f"Val: {len(val_dataset)} molecules")
    logger.info(f"Test: {len(test_dataset)} molecules")
    logger.info(f"Dataset stats: {train_dataset.get_statistics()}")

    # Run experiments with different models
    models_to_test = ['magnet', 'gcn', 'gat', 'mlp']
    baseline_results = []

    for model_name in models_to_test:
        result = run_single_experiment(
            model_name=model_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            config=config,
        )
        baseline_results.append(result)

    # Run ablation study
    ablation_results = run_ablation_study(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        config=config,
    )

    # Generate figures
    logger.info("\nGenerating figures...")
    generate_figures(baseline_results, ablation_results, figure_dir)

    # Save results
    results_file = os.path.join(result_dir, 'experiment_results.json')

    # Helper function to make metrics JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': make_serializable(config),
        'baseline_results': [
            {
                'model_name': r['model_name'],
                'test_metrics': {k: v for k, v in r['test_metrics'].items()
                               if k not in ('confusion_matrix', 'roc_curve', 'pr_curve')},
                'train_time': float(r['train_time']),
                'best_val_auroc': float(r['best_val_auroc']),
            }
            for r in baseline_results
        ],
        'ablation_results': {
            k: {
                'num_scales': v.get('num_scales'),
                'num_heads': v.get('num_heads'),
                'test_metrics': {k2: v2 for k2, v2 in v['test_metrics'].items()
                               if k2 not in ('confusion_matrix', 'roc_curve', 'pr_curve')},
                'best_val_auroc': float(v['best_val_auroc']),
            }
            for k, v in ablation_results.items()
        },
    }

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"Figures saved to {figure_dir}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    for result in baseline_results:
        logger.info(f"\n{result['model_name'].upper()}:")
        logger.info(f"  Accuracy: {result['test_metrics']['accuracy']:.4f}")
        logger.info(f"  AUROC: {result['test_metrics'].get('auc', 'N/A')}")
        logger.info(f"  Training time: {result['train_time']:.2f}s")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
