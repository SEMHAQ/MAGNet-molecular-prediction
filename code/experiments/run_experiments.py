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
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss()
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

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])

            # Validate
            val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_metrics.get('loss', 0.0))
            history['val_acc'].append(val_metrics['accuracy'])

            # Early stopping
            val_metric = val_metrics['accuracy']
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_model_state = self.model.state_dict().copy()
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
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

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

    # Create trainer
    trainer = MolecularTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['early_stopping'],
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
        'best_val_acc': trainer.best_val_metric,
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

    results = {}

    # Ablation 1: Number of scales
    for num_scales in [1, 2, 3, 4]:
        logger.info(f"\nAblation: num_scales={num_scales}")
        config_copy = config.copy()
        config_copy['model'] = config['model'].copy()

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
        )

        history = trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])
        test_metrics = trainer.evaluate(test_loader)

        results[f'scales_{num_scales}'] = {
            'num_scales': num_scales,
            'test_metrics': test_metrics,
            'best_val_acc': trainer.best_val_metric,
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
        )

        history = trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])
        test_metrics = trainer.evaluate(test_loader)

        results[f'heads_{num_heads}'] = {
            'num_heads': num_heads,
            'test_metrics': test_metrics,
            'best_val_acc': trainer.best_val_metric,
        }

    return results


def generate_figures(
    baseline_results: List[Dict],
    ablation_results: Dict,
    output_dir: str,
):
    """Generate publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Training curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for result in baseline_results:
        axes[0].plot(result['history']['train_loss'], label=result['model_name'])
        axes[1].plot(result['history']['val_acc'], label=result['model_name'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Validation Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Model comparison bar chart
    model_names = [r['model_name'] for r in baseline_results]
    accuracies = [r['test_metrics']['accuracy'] for r in baseline_results]
    aurocs = [r['test_metrics'].get('auc', 0.0) for r in baseline_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2E86AB')
    bars2 = ax.bar(x + width/2, aurocs, width, label='AUROC', color='#A23B72')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in model_names])
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Ablation study
    if ablation_results:
        # Scales ablation
        scales_results = {k: v for k, v in ablation_results.items() if k.startswith('scales_')}
        if scales_results:
            fig, ax = plt.subplots(figsize=(8, 6))
            scales = [v['num_scales'] for v in scales_results.values()]
            accs = [v['test_metrics']['accuracy'] for v in scales_results.values()]
            ax.plot(scales, accs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            ax.set_xlabel('Number of Scales')
            ax.set_ylabel('Test Accuracy')
            ax.set_title('Ablation: Effect of Multi-scale Feature Extraction')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(scales)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ablation_scales.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Heads ablation
        heads_results = {k: v for k, v in ablation_results.items() if k.startswith('heads_')}
        if heads_results:
            fig, ax = plt.subplots(figsize=(8, 6))
            heads = [v['num_heads'] for v in heads_results.values()]
            accs = [v['test_metrics']['accuracy'] for v in heads_results.values()]
            ax.plot(heads, accs, 'o-', linewidth=2, markersize=8, color='#A23B72')
            ax.set_xlabel('Number of Attention Heads')
            ax.set_ylabel('Test Accuracy')
            ax.set_title('Ablation: Effect of Attention Heads')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(heads)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ablation_heads.png'), dpi=300, bbox_inches='tight')
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
    models_to_test = ['magnet', 'gcn', 'gat']
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
                'best_val_acc': float(r['best_val_acc']),
            }
            for r in baseline_results
        ],
        'ablation_results': {
            k: {
                'num_scales': v.get('num_scales'),
                'num_heads': v.get('num_heads'),
                'test_metrics': {k2: v2 for k2, v2 in v['test_metrics'].items()
                               if k2 not in ('confusion_matrix', 'roc_curve', 'pr_curve')},
                'best_val_acc': float(v['best_val_acc']),
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
