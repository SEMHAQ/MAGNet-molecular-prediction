"""
Main experiment script for CompBioChem project.
Target journal: Computational Biology and Chemistry (Elsevier, IF 3.29)
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
import torch
import random
from pathlib import Path

from code.models.baseline import BaselineModel, GNNModel
from code.data.dataset import SampleBiologicalDataset, create_dataloaders
from code.experiments.trainer import Trainer, EarlyStopping
from code.utils.metrics import compute_classification_metrics, format_metrics_table
from code.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
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


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_str: str = "cuda") -> torch.device:
    """Get PyTorch device."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Get device
    device = get_device(config["experiment"]["device"])
    logger.info(f"Using device: {device}")

    # Create output directories
    for dir_key in ["checkpoint_dir", "log_dir", "result_dir"]:
        os.makedirs(config["output"][dir_key], exist_ok=True)

    logger.info(f"Experiment: {config['experiment']['name']}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Create synthetic datasets (replace with real data loading)
    train_dataset = SampleBiologicalDataset(
        n_samples=800,
        n_features=model_cfg["input_dim"],
        n_classes=2,
        noise=0.1,
        seed=config["experiment"]["seed"],
    )
    val_dataset = SampleBiologicalDataset(
        n_samples=100,
        n_features=model_cfg["input_dim"],
        n_classes=2,
        noise=0.1,
        seed=config["experiment"]["seed"] + 1,
    )
    test_dataset = SampleBiologicalDataset(
        n_samples=100,
        n_features=model_cfg["input_dim"],
        n_classes=2,
        noise=0.1,
        seed=config["experiment"]["seed"] + 2,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = BaselineModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        dropout=model_cfg["dropout"],
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # -----------------------------------------------------------------------
    # Optimizer, Loss, Scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    early_stopping = EarlyStopping(
        patience=train_cfg["early_stopping"],
        mode="max",
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        early_stopping=early_stopping,
        gradient_clip_val=1.0,
        checkpoint_dir=config["output"]["checkpoint_dir"],
        experiment_name=config["experiment"]["name"],
    )

    # Build scheduler
    trainer._build_scheduler(
        train_cfg["scheduler"],
        total_steps=train_cfg["epochs"] * len(train_loader),
        T_max=train_cfg["epochs"],
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    if args.mode == "train":
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_cfg["epochs"],
            val_metric="accuracy",
            val_metric_mode="max",
            log_interval=1,
        )

        # Plot training curves
        plot_training_curves(
            train_losses=history["train_loss"],
            val_losses=history["val_loss"],
            train_metrics=history["train_acc"],
            val_metrics=history["val_acc"],
            metric_name="Accuracy",
            save_path=os.path.join(config["output"]["result_dir"], "training_curves.png"),
        )

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    trainer.load_best_model()
    results = trainer.predict(test_loader)

    metrics = compute_classification_metrics(
        results["labels"],
        results["predictions"],
        results["probabilities"],
    )
    logger.info(format_metrics_table(metrics, title="Test Results"))

    # Plot confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=["Negative", "Positive"],
        save_path=os.path.join(config["output"]["result_dir"], "confusion_matrix.png"),
    )

    # Plot ROC curve
    if "roc_curve" in metrics:
        plot_roc_curve(
            metrics["roc_curve"]["fpr"],
            metrics["roc_curve"]["tpr"],
            auc_value=metrics.get("auc"),
            save_path=os.path.join(config["output"]["result_dir"], "roc_curve.png"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CompBioChem Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "test"],
        help="Experiment mode",
    )
    args = parser.parse_args()

    main(args)
