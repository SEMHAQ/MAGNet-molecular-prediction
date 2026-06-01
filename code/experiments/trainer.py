"""
Trainer module with complete training loop, validation, early stopping,
learning rate scheduling, and checkpoint management.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    OneCycleLR,
)
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
from pathlib import Path

from code.utils.helpers import save_checkpoint, load_checkpoint
from code.utils.metrics import compute_classification_metrics, MetricsTracker

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss, 'max' for accuracy/F1.
        verbose: Whether to print messages.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("EarlyStopping: stopping training")

        return self.early_stop


class Trainer:
    """
    Complete training loop with validation, early stopping, LR scheduling,
    gradient clipping, and checkpoint management.

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to train on ('cuda' or 'cpu').
        scheduler: Learning rate scheduler (optional).
        early_stopping: EarlyStopping instance (optional).
        gradient_clip_val: Max gradient norm for clipping. 0 to disable.
        checkpoint_dir: Directory to save checkpoints.
        experiment_name: Name for this experiment run.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        gradient_clip_val: float = 0.0,
        checkpoint_dir: str = "./checkpoints",
        experiment_name: str = "experiment",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.gradient_clip_val = gradient_clip_val
        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.metrics_tracker = MetricsTracker()
        self.best_val_score: Optional[float] = None
        self.current_epoch = 0

    def _build_scheduler(
        self,
        scheduler_name: str,
        total_steps: int,
        **kwargs,
    ):
        """Build a learning rate scheduler by name."""
        schedulers = {
            "cosine": lambda: CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get("T_max", 100),
                eta_min=kwargs.get("eta_min", 1e-6),
            ),
            "plateau": lambda: ReduceLROnPlateau(
                self.optimizer,
                mode=kwargs.get("mode", "max"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5),
                verbose=True,
            ),
            "step": lambda: StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1),
            ),
            "onecycle": lambda: OneCycleLR(
                self.optimizer,
                max_lr=kwargs.get("max_lr", 0.01),
                total_steps=total_steps,
            ),
        }
        if scheduler_name not in schedulers:
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Choose from {list(schedulers.keys())}")
        self.scheduler = schedulers[scheduler_name]()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Run one training epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary of training metrics (loss, etc.).
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        for batch in train_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            else:
                raise ValueError("Batch must be a tuple of (inputs, labels)")

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

            self.optimizer.step()

            # Step OneCycleLR per batch
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                preds = (torch.sigmoid(outputs) > 0.5).long().ravel()
            else:
                preds = outputs.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        avg_loss = total_loss / max(n_batches, 1)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds == all_labels).float().mean().item()

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        n_batches = 0

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            else:
                raise ValueError("Batch must be a tuple of (inputs, labels)")

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions and probabilities
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().ravel()
                all_probs.append(probs.detach().cpu())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                all_probs.append(probs.detach().cpu())

            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        avg_loss = total_loss / max(n_batches, 1)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)

        # Compute comprehensive metrics
        metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = avg_loss

        # Remove non-serializable entries for tracking
        trackable = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k not in ("confusion_matrix",)
        }
        self.metrics_tracker.update(trackable, self.current_epoch)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        val_metric: str = "accuracy",
        val_metric_mode: str = "max",
        log_interval: int = 1,
        save_best: bool = True,
        save_last: bool = True,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Total number of epochs.
            val_metric: Validation metric to track for best model / early stopping.
            val_metric_mode: 'max' or 'min' for val_metric.
            log_interval: Log every N epochs.
            save_best: Whether to save the best model checkpoint.
            save_last: Whether to save the last epoch checkpoint.

        Returns:
            Dictionary of training history (lists of per-epoch values).
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # LR scheduling (non-batch schedulers)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(val_metric, 0.0))
                else:
                    self.scheduler.step()

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics.get("accuracy", 0.0))
            history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Logging
            if epoch % log_interval == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{epochs} "
                    f"[{epoch_time:.1f}s] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_acc={train_metrics['accuracy']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics.get('accuracy', 0.0):.4f} "
                    f"val_{val_metric}={val_metrics.get(val_metric, 0.0):.4f} "
                    f"lr={current_lr:.6f}"
                )

            # Save best checkpoint
            current_val_score = val_metrics.get(val_metric, 0.0)
            if save_best:
                is_best = (
                    self.best_val_score is None
                    or (
                        current_val_score > self.best_val_score
                        if val_metric_mode == "max"
                        else current_val_score < self.best_val_score
                    )
                )
                if is_best:
                    self.best_val_score = current_val_score
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_metrics["loss"],
                        os.path.join(self.checkpoint_dir, "best_model.pt"),
                    )
                    logger.info(f"  -> New best {val_metric}: {current_val_score:.4f}")

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(current_val_score):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Save last checkpoint
        if save_last:
            save_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                history["val_loss"][-1],
                os.path.join(self.checkpoint_dir, "last_model.pt"),
            )

        logger.info("Training complete!")
        logger.info(f"Best val_{val_metric}: {self.best_val_score:.4f}")

        return history

    def load_best_model(self):
        """Load the best model checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            info = load_checkpoint(self.model, None, best_path)
            logger.info(f"Loaded best model from epoch {info['epoch']}")
        else:
            logger.warning("No best model checkpoint found")

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Generate predictions for a dataset.

        Args:
            dataloader: Data loader for prediction.

        Returns:
            Dictionary with 'predictions', 'probabilities', 'labels' arrays.
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                labels = batch[1]
            else:
                raise ValueError("Batch must be a tuple of (inputs, labels)")

            outputs = self.model(inputs)

            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().ravel()
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

        return {
            "predictions": np.concatenate(all_preds),
            "probabilities": np.concatenate(all_probs),
            "labels": np.concatenate(all_labels),
        }
