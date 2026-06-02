"""
Evaluation metrics for computational biology tasks.

Includes standard classification metrics and bioinformatics-specific metrics
such as Matthew's Correlation Coefficient (MCC), sensitivity, specificity,
and area under precision-recall curve (AUPRC).
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
    cohen_kappa_score,
    log_loss,
)


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    average: str = "binary",
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels (hard predictions).
        y_prob: Predicted probabilities. Required for AUC/AP metrics.
            - Binary: shape (N,) or (N, 1) for positive class probability.
            - Multi-class: shape (N, C) for class probabilities.
        average: Averaging strategy for multi-class ('binary', 'macro', 'micro', 'weighted').
        num_classes: Number of classes. Inferred from y_true if None.

    Returns:
        Dictionary of metric name -> value.
    """
    y_true = to_numpy(y_true).ravel()
    y_pred = to_numpy(y_pred).ravel()

    if num_classes is None:
        num_classes = len(np.unique(y_true))

    is_binary = num_classes == 2
    avg = "binary" if is_binary else average

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    # Confusion matrix derived metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm

    if is_binary:
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # same as recall
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # negative predictive value
        metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # positive predictive value

    # Probability-based metrics
    if y_prob is not None:
        y_prob_arr = to_numpy(y_prob)

        # Replace NaN values with 0.5 (neutral probability)
        if np.any(np.isnan(y_prob_arr)):
            y_prob_arr = np.nan_to_num(y_prob_arr, nan=0.5)

        if is_binary:
            # Handle shape (N, 1) -> (N,)
            if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 1:
                y_prob_arr = y_prob_arr.ravel()
            # Handle shape (N, 2) -> use positive class
            elif y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
                y_prob_arr = y_prob_arr[:, 1]

            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob_arr)
                metrics["ap"] = average_precision_score(y_true, y_prob_arr)
                metrics["log_loss"] = log_loss(y_true, y_prob_arr)

                # ROC curve points
                fpr, tpr, thresholds = roc_curve(y_true, y_prob_arr)
                metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

                # Precision-recall curve points
                pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_prob_arr)
                metrics["pr_curve"] = {
                    "precision": pr_precision,
                    "recall": pr_recall,
                    "thresholds": pr_thresholds,
                }
            except ValueError:
                metrics["auc"] = 0.5
                metrics["ap"] = 0.0
                metrics["log_loss"] = 1.0
        else:
            # Multi-class AUC
            try:
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_prob_arr, multi_class="ovr", average="macro"
                )
                metrics["auc_weighted"] = roc_auc_score(
                    y_true, y_prob_arr, multi_class="ovr", average="weighted"
                )
            except ValueError:
                # May fail if only one class present in y_true
                metrics["auc_macro"] = 0.0
                metrics["auc_weighted"] = 0.0

            metrics["log_loss"] = log_loss(y_true, y_prob_arr)

    return metrics


def compute_metrics_for_bioinformatics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute metrics commonly used in bioinformatics papers.

    Includes: Accuracy, Sensitivity (Recall/TPR), Specificity (TNR),
    Precision (PPV), F1, MCC, AUC, AUPRC.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dictionary of metric name -> value.
    """
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, average="binary")

    # Rename for bioinformatics convention
    bio_metrics = {
        "Accuracy": metrics["accuracy"],
        "Sensitivity": metrics.get("sensitivity", metrics["recall"]),
        "Specificity": metrics.get("specificity", 0.0),
        "Precision": metrics["precision"],
        "F1-Score": metrics["f1"],
        "MCC": metrics["mcc"],
    }

    if "auc" in metrics:
        bio_metrics["AUC"] = metrics["auc"]
        bio_metrics["AUPRC"] = metrics["ap"]

    return bio_metrics


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """
    Format metrics as a readable string table.

    Args:
        metrics: Dictionary of metric name -> value.
        title: Table title.

    Returns:
        Formatted string.
    """
    lines = [f"\n{'='*40}", f"  {title}", f"{'='*40}"]
    for key, val in metrics.items():
        if key in ("confusion_matrix", "roc_curve", "pr_curve"):
            continue
        if isinstance(val, float):
            lines.append(f"  {key:<20s}: {val:.4f}")
        else:
            lines.append(f"  {key:<20s}: {val}")
    lines.append(f"{'='*40}")
    return "\n".join(lines)


class MetricsTracker:
    """
    Track metrics across training epochs.

    Stores per-epoch metrics and provides access to the best values.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float], epoch: int):
        """Record metrics for an epoch."""
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(val)

    def get_best(self, metric: str, mode: str = "max") -> Tuple[float, int]:
        """
        Get best value and epoch for a metric.

        Args:
            metric: Metric name.
            mode: 'max' or 'min'.

        Returns:
            Tuple of (best_value, best_epoch).
        """
        values = self.history.get(metric, [])
        if not values:
            return 0.0, 0
        if mode == "max":
            best_epoch = int(np.argmax(values))
        else:
            best_epoch = int(np.argmin(values))
        return values[best_epoch], best_epoch

    def get_history(self, metric: str) -> List[float]:
        """Get history for a metric."""
        return self.history.get(metric, [])

    def get_latest(self, metric: str) -> float:
        """Get latest value for a metric."""
        values = self.history.get(metric, [])
        return values[-1] if values else 0.0

    def summary(self) -> str:
        """Return a summary string of all tracked metrics."""
        lines = []
        for key, values in self.history.items():
            if values:
                lines.append(f"{key}: last={values[-1]:.4f}, best={max(values):.4f}")
        return "\n".join(lines)
