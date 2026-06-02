"""
Visualization utilities for model training and evaluation.

Generates publication-quality figures for training curves, confusion matrices,
ROC curves, and precision-recall curves using matplotlib and seaborn.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Publication style defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot training and validation loss/metric curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        train_metrics: Per-epoch training metric values (optional).
        val_metrics: Per-epoch validation metric values (optional).
        metric_name: Name of the metric for y-axis label.
        save_path: Path to save figure. None to skip saving.
        title: Figure title.

    Returns:
        matplotlib Figure object.
    """
    has_metrics = train_metrics is not None and val_metrics is not None
    n_plots = 2 if has_metrics else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metric plot
    if has_metrics:
        axes[1].plot(epochs, train_metrics, "b-", label="Train", linewidth=2)
        axes[1].plot(epochs, val_metrics, "r-", label="Validation", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f"{title} - {metric_name}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes).
        class_names: Names for each class axis.
        save_path: Path to save figure.
        title: Figure title.
        normalize: Whether to show normalized (percentage) values.
        cmap: Colormap name.

    Returns:
        matplotlib Figure object.
    """
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm.astype(float)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes - 1)))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
    label: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_value: AUC value for annotation.
        save_path: Path to save figure.
        title: Figure title.
        label: Label for the curve legend.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    curve_label = label or "Model"
    if auc_value is not None:
        curve_label += f" (AUC = {auc_value:.4f})"

    ax.plot(fpr, tpr, "b-", linewidth=2, label=curve_label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap_value: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
    label: Optional[str] = None,
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        precision: Precision values.
        recall: Recall values.
        ap_value: Average precision value for annotation.
        save_path: Path to save figure.
        title: Figure title.
        label: Label for the curve legend.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    curve_label = label or "Model"
    if ap_value is not None:
        curve_label += f" (AP = {ap_value:.4f})"

    ax.plot(recall, precision, "b-", linewidth=2, label=curve_label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_multi_roc_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    title: str = "ROC Curves Comparison",
) -> plt.Figure:
    """
    Plot multiple ROC curves for model comparison.

    Args:
        curves: Dict of model_name -> {'fpr': array, 'tpr': array, 'auc': float}.
        save_path: Path to save figure.
        title: Figure title.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    for (name, data), color in zip(curves.items(), colors):
        auc_val = data.get("auc")
        label = f"{name} (AUC={auc_val:.4f})" if auc_val is not None else name
        ax.plot(data["fpr"], data["tpr"], color=color, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Plot grouped bar chart comparing models across metrics.

    Args:
        results: Dict of model_name -> {metric_name: value}.
        metrics: List of metric names to plot.
        save_path: Path to save figure.
        title: Figure title.

    Returns:
        matplotlib Figure object.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [results[name].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_comparison_bar(
    results: Dict[str, float],
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
    color: str = "#2E86AB",
) -> plt.Figure:
    """
    Plot simple bar chart comparing models on a single metric.

    Args:
        results: Dict of model_name -> metric_value.
        metric_name: Name of the metric for y-axis label.
        save_path: Path to save figure.
        title: Figure title.
        color: Bar color.

    Returns:
        matplotlib Figure object.
    """
    model_names = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(model_names, values, color=color, alpha=0.85, edgecolor='white')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, Any]],
    x_key: str = "num_scales",
    metric_key: str = "accuracy",
    save_path: Optional[str] = None,
    title: str = "Ablation Study",
    xlabel: str = "Number of Scales",
    color: str = "#A23B72",
) -> plt.Figure:
    """
    Plot ablation study results as a line chart.

    Args:
        ablation_results: Dict of experiment_name -> {x_key: value, metric_key: value}.
        x_key: Key for x-axis values.
        metric_key: Key for metric values in test_metrics.
        save_path: Path to save figure.
        title: Figure title.
        xlabel: X-axis label.
        color: Line color.

    Returns:
        matplotlib Figure object.
    """
    x_values = []
    y_values = []

    for exp_name, exp_data in ablation_results.items():
        if x_key in exp_data:
            x_values.append(exp_data[x_key])
            if 'test_metrics' in exp_data:
                y_values.append(exp_data['test_metrics'].get(metric_key, 0.0))
            else:
                y_values.append(exp_data.get(metric_key, 0.0))

    # Sort by x values
    sorted_pairs = sorted(zip(x_values, y_values))
    x_values = [p[0] for p in sorted_pairs]
    y_values = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, y_values, 'o-', linewidth=2, markersize=10, color=color)

    # Add value labels
    for x, y in zip(x_values, y_values):
        ax.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_values)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig
