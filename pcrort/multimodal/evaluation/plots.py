"""
Publication-quality plotting utilities.

All figures are saved as PNG and PDF at 300+ DPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Colab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def _save_fig(fig: plt.Figure, path: Path, formats: List[str] = None) -> None:
    """Save figure in multiple formats."""
    if formats is None:
        formats = ["png", "pdf"]
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, list],
    output_dir: Union[str, Path],
    experiment_name: str,
    formats: List[str] = None,
) -> None:
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Training history dict with epoch, train_loss, val_loss, etc.
        output_dir: Directory to save figures.
        experiment_name: Name prefix for files.
        formats: Output formats.
    """
    output_dir = Path(output_dir)
    epochs = history.get("epoch", list(range(len(history.get("train_loss", [])))))

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training & Validation Loss — {experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"{experiment_name}_loss_curves", formats)

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(8, 5))
    if "train_accuracy" in history:
        ax.plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history:
        ax.plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Training & Validation Accuracy — {experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"{experiment_name}_accuracy_curves", formats)

    # AUROC curve over epochs
    fig, ax = plt.subplots(figsize=(8, 5))
    if "val_auroc" in history:
        ax.plot(epochs, history["val_auroc"], label="Val AUROC", linewidth=2, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Validation AUROC — {experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"{experiment_name}_auroc_curve", formats)


def plot_roc_curve(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_dir: Union[str, Path],
    experiment_name: str,
    formats: List[str] = None,
) -> None:
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc

    output_dir = Path(output_dir)
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {experiment_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"{experiment_name}_roc", formats)


def plot_pr_curve(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_dir: Union[str, Path],
    experiment_name: str,
    formats: List[str] = None,
) -> None:
    """Plot Precision-Recall curve."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    output_dir = Path(output_dir)
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    ap = average_precision_score(true_labels, pred_probs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, linewidth=2, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {experiment_name}")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"{experiment_name}_pr", formats)


def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    output_dir: Union[str, Path],
    experiment_name: str,
    class_names: Optional[List[str]] = None,
    formats: List[str] = None,
) -> None:
    """Plot confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    output_dir = Path(output_dir)
    cm = confusion_matrix(true_labels, pred_labels)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {experiment_name}")
    _save_fig(fig, output_dir / f"confusion_matrix_{experiment_name}", formats)


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str,
    output_dir: Union[str, Path],
    filename: str = "model_comparison",
    formats: List[str] = None,
) -> None:
    """
    Bar chart comparing models on a metric.

    Args:
        results: Dict of model_name → metrics_dict.
        metric: Metric to compare.
        output_dir: Output directory.
        filename: Output filename.
        formats: Output formats.
    """
    output_dir = Path(output_dir)
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(models))
    bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, output_dir / filename, formats)


def plot_ablation_study(
    ablation_df: "pd.DataFrame",
    output_dir: Union[str, Path],
    formats: List[str] = None,
) -> None:
    """
    Plot ablation study results — grouped bar chart.

    Args:
        ablation_df: DataFrame with columns: model, modalities, fusion_method, and metric columns.
        output_dir: Output directory.
        formats: Output formats.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    metrics_to_plot = ["accuracy", "auroc", "f1"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        if metric in ablation_df.columns:
            models = ablation_df["model"].tolist()
            values = ablation_df[metric].tolist()
            colors = sns.color_palette("husl", len(models))
            bars = ax.bar(range(len(models)), values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric.upper())
            ax.set_title(metric.upper())
            ax.grid(axis="y", alpha=0.3)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

    fig.suptitle("Ablation Study Results", fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_dir / "ablation_study", formats)


def plot_modality_contributions(
    contributions: Dict[str, float],
    output_dir: Union[str, Path],
    experiment_name: str = "multimodal",
    formats: List[str] = None,
) -> None:
    """Plot average modality contribution pie/bar chart."""
    output_dir = Path(output_dir)
    modalities = list(contributions.keys())
    values = list(contributions.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    colors = sns.color_palette("Set2", len(modalities))
    ax1.bar(modalities, values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Average Contribution Weight")
    ax1.set_title("Modality Contributions (Bar)")
    ax1.grid(axis="y", alpha=0.3)
    for i, (mod, val) in enumerate(zip(modalities, values)):
        ax1.text(i, val + 0.005, f"{val:.3f}", ha="center", va="bottom")

    # Pie chart
    ax2.pie(values, labels=modalities, autopct="%1.1f%%", colors=colors, startangle=90)
    ax2.set_title("Modality Contributions (Proportion)")

    fig.suptitle(f"Modality Contribution Analysis — {experiment_name}", fontsize=14)
    plt.tight_layout()
    _save_fig(fig, output_dir / f"modality_contributions_{experiment_name}", formats)


def plot_calibration_curve(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_dir: Union[str, Path],
    experiment_name: str,
    n_bins: int = 10,
    formats: List[str] = None,
) -> None:
    """Plot calibration curve (reliability diagram)."""
    from sklearn.calibration import calibration_curve

    output_dir = Path(output_dir)
    prob_true, prob_pred = calibration_curve(true_labels, pred_probs, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(prob_pred, prob_true, "o-", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {experiment_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"calibration_{experiment_name}", formats)


def plot_fusion_comparison(
    results: Dict[str, Dict[str, float]],
    output_dir: Union[str, Path],
    formats: List[str] = None,
) -> None:
    """
    Compare fusion strategies with grouped bar chart.

    Args:
        results: Dict of fusion_method → metrics_dict.
    """
    output_dir = Path(output_dir)
    metrics = ["accuracy", "auroc", "f1", "sensitivity", "specificity"]
    methods = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)
    colors = sns.color_palette("Set2", len(methods))

    for i, method in enumerate(methods):
        values = [results[method].get(m, 0) for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Fusion Strategy Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    _save_fig(fig, output_dir / "fusion_comparison", formats)
