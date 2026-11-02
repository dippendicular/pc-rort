"""Evaluation metrics computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def compute_all_metrics(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    pred_labels: np.ndarray,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        true_labels: Ground-truth labels.
        pred_probs: Predicted probabilities.
        pred_labels: Predicted class labels.
        num_classes: Number of classes.

    Returns:
        Dict of metric name → value.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(true_labels, pred_labels))

    if num_classes == 2:
        metrics["precision"] = float(precision_score(true_labels, pred_labels, zero_division=0))
        metrics["recall"] = float(recall_score(true_labels, pred_labels, zero_division=0))
        metrics["f1"] = float(f1_score(true_labels, pred_labels, zero_division=0))

        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        metrics["tp"] = int(tp)
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        try:
            metrics["auroc"] = float(roc_auc_score(true_labels, pred_probs))
        except ValueError:
            metrics["auroc"] = 0.0
        try:
            metrics["auprc"] = float(average_precision_score(true_labels, pred_probs))
        except ValueError:
            metrics["auprc"] = 0.0
    else:
        metrics["precision"] = float(precision_score(true_labels, pred_labels, average="macro", zero_division=0))
        metrics["recall"] = float(recall_score(true_labels, pred_labels, average="macro", zero_division=0))
        metrics["f1"] = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
        metrics["sensitivity"] = metrics["recall"]
        metrics["specificity"] = 0.0

        try:
            metrics["auroc"] = float(
                roc_auc_score(true_labels, pred_probs, multi_class="ovr", average="macro")
            ) if pred_probs.ndim > 1 else 0.0
        except ValueError:
            metrics["auroc"] = 0.0

    return metrics


def save_metrics(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    experiment_name: str,
) -> None:
    """Save metrics as JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{experiment_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # CSV (flatten one level)
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"{k}_{k2}"] = v2
        else:
            flat[k] = v
    csv_path = output_dir / f"{experiment_name}_metrics.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)


def save_predictions(
    predictions: Dict[str, Any],
    output_dir: Union[str, Path],
    experiment_name: str,
    fusion_weights: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """
    Save predictions as CSV.

    Args:
        predictions: Dict with patient_ids, true_labels, predicted_labels, prediction_probabilities.
        output_dir: Output directory.
        experiment_name: Experiment name prefix.
        fusion_weights: Optional modality weights per patient.

    Returns:
        Path to saved CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_data = {
        "patient_id": predictions["patient_ids"],
        "true_label": predictions["true_labels"],
        "predicted_label": predictions["predicted_labels"],
    }

    probs = predictions["prediction_probabilities"]
    if probs.ndim == 1:
        df_data["prediction_probability"] = probs
    else:
        for i in range(probs.shape[1]):
            df_data[f"prob_class_{i}"] = probs[:, i]

    # Add fusion weights
    if fusion_weights:
        for mod_name, weights in fusion_weights.items():
            df_data[f"{mod_name}_weight"] = weights

    df = pd.DataFrame(df_data)
    path = output_dir / f"{experiment_name}_predictions.csv"
    df.to_csv(path, index=False)
    return path
