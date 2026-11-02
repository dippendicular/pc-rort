"""Explainability and interpretability utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def compute_modality_importance(
    fusion_weights: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute average modality importance from fusion weights.

    Args:
        fusion_weights: Dict of modality_name → weight array (N,).

    Returns:
        Dict of modality_name → average weight.
    """
    importance = {}
    for name, weights in fusion_weights.items():
        importance[name] = float(np.mean(weights))

    # Normalize to sum to 1
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


def compute_class_specific_modality_importance(
    fusion_weights: Dict[str, np.ndarray],
    labels: np.ndarray,
    num_classes: int = 2,
) -> Dict[int, Dict[str, float]]:
    """
    Compute modality importance per class.

    Args:
        fusion_weights: Modality name → weight array (N,).
        labels: True labels (N,).
        num_classes: Number of classes.

    Returns:
        Dict of class_label → modality_importance_dict.
    """
    results = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        cls_weights = {name: weights[mask] for name, weights in fusion_weights.items()}
        results[cls] = compute_modality_importance(cls_weights)
    return results


def generate_importance_report(
    modality_importance: Dict[str, float],
    class_importance: Optional[Dict[int, Dict[str, float]]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    experiment_name: str = "multimodal",
) -> str:
    """
    Generate a text report of modality importance.

    Returns:
        Report string.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("MODALITY IMPORTANCE ANALYSIS")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Overall modality contributions:")
    for name, weight in sorted(modality_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(weight * 40)
        lines.append(f"  {name:15s}: {weight:.4f} {bar}")

    if class_importance:
        lines.append("")
        lines.append("Per-class modality contributions:")
        for cls, importance in sorted(class_importance.items()):
            lines.append(f"  Class {cls}:")
            for name, weight in sorted(importance.items(), key=lambda x: -x[1]):
                lines.append(f"    {name:15s}: {weight:.4f}")

    report = "\n".join(lines)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{experiment_name}_importance_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report
