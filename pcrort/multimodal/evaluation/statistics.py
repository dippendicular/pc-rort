"""Statistical analysis — confidence intervals and significance tests."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def bootstrap_confidence_interval(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        true_labels: True labels.
        pred_probs: Predicted probabilities.
        metric_fn: Function(true, pred) -> float.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.RandomState(seed)
    n = len(true_labels)
    scores = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        try:
            score = metric_fn(true_labels[indices], pred_probs[indices])
            scores.append(score)
        except (ValueError, IndexError):
            continue

    scores = np.array(scores)
    point_estimate = metric_fn(true_labels, pred_probs)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))

    return point_estimate, lower, upper


def compute_ci_for_auroc(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute AUROC with bootstrap 95% CI."""
    from sklearn.metrics import roc_auc_score

    est, lower, upper = bootstrap_confidence_interval(
        true_labels, pred_probs, roc_auc_score, n_bootstrap, confidence, seed
    )
    return {"auroc": est, "auroc_lower": lower, "auroc_upper": upper}


def mcnemar_test(
    true_labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar's test to compare two classifiers.

    Args:
        true_labels: Ground-truth labels.
        preds_a: Predictions from model A.
        preds_b: Predictions from model B.

    Returns:
        Dict with test statistic and p-value.
    """
    from scipy.stats import binom_test

    correct_a = (preds_a == true_labels)
    correct_b = (preds_b == true_labels)

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # McNemar statistic
    if (b + c) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    statistic = float((abs(b - c) - 1) ** 2 / (b + c))

    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(statistic, df=1))

    return {"statistic": statistic, "p_value": p_value, "b": b, "c": c}
