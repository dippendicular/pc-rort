"""Training callbacks — early stopping and metrics logging."""

from __future__ import annotations

from typing import Any, Dict, Optional


class EarlyStopping:
    """
    Early stopping to terminate training when a metric stops improving.
    """

    def __init__(
        self,
        patience: int = 15,
        metric: str = "val_auroc",
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        """
        Args:
            patience: Number of epochs to wait after last improvement.
            metric: Metric name to monitor.
            mode: 'max' (higher is better) or 'min' (lower is better).
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_epoch = -1

    def __call__(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            metrics: Current epoch metrics.
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        current = metrics.get(self.metric)
        if current is None:
            return False

        if self.mode == "max":
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class MetricsTracker:
    """
    Tracks and stores metrics across epochs.
    """

    def __init__(self):
        self.history: Dict[str, list] = {}

    def update(self, metrics: Dict[str, float], epoch: int) -> None:
        """Record metrics for an epoch."""
        if "epoch" not in self.history:
            self.history["epoch"] = []
        self.history["epoch"].append(epoch)

        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_best(self, metric: str, mode: str = "max") -> Dict[str, Any]:
        """Get the best value and epoch for a metric."""
        if metric not in self.history:
            return {"value": None, "epoch": None}

        values = self.history[metric]
        if mode == "max":
            idx = max(range(len(values)), key=lambda i: values[i])
        else:
            idx = min(range(len(values)), key=lambda i: values[i])

        return {
            "value": values[idx],
            "epoch": self.history["epoch"][idx],
        }

    def to_dict(self) -> Dict[str, list]:
        """Return full history as dict."""
        return dict(self.history)
