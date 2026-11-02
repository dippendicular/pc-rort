"""Checkpoint saving and loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages model checkpoints: save, load, and track best model.

    Features:
    - Save/load full training state (model, optimizer, scheduler, epoch, metrics)
    - Track and save the best model based on a monitored metric
    - Save last checkpoint for resuming interrupted training
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        experiment_name: str,
        best_metric: str = "val_auroc",
        best_mode: str = "max",
    ):
        """
        Initialize CheckpointManager.

        Args:
            save_dir: Directory to save checkpoints.
            experiment_name: Name prefix for checkpoint files.
            best_metric: Metric to track for best model selection.
            best_mode: 'max' or 'min' — whether higher or lower is better.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.best_metric = best_metric
        self.best_mode = best_mode

        if best_mode == "max":
            self.best_value = float("-inf")
        else:
            self.best_value = float("inf")

        self.best_epoch = -1

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: str = "last",
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            model: The model to save.
            optimizer: Optimizer state.
            epoch: Current epoch number.
            metrics: Dict of current metrics.
            scheduler: Optional LR scheduler.
            scaler: Optional GradScaler for mixed precision.
            config: Optional config dict to save alongside.
            is_best: Whether this is the best model so far.
            tag: Filename tag ('last', 'best', or epoch number).

        Returns:
            Path to saved checkpoint file.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        if config is not None:
            checkpoint["config"] = config

        # Save with tag
        path = self.save_dir / f"{self.experiment_name}_{tag}.pt"
        torch.save(checkpoint, path)

        # If best, also save a copy as 'best'
        if is_best:
            best_path = self.save_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)

        return path

    def check_is_best(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        Check if current metrics are the best seen so far.

        Args:
            metrics: Current epoch metrics.
            epoch: Current epoch number.

        Returns:
            True if this is a new best.
        """
        current = metrics.get(self.best_metric, None)
        if current is None:
            return False

        if self.best_mode == "max":
            is_best = current > self.best_value
        else:
            is_best = current < self.best_value

        if is_best:
            self.best_value = current
            self.best_epoch = epoch

        return is_best

    def load_checkpoint(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model/optimizer/scheduler state.

        Args:
            path: Path to checkpoint file.
            model: Model to load state into.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            scaler: Optional GradScaler to restore.
            device: Device to map tensors to.

        Returns:
            Dict with 'epoch', 'metrics', and optionally 'config'.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = device if device else "cpu"
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config", None),
        }

    def get_best_model_path(self) -> Path:
        """Return path to the best saved model."""
        return self.save_dir / f"{self.experiment_name}_best.pt"

    def get_last_model_path(self) -> Path:
        """Return path to the last saved model."""
        return self.save_dir / f"{self.experiment_name}_last.pt"

    def save_training_summary(self, metrics: Dict[str, Any]) -> Path:
        """
        Save a JSON summary of the best training results.

        Args:
            metrics: Final metrics dict.

        Returns:
            Path to saved summary.
        """
        summary = {
            "experiment_name": self.experiment_name,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_value": self.best_value,
            "final_metrics": metrics,
        }
        path = self.save_dir / f"{self.experiment_name}_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return path
