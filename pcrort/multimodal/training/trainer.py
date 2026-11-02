"""
Generic Trainer — handles train/val/test loops for any model type.

Supports:
- Mixed precision training (AMP)
- Gradient clipping
- LR scheduling (cosine, step, plateau)
- Early stopping
- Checkpointing (best + last)
- Comprehensive metrics tracking
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.checkpoint import CheckpointManager
from ..utils.logging import ExperimentLogger
from .callbacks import EarlyStopping, MetricsTracker
from .losses import build_loss


def _compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        all_labels: Ground-truth labels.
        all_probs: Predicted probabilities.
        all_preds: Predicted class labels.
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

    metrics["accuracy"] = float(accuracy_score(all_labels, all_preds))

    if num_classes == 2:
        # Binary metrics
        metrics["precision"] = float(
            precision_score(all_labels, all_preds, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(all_labels, all_preds, zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(all_labels, all_preds, zero_division=0)
        )

        # TP, TN, FP, FN
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        metrics["tp"] = int(tp)
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)

        # Sensitivity and Specificity
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # AUROC and AUPRC
        try:
            metrics["auroc"] = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            metrics["auroc"] = 0.0
        try:
            metrics["auprc"] = float(average_precision_score(all_labels, all_probs))
        except ValueError:
            metrics["auprc"] = 0.0

    else:
        # Multiclass metrics (macro average)
        metrics["precision"] = float(
            precision_score(all_labels, all_preds, average="macro", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(all_labels, all_preds, average="macro", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
        )
        metrics["sensitivity"] = metrics["recall"]
        metrics["specificity"] = 0.0  # Needs per-class computation

        try:
            if all_probs.ndim == 1:
                metrics["auroc"] = 0.0
            else:
                metrics["auroc"] = float(
                    roc_auc_score(
                        all_labels, all_probs, multi_class="ovr", average="macro"
                    )
                )
        except ValueError:
            metrics["auroc"] = 0.0

        try:
            metrics["auprc"] = float(
                average_precision_score(
                    all_labels, all_probs, average="macro"
                )
            ) if all_probs.ndim > 1 else 0.0
        except (ValueError, IndexError):
            metrics["auprc"] = 0.0

    return metrics


class Trainer:
    """
    Generic training loop for unimodal and multimodal models.

    Works with any model that returns a dict containing 'logits'
    and any model input format (the batch dict is passed directly).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        logger: Optional[ExperimentLogger] = None,
        extract_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model: The model to train.
            config: Full merged config dict.
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            test_loader: Optional test dataloader.
            device: Compute device.
            logger: Experiment logger.
            extract_fn: Function to extract (inputs, labels) from a batch.
                        Signature: extract_fn(batch) -> (model_input, labels)
                        If None, batch is passed directly to model.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger
        self.extract_fn = extract_fn

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        # Training config
        train_cfg = config.get("training", {})
        dataset_cfg = config.get("dataset", {})
        self.num_classes = dataset_cfg.get("num_classes", 2)
        self.epochs = train_cfg.get("epochs", 100)
        self.gradient_clip_norm = train_cfg.get("gradient_clip_norm", 1.0)
        self.accumulation_steps = train_cfg.get("accumulation_steps", 1)

        # Mixed precision
        device_cfg = config.get("device", {})
        self.use_amp = (
            device_cfg.get("mixed_precision", True)
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler(enabled=self.use_amp)

        # Loss
        class_weights = dataset_cfg.get("class_weights")
        self.criterion = build_loss(
            num_classes=self.num_classes,
            class_weights=class_weights,
            device=self.device,
        )

        # Optimizer
        self.optimizer = self._build_optimizer(train_cfg)

        # Scheduler
        self.scheduler = self._build_scheduler(train_cfg)

        # Early stopping
        es_cfg = config.get("early_stopping", {})
        self.early_stopping = None
        if es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 15),
                metric=es_cfg.get("metric", "val_auroc"),
                mode=es_cfg.get("mode", "max"),
            )

        # Checkpoint manager
        ckpt_cfg = config.get("checkpoint", {})
        experiment_name = config.get("experiment", {}).get("name", "experiment")
        self.checkpoint_manager = CheckpointManager(
            save_dir=ckpt_cfg.get("save_dir", "outputs/models"),
            experiment_name=experiment_name,
            best_metric=ckpt_cfg.get("best_metric", "val_auroc"),
            best_mode=ckpt_cfg.get("best_mode", "max"),
        )

        # Resume from checkpoint
        self.start_epoch = 0
        resume_path = ckpt_cfg.get("resume_from")
        if resume_path and Path(resume_path).exists():
            info = self.checkpoint_manager.load_checkpoint(
                resume_path, self.model, self.optimizer,
                self.scheduler, self.scaler, self.device,
            )
            self.start_epoch = info["epoch"] + 1
            if self.logger:
                self.logger.info(f"Resumed from epoch {info['epoch']}")

        # Metrics tracker
        self.metrics_tracker = MetricsTracker()

    def _build_optimizer(self, train_cfg: Dict) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        opt_name = train_cfg.get("optimizer", "adam").lower()
        lr = train_cfg.get("learning_rate", 1e-4)
        wd = train_cfg.get("weight_decay", 1e-4)

        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self, train_cfg: Dict):
        """Build LR scheduler from config."""
        sched_name = train_cfg.get("scheduler", "cosine").lower()
        params = train_cfg.get("scheduler_params", {})

        if sched_name == "cosine":
            T_max = params.get("T_max") or train_cfg.get("epochs", 100)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 30),
                gamma=params.get("gamma", 0.1),
            )
        elif sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=params.get("patience", 10),
                factor=params.get("gamma", 0.1),
            )
        elif sched_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    def _forward_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass and return (logits, labels).

        Handles both unimodal and multimodal model interfaces.
        """
        if self.extract_fn:
            model_input, labels = self.extract_fn(batch, self.device)
            output = self.model(model_input)
        else:
            # Move all tensors to device
            batch_gpu = {}
            labels = None
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(self.device)
                    if k == "label":
                        labels = v.to(self.device)
                else:
                    batch_gpu[k] = v

            if labels is None:
                labels = batch_gpu.get("label", batch_gpu.get("labels"))

            output = self.model(batch_gpu)

        # Extract logits
        if isinstance(output, dict):
            logits = output["logits"]
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            logits = output[0]

        return logits, labels

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            with autocast(enabled=self.use_amp):
                logits, labels = self._forward_batch(batch)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            n_batches += 1

            # Collect predictions
            with torch.no_grad():
                if self.num_classes == 2:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = probs.argmax(axis=-1)

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs)
                all_preds.append(preds)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)

        metrics = _compute_metrics(all_labels, all_probs, all_preds, self.num_classes)
        metrics["loss"] = total_loss / max(n_batches, 1)

        # Prefix with train_
        return {f"train_{k}": v for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, prefix: str = "val"
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Evaluate on a data loader.

        Returns:
            Tuple of (metrics_dict, predictions_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []
        all_patient_ids = []
        all_fusion_weights: Dict[str, list] = {}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Eval ({prefix})", leave=False)

        for batch in pbar:
            with autocast(enabled=self.use_amp):
                logits, labels = self._forward_batch(batch)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=-1)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)

            # Collect patient IDs
            pids = batch.get("patient_ids", batch.get("patient_id", []))
            if isinstance(pids, list):
                all_patient_ids.extend(pids)
            else:
                all_patient_ids.append(pids)

            # Collect fusion weights if available
            if isinstance(batch, dict):
                # Get model output for weights (re-run is avoided; we use batch-level)
                pass  # Fusion weights need model output, handle separately

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)

        metrics = _compute_metrics(all_labels, all_probs, all_preds, self.num_classes)
        metrics["loss"] = total_loss / max(n_batches, 1)

        prefixed_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

        predictions = {
            "patient_ids": all_patient_ids,
            "true_labels": all_labels,
            "predicted_labels": all_preds,
            "prediction_probabilities": all_probs,
        }

        return prefixed_metrics, predictions

    def train(self) -> Dict[str, Any]:
        """
        Full training loop with validation, early stopping, and checkpointing.

        Returns:
            Dict with training history, best metrics, and predictions.
        """
        if self.logger:
            self.logger.log_section("TRAINING START")
            self.logger.info(f"Epochs: {self.epochs}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Mixed precision: {self.use_amp}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_metrics = {}
        best_predictions = {}

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics, val_predictions = self.evaluate(self.val_loader, prefix="val")

            # Merge metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            epoch_metrics["epoch_time"] = time.time() - epoch_start

            # Track
            self.metrics_tracker.update(epoch_metrics, epoch)

            # Log
            if self.logger:
                self.logger.log_metrics(
                    {
                        "train_loss": epoch_metrics["train_loss"],
                        "val_loss": epoch_metrics["val_loss"],
                        "val_auroc": epoch_metrics.get("val_auroc", 0),
                        "val_f1": epoch_metrics.get("val_f1", 0),
                        "lr": epoch_metrics["lr"],
                    },
                    step=epoch,
                )

            # Checkpoint
            is_best = self.checkpoint_manager.check_is_best(epoch_metrics, epoch)
            if is_best:
                best_val_metrics = val_metrics
                best_predictions = val_predictions
                if self.logger:
                    self.logger.info(
                        f"  ★ New best model at epoch {epoch} "
                        f"({self.checkpoint_manager.best_metric}: "
                        f"{self.checkpoint_manager.best_value:.4f})"
                    )

            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                scheduler=self.scheduler,
                scaler=self.scaler if self.use_amp else None,
                config=self.config,
                is_best=is_best,
                tag="last",
            )

            # LR scheduler step
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(epoch_metrics.get("val_auroc", 0))
                else:
                    self.scheduler.step()

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(epoch_metrics, epoch):
                    if self.logger:
                        self.logger.info(
                            f"Early stopping at epoch {epoch}. "
                            f"Best epoch: {self.early_stopping.best_epoch}"
                        )
                    break

        # Final test evaluation
        test_metrics = {}
        test_predictions = {}
        if self.test_loader is not None:
            if self.logger:
                self.logger.log_section("TEST EVALUATION")

            # Load best model for testing
            best_path = self.checkpoint_manager.get_best_model_path()
            if best_path.exists():
                self.checkpoint_manager.load_checkpoint(
                    best_path, self.model, device=self.device
                )
                if self.logger:
                    self.logger.info(f"Loaded best model from {best_path}")

            test_metrics, test_predictions = self.evaluate(
                self.test_loader, prefix="test"
            )

            if self.logger:
                self.logger.log_metrics(test_metrics)

        # Save training summary
        self.checkpoint_manager.save_training_summary(
            {**best_val_metrics, **test_metrics}
        )

        return {
            "history": self.metrics_tracker.to_dict(),
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "test_predictions": test_predictions,
            "best_epoch": self.checkpoint_manager.best_epoch,
        }
