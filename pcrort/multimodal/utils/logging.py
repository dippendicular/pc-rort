"""Structured logging with file and console output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ExperimentLogger:
    """
    Structured experiment logger with file and console output.

    Supports:
    - Console logging with colored level indicators
    - File logging with timestamps
    - JSON metadata logging for experiment tracking
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: Union[str, Path],
        log_level: str = "INFO",
        console: bool = True,
        file: bool = True,
    ):
        """
        Initialize the experiment logger.

        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory to save log files.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            console: Whether to log to console.
            file: Whether to log to file.
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.handlers = []  # Clear existing handlers

        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file

        # Metrics history
        self.metrics_history: list[Dict[str, Any]] = []

    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log a dict of metrics.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Optional step/epoch number.
        """
        entry = {"timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step
        entry.update(metrics)
        self.metrics_history.append(entry)

        # Log to console/file
        parts = []
        if step is not None:
            parts.append(f"Step {step}")
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        self.info(" | ".join(parts))

    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log all hyperparameters from config."""
        self.info("=" * 60)
        self.info("HYPERPARAMETERS")
        self.info("=" * 60)

        def _log_dict(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    _log_dict(value, full_key)
                else:
                    self.info(f"  {full_key}: {value}")

        _log_dict(config)
        self.info("=" * 60)

    def save_metrics_history(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save metrics history to JSON.

        Args:
            path: Optional output path. Defaults to log_dir/metrics_history.json.

        Returns:
            Path to saved file.
        """
        if path is None:
            path = self.log_dir / f"{self.experiment_name}_metrics_history.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        self.info(f"Metrics history saved to {path}")
        return path

    def log_section(self, title: str) -> None:
        """Log a section header for visual separation."""
        self.info("")
        self.info("=" * 60)
        self.info(f" {title}")
        self.info("=" * 60)
