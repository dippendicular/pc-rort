"""
Train a radiology-only model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pcrort.multimodal.datasets.radiology_dataset import RadiologyDataset
from pcrort.multimodal.preprocessing.patient_matching import (
    create_patient_splits,
    load_existing_splits,
    load_metadata,
    print_dataset_statistics,
)
from pcrort.multimodal.preprocessing.radiology import get_radiology_transforms
from pcrort.multimodal.evaluation.metrics import save_metrics, save_predictions
from pcrort.multimodal.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)
from pcrort.multimodal.models.radiology_encoder import RadiologyEncoder
from pcrort.multimodal.training.trainer import Trainer
from pcrort.multimodal.utils.config import load_config, print_config, resolve_paths
from pcrort.multimodal.utils.logging import ExperimentLogger
from pcrort.multimodal.utils.seed import get_device, print_device_info, set_seed


def extract_radiology_batch(batch: Dict[str, Any], device: torch.device):
    """Extract input and label for radiology model."""
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    return images, labels


def run_experiment(config_path: str, overrides: Dict[str, Any] = None):
    # 1. Load configuration
    config = load_config(["configs/multimodal/base.yaml", config_path], overrides=overrides)
    # resolve relative paths relative to project root
    config = resolve_paths(config, Path.cwd())

    # 2. Setup logging & seeds
    exp_name = config.get("experiment", {}).get("name", "radiology_experiment")
    logger = ExperimentLogger(exp_name, config.get("logging", {}).get("log_dir", "outputs/logs"))
    logger.log_hyperparameters(config)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    device = get_device(config.get("device", {}).get("use_cuda", True))
    print_device_info(device)
    logger.info(f"Using device: {device}")

    # 3. Load & split data
    dataset_cfg = config.get("dataset", {})
    metadata_csv = dataset_cfg.get("metadata_csv")
    if not metadata_csv or not Path(metadata_csv).exists():
        logger.error(f"Metadata CSV not found: {metadata_csv}")
        return

    df = load_metadata(
        metadata_csv,
        patient_id_col=dataset_cfg.get("patient_id_col", "patient_id"),
        label_col=dataset_cfg.get("label_col", "label"),
    )

    stats = print_dataset_statistics(
        df,
        patient_id_col=dataset_cfg.get("patient_id_col", "patient_id"),
        label_col=dataset_cfg.get("label_col", "label"),
        radiology_path_col=dataset_cfg.get("radiology_path_col", "radiology_path"),
    )

    # Filter only patients with radiology
    rad_col = dataset_cfg.get("radiology_path_col", "radiology_path")
    df = df[df[rad_col].notna() & (df[rad_col] != "")].reset_index(drop=True)

    # Smoke test mode
    if config.get("experiment", {}).get("smoke_test", False):
        smoke_samples = config.get("experiment", {}).get("smoke_test_samples", 32)
        logger.info(f"Running SMOKE TEST with {smoke_samples} samples")
        df = df.sample(min(smoke_samples, len(df)), random_state=seed).reset_index(drop=True)
        config["training"]["epochs"] = 1

    splitting_cfg = config.get("splitting", {})
    if splitting_cfg.get("existing_splits"):
        train_df, val_df, test_df = load_existing_splits(splitting_cfg["existing_splits"])
        # ensure only using patients with radiology in existing splits
        train_df = train_df[train_df[rad_col].notna()].reset_index(drop=True)
        val_df = val_df[val_df[rad_col].notna()].reset_index(drop=True)
        test_df = test_df[test_df[rad_col].notna()].reset_index(drop=True)
    else:
        train_df, val_df, test_df = create_patient_splits(
            df,
            patient_id_col=dataset_cfg.get("patient_id_col", "patient_id"),
            label_col=dataset_cfg.get("label_col", "label"),
            train_ratio=splitting_cfg.get("train_ratio", 0.7),
            val_ratio=splitting_cfg.get("val_ratio", 0.15),
            test_ratio=splitting_cfg.get("test_ratio", 0.15),
            stratify=splitting_cfg.get("stratify", True),
            seed=seed,
            splits_dir=splitting_cfg.get("splits_dir", "outputs/splits"),
        )

    # 4. Build dataloaders
    train_transform = get_radiology_transforms(config.get("radiology", {}), is_training=True)
    eval_transform = get_radiology_transforms(config.get("radiology", {}), is_training=False)

    num_classes = dataset_cfg.get("num_classes", 2)

    train_ds = RadiologyDataset(
        train_df,
        dataset_cfg.get("patient_id_col", "patient_id"),
        dataset_cfg.get("label_col", "label"),
        rad_col,
        transform=train_transform,
        num_classes=num_classes,
    )
    val_ds = RadiologyDataset(
        val_df,
        dataset_cfg.get("patient_id_col", "patient_id"),
        dataset_cfg.get("label_col", "label"),
        rad_col,
        transform=eval_transform,
        num_classes=num_classes,
    )
    test_ds = RadiologyDataset(
        test_df,
        dataset_cfg.get("patient_id_col", "patient_id"),
        dataset_cfg.get("label_col", "label"),
        rad_col,
        transform=eval_transform,
        num_classes=num_classes,
    )

    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("device", {}).get("num_workers", 4)
    pin_memory = config.get("device", {}).get("pin_memory", True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # 5. Build Model
    rad_cfg = config.get("radiology", {})
    model = RadiologyEncoder(
        backbone=rad_cfg.get("backbone", "densenet121"),
        pretrained=rad_cfg.get("pretrained", True),
        embedding_dim=rad_cfg.get("embedding_dim", 512),
        projection_dim=rad_cfg.get("projection_dim", 512),
        dropout=rad_cfg.get("dropout", 0.3),
        num_classes=num_classes,
        freeze_backbone=rad_cfg.get("freeze_backbone", False),
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        logger=logger,
        extract_fn=extract_radiology_batch,
    )

    results = trainer.train()

    # 7. Evaluate & Plot
    logger.log_section("EVALUATION & PLOTS")
    eval_cfg = config.get("evaluation", {})
    metrics_dir = Path(eval_cfg.get("metrics_dir", "outputs/metrics"))
    preds_dir = Path(eval_cfg.get("predictions_dir", "outputs/predictions"))
    figs_dir = Path(eval_cfg.get("figures_dir", "outputs/figures"))
    formats = eval_cfg.get("figure_format", ["png", "pdf"])
    class_names = dataset_cfg.get("class_names")

    test_metrics = results["test_metrics"]
    test_preds = results["test_predictions"]

    save_metrics(test_metrics, metrics_dir, exp_name)
    save_predictions(test_preds, preds_dir, exp_name)
    logger.save_metrics_history()

    plot_training_curves(results["history"], figs_dir, exp_name, formats)
    plot_confusion_matrix(
        test_preds["true_labels"],
        test_preds["predicted_labels"],
        figs_dir,
        exp_name,
        class_names,
        formats,
    )

    if num_classes == 2:
        plot_roc_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats)
        plot_pr_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats)
        plot_calibration_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats=formats)

    logger.info("Radiology experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal/radiology.yaml", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke test mode")
    args = parser.parse_args()

    overrides = {}
    if args.smoke_test:
        overrides["experiment"] = {"smoke_test": True}

    run_experiment(args.config, overrides)
