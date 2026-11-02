"""
Train a pathology-only model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pcrort.multimodal.datasets.pathology_dataset import PathologyDataset, pathology_collate_fn
from pcrort.multimodal.preprocessing.patient_matching import (
    create_patient_splits,
    load_existing_splits,
    load_metadata,
    print_dataset_statistics,
)
from pcrort.multimodal.preprocessing.pathology import get_pathology_transforms
from pcrort.multimodal.evaluation.metrics import save_metrics, save_predictions
from pcrort.multimodal.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)
from pcrort.multimodal.models.pathology_encoder import PathologyEncoder
from pcrort.multimodal.training.trainer import Trainer
from pcrort.multimodal.utils.config import load_config, print_config, resolve_paths
from pcrort.multimodal.utils.logging import ExperimentLogger
from pcrort.multimodal.utils.seed import get_device, print_device_info, set_seed


def extract_pathology_batch(batch: Dict[str, Any], device: torch.device):
    """Extract input and label for pathology model."""
    # PathologyEncoder expects (patches, mask)
    patches = batch["patches"].to(device)
    mask = batch["mask"].to(device)
    labels = batch["label"].to(device)
    # We pass a dict to the model so it can accept multiple args via unpack if needed,
    # but our Trainer expects the first return value to be the model input.
    # To keep it simple, we return a dict for the input.
    model_input = {"patches": patches, "mask": mask}
    return model_input, labels


class DictInputWrapper(torch.nn.Module):
    """Wrapper to pass dict input as kwargs to model."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_dict):
        return self.model(**x_dict)


def run_experiment(config_path: str, overrides: Dict[str, Any] = None):
    config = load_config(["configs/multimodal/base.yaml", config_path], overrides=overrides)
    config = resolve_paths(config, Path.cwd())

    exp_name = config.get("experiment", {}).get("name", "pathology_experiment")
    logger = ExperimentLogger(exp_name, config.get("logging", {}).get("log_dir", "outputs/logs"))
    logger.log_hyperparameters(config)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("device", {}).get("use_cuda", True))

    dataset_cfg = config.get("dataset", {})
    path_cfg = config.get("pathology", {})
    metadata_csv = dataset_cfg.get("metadata_csv")
    
    if not metadata_csv or not Path(metadata_csv).exists():
        logger.error(f"Metadata CSV not found: {metadata_csv}")
        return

    df = load_metadata(
        metadata_csv,
        patient_id_col=dataset_cfg.get("patient_id_col", "patient_id"),
        label_col=dataset_cfg.get("label_col", "label"),
    )

    # In pathology, availability is determined by patch dir/manifest. We don't filter immediately unless we have single path
    if path_cfg.get("patches_manifest_col") and path_cfg.get("patches_manifest_col") in df.columns:
        manifest_col = path_cfg.get("patches_manifest_col")
        df = df[df[manifest_col].notna() & (df[manifest_col] != "")].reset_index(drop=True)
    elif dataset_cfg.get("pathology_path_col") and dataset_cfg.get("pathology_path_col") in df.columns:
        path_col = dataset_cfg.get("pathology_path_col")
        df = df[df[path_col].notna() & (df[path_col] != "")].reset_index(drop=True)
    
    # Smoke test mode
    if config.get("experiment", {}).get("smoke_test", False):
        smoke_samples = config.get("experiment", {}).get("smoke_test_samples", 32)
        logger.info(f"Running SMOKE TEST with {smoke_samples} samples")
        df = df.sample(min(smoke_samples, len(df)), random_state=seed).reset_index(drop=True)
        config["training"]["epochs"] = 1

    splitting_cfg = config.get("splitting", {})
    if splitting_cfg.get("existing_splits"):
        train_df, val_df, test_df = load_existing_splits(splitting_cfg["existing_splits"])
        # To strictly enforce presence, we rely on PathologyDataset dropping invalid patients.
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

    train_transform = get_pathology_transforms(path_cfg, is_training=True)
    eval_transform = get_pathology_transforms(path_cfg, is_training=False)
    num_classes = dataset_cfg.get("num_classes", 2)

    kwargs = {
        "patient_id_col": dataset_cfg.get("patient_id_col", "patient_id"),
        "label_col": dataset_cfg.get("label_col", "label"),
        "patches_dir": path_cfg.get("patches_dir"),
        "patches_manifest_col": path_cfg.get("patches_manifest_col"),
        "pathology_path_col": dataset_cfg.get("pathology_path_col", "pathology_path"),
        "max_patches": path_cfg.get("max_patches_per_patient", 100),
        "num_classes": num_classes,
    }

    train_ds = PathologyDataset(train_df, transform=train_transform, **kwargs)
    val_ds = PathologyDataset(val_df, transform=eval_transform, **kwargs)
    test_ds = PathologyDataset(test_df, transform=eval_transform, **kwargs)

    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = config.get("device", {}).get("num_workers", 4)
    pin_memory = config.get("device", {}).get("pin_memory", True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pathology_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pathology_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=pathology_collate_fn)

    model = PathologyEncoder(
        backbone=path_cfg.get("backbone", "resnet50"),
        pretrained=path_cfg.get("pretrained", True),
        patch_embedding_dim=path_cfg.get("patch_embedding_dim", 512),
        embedding_dim=path_cfg.get("embedding_dim", 512),
        attention_hidden_dim=path_cfg.get("attention_hidden_dim", 256),
        dropout=path_cfg.get("dropout", 0.3),
        num_classes=num_classes,
    )
    wrapped_model = DictInputWrapper(model)

    trainer = Trainer(
        model=wrapped_model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        logger=logger,
        extract_fn=extract_pathology_batch,
    )

    results = trainer.train()

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
    
    plot_training_curves(results["history"], figs_dir, exp_name, formats)
    plot_confusion_matrix(test_preds["true_labels"], test_preds["predicted_labels"], figs_dir, exp_name, class_names, formats)

    if num_classes == 2:
        plot_roc_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats)
        plot_pr_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats)
        plot_calibration_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats=formats)

    logger.info("Pathology experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal/pathology.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.smoke_test:
        overrides["experiment"] = {"smoke_test": True}

    run_experiment(args.config, overrides)
