"""
Train a genomics-only model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pcrort.multimodal.datasets.genomics_dataset import GenomicsDataset, load_expression_matrix
from pcrort.multimodal.preprocessing.genomics import GenomicsPreprocessor
from pcrort.multimodal.preprocessing.patient_matching import (
    create_patient_splits,
    load_existing_splits,
    load_metadata,
)
from pcrort.multimodal.evaluation.metrics import save_metrics, save_predictions
from pcrort.multimodal.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)
from pcrort.multimodal.models.genomics_encoder import GenomicsEncoder
from pcrort.multimodal.training.trainer import Trainer
from pcrort.multimodal.utils.config import load_config, resolve_paths
from pcrort.multimodal.utils.logging import ExperimentLogger
from pcrort.multimodal.utils.seed import get_device, print_device_info, set_seed


def extract_genomics_batch(batch: Dict[str, Any], device: torch.device):
    return batch["expression"].to(device), batch["label"].to(device)


def run_experiment(config_path: str, overrides: Dict[str, Any] = None):
    config = load_config(["configs/multimodal/base.yaml", config_path], overrides=overrides)
    config = resolve_paths(config, Path.cwd())

    exp_name = config.get("experiment", {}).get("name", "genomics_experiment")
    logger = ExperimentLogger(exp_name, config.get("logging", {}).get("log_dir", "outputs/logs"))
    logger.log_hyperparameters(config)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("device", {}).get("use_cuda", True))

    dataset_cfg = config.get("dataset", {})
    gen_cfg = config.get("genomics", {})
    metadata_csv = dataset_cfg.get("metadata_csv")

    if not metadata_csv or not Path(metadata_csv).exists():
        logger.error(f"Metadata CSV not found: {metadata_csv}")
        return

    df = load_metadata(
        metadata_csv,
        patient_id_col=dataset_cfg.get("patient_id_col", "patient_id"),
        label_col=dataset_cfg.get("label_col", "label"),
    )

    # Load expression data
    try:
        expr_df = load_expression_matrix(
            metadata_df=df,
            patient_id_col=dataset_cfg.get("patient_id_col", "patient_id"),
            expression_matrix_path=gen_cfg.get("expression_matrix"),
            expression_file_col=gen_cfg.get("expression_file_col"),
            transpose=gen_cfg.get("transpose", False),
        )
    except Exception as e:
        logger.error(f"Failed to load expression data: {e}")
        return

    # Filter metadata to match available expression data
    df = df[df[dataset_cfg.get("patient_id_col", "patient_id")].astype(str).isin(expr_df.index)].reset_index(drop=True)

    if config.get("experiment", {}).get("smoke_test", False):
        smoke_samples = config.get("experiment", {}).get("smoke_test_samples", 32)
        df = df.sample(min(smoke_samples, len(df)), random_state=seed).reset_index(drop=True)
        expr_df = expr_df.loc[df[dataset_cfg.get("patient_id_col", "patient_id")].astype(str)]
        config["training"]["epochs"] = 1

    splitting_cfg = config.get("splitting", {})
    if splitting_cfg.get("existing_splits"):
        train_df, val_df, test_df = load_existing_splits(splitting_cfg["existing_splits"])
        # filter valid
        pid_col = dataset_cfg.get("patient_id_col", "patient_id")
        train_df = train_df[train_df[pid_col].astype(str).isin(expr_df.index)].reset_index(drop=True)
        val_df = val_df[val_df[pid_col].astype(str).isin(expr_df.index)].reset_index(drop=True)
        test_df = test_df[test_df[pid_col].astype(str).isin(expr_df.index)].reset_index(drop=True)
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

    # Preprocessing
    preprocessor = GenomicsPreprocessor(config=gen_cfg)
    train_expr_df = expr_df.loc[train_df[dataset_cfg.get("patient_id_col", "patient_id")].astype(str)]
    val_expr_df = expr_df.loc[val_df[dataset_cfg.get("patient_id_col", "patient_id")].astype(str)]
    test_expr_df = expr_df.loc[test_df[dataset_cfg.get("patient_id_col", "patient_id")].astype(str)]

    train_expr_arr = preprocessor.fit_transform(train_expr_df)
    val_expr_arr = preprocessor.transform(val_expr_df)
    test_expr_arr = preprocessor.transform(test_expr_df)

    # Save preprocessor state
    ckpt_cfg = config.get("checkpoint", {})
    preproc_path = Path(ckpt_cfg.get("save_dir", "outputs/models")) / f"{exp_name}_preprocessor.json"
    preprocessor.save(preproc_path)
    
    num_classes = dataset_cfg.get("num_classes", 2)
    pid_col = dataset_cfg.get("patient_id_col", "patient_id")
    label_col = dataset_cfg.get("label_col", "label")

    train_ds = GenomicsDataset(train_df, train_expr_arr, pid_col, label_col, num_classes)
    val_ds = GenomicsDataset(val_df, val_expr_arr, pid_col, label_col, num_classes)
    test_ds = GenomicsDataset(test_df, test_expr_arr, pid_col, label_col, num_classes)

    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("device", {}).get("num_workers", 4)
    pin_memory = config.get("device", {}).get("pin_memory", True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = GenomicsEncoder(
        input_dim=preprocessor.get_input_dim(),
        hidden_dims=gen_cfg.get("hidden_dims", [1024, 512, 256]),
        embedding_dim=gen_cfg.get("embedding_dim", 256),
        dropout=gen_cfg.get("dropout", 0.3),
        activation=gen_cfg.get("activation", "gelu"),
        normalization=gen_cfg.get("normalization", "layernorm"),
        num_classes=num_classes,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        logger=logger,
        extract_fn=extract_genomics_batch,
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

    logger.info("Genomics experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal/genomics.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.smoke_test:
        overrides["experiment"] = {"smoke_test": True}

    run_experiment(args.config, overrides)
