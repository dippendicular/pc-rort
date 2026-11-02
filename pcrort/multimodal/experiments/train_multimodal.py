"""
Train multimodal model (bimodal or trimodal).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pcrort.multimodal.datasets.genomics_dataset import load_expression_matrix
from pcrort.multimodal.datasets.multimodal_dataset import MultimodalDataset, multimodal_collate_fn
from pcrort.multimodal.preprocessing.genomics import GenomicsPreprocessor
from pcrort.multimodal.preprocessing.patient_matching import (
    create_patient_splits,
    load_existing_splits,
    load_metadata,
)
from pcrort.multimodal.preprocessing.pathology import get_pathology_transforms
from pcrort.multimodal.preprocessing.radiology import get_radiology_transforms
from pcrort.multimodal.evaluation.metrics import save_metrics, save_predictions
from pcrort.multimodal.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_modality_contributions,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)
from pcrort.multimodal.models.multimodal_model import MultimodalModel
from pcrort.multimodal.training.trainer import Trainer
from pcrort.multimodal.utils.config import load_config, resolve_paths
from pcrort.multimodal.utils.logging import ExperimentLogger
from pcrort.multimodal.utils.seed import get_device, print_device_info, set_seed


def run_experiment(config_path: str, overrides: Dict[str, Any] = None):
    config = load_config(["configs/multimodal/base.yaml", config_path], overrides=overrides)
    config = resolve_paths(config, Path.cwd())

    exp_name = config.get("experiment", {}).get("name", "multimodal_experiment")
    logger = ExperimentLogger(exp_name, config.get("logging", {}).get("log_dir", "outputs/logs"))
    logger.log_hyperparameters(config)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("device", {}).get("use_cuda", True))

    dataset_cfg = config.get("dataset", {})
    modalities_cfg = config.get("modalities", {})
    use_rad = modalities_cfg.get("use_radiology", True)
    use_path = modalities_cfg.get("use_pathology", True)
    use_gen = modalities_cfg.get("use_genomics", True)
    
    metadata_csv = dataset_cfg.get("metadata_csv")
    if not metadata_csv or not Path(metadata_csv).exists():
        logger.error(f"Metadata CSV not found: {metadata_csv}")
        return

    pid_col = dataset_cfg.get("patient_id_col", "patient_id")
    label_col = dataset_cfg.get("label_col", "label")
    df = load_metadata(metadata_csv, pid_col, label_col)

    # Genomics Preprocessing if used
    genomics_input_dim = None
    expr_df = None
    preprocessor = None
    train_expr_arr = val_expr_arr = test_expr_arr = None
    gen_cfg = config.get("genomics", {})
    if use_gen:
        try:
            expr_df = load_expression_matrix(
                metadata_df=df,
                patient_id_col=pid_col,
                expression_matrix_path=gen_cfg.get("expression_matrix"),
                expression_file_col=gen_cfg.get("expression_file_col"),
                transpose=gen_cfg.get("transpose", False),
            )
        except Exception as e:
            logger.error(f"Failed to load expression data: {e}")
            return
        
        # Only keep patients that have genomics data (if strict)
        if dataset_cfg.get("modality_mode", "STRICT_MULTIMODAL") == "STRICT_MULTIMODAL":
             df = df[df[pid_col].astype(str).isin(expr_df.index)].reset_index(drop=True)

    if config.get("experiment", {}).get("smoke_test", False):
        smoke_samples = config.get("experiment", {}).get("smoke_test_samples", 32)
        df = df.sample(min(smoke_samples, len(df)), random_state=seed).reset_index(drop=True)
        config["training"]["epochs"] = 1

    splitting_cfg = config.get("splitting", {})
    if splitting_cfg.get("existing_splits"):
        train_df, val_df, test_df = load_existing_splits(splitting_cfg["existing_splits"])
        train_df = train_df[train_df[pid_col].isin(df[pid_col])].reset_index(drop=True)
        val_df = val_df[val_df[pid_col].isin(df[pid_col])].reset_index(drop=True)
        test_df = test_df[test_df[pid_col].isin(df[pid_col])].reset_index(drop=True)
    else:
        train_df, val_df, test_df = create_patient_splits(
            df, pid_col, label_col,
            train_ratio=splitting_cfg.get("train_ratio", 0.7),
            val_ratio=splitting_cfg.get("val_ratio", 0.15),
            test_ratio=splitting_cfg.get("test_ratio", 0.15),
            stratify=splitting_cfg.get("stratify", True),
            seed=seed,
            splits_dir=splitting_cfg.get("splits_dir", "outputs/splits"),
        )

    if use_gen:
        preprocessor = GenomicsPreprocessor(config=gen_cfg)
        # Handle patients missing in expression data gracefully if allowing missing modalities
        missing_train = set(train_df[pid_col].astype(str)) - set(expr_df.index)
        if missing_train:
             logger.warning(f"Missing expression for {len(missing_train)} training patients.")
             # Fill with zeros for missing
             for pid in missing_train:
                  expr_df.loc[pid] = 0.0

        train_expr_df = expr_df.loc[train_df[pid_col].astype(str)]
        val_expr_df = expr_df.loc[val_df[pid_col].astype(str).intersection(expr_df.index)]
        test_expr_df = expr_df.loc[test_df[pid_col].astype(str).intersection(expr_df.index)]

        train_expr_arr = preprocessor.fit_transform(train_expr_df)
        
        # Missing imputation handles any test missing
        missing_val = set(val_df[pid_col].astype(str)) - set(expr_df.index)
        for pid in missing_val: val_expr_df.loc[pid] = 0.0
        missing_test = set(test_df[pid_col].astype(str)) - set(expr_df.index)
        for pid in missing_test: test_expr_df.loc[pid] = 0.0

        val_expr_arr = preprocessor.transform(val_expr_df.loc[val_df[pid_col].astype(str)])
        test_expr_arr = preprocessor.transform(test_expr_df.loc[test_df[pid_col].astype(str)])
        
        genomics_input_dim = preprocessor.get_input_dim()
        preproc_path = Path(config.get("checkpoint", {}).get("save_dir", "outputs/models")) / f"{exp_name}_preprocessor.json"
        preprocessor.save(preproc_path)

    num_classes = dataset_cfg.get("num_classes", 2)
    path_cfg = config.get("pathology", {})
    rad_cfg = config.get("radiology", {})

    kwargs_common = {
        "patient_id_col": pid_col,
        "label_col": label_col,
        "num_classes": num_classes,
        "use_radiology": use_rad,
        "use_pathology": use_path,
        "use_genomics": use_gen,
        "radiology_path_col": dataset_cfg.get("radiology_path_col"),
        "pathology_path_col": dataset_cfg.get("pathology_path_col"),
        "patches_dir": path_cfg.get("patches_dir"),
        "patches_manifest_col": path_cfg.get("patches_manifest_col"),
        "max_patches": path_cfg.get("max_patches_per_patient", 100),
        "modality_mode": dataset_cfg.get("modality_mode", "STRICT_MULTIMODAL"),
    }

    train_ds = MultimodalDataset(
        train_df,
        radiology_transform=get_radiology_transforms(rad_cfg, True) if use_rad else None,
        pathology_transform=get_pathology_transforms(path_cfg, True) if use_path else None,
        genomics_expression=train_expr_arr,
        genomics_patient_ids=train_df[pid_col].astype(str).tolist() if use_gen else None,
        **kwargs_common
    )
    val_ds = MultimodalDataset(
        val_df,
        radiology_transform=get_radiology_transforms(rad_cfg, False) if use_rad else None,
        pathology_transform=get_pathology_transforms(path_cfg, False) if use_path else None,
        genomics_expression=val_expr_arr,
        genomics_patient_ids=val_df[pid_col].astype(str).tolist() if use_gen else None,
        **kwargs_common
    )
    test_ds = MultimodalDataset(
        test_df,
        radiology_transform=get_radiology_transforms(rad_cfg, False) if use_rad else None,
        pathology_transform=get_pathology_transforms(path_cfg, False) if use_path else None,
        genomics_expression=test_expr_arr,
        genomics_patient_ids=test_df[pid_col].astype(str).tolist() if use_gen else None,
        **kwargs_common
    )

    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = config.get("device", {}).get("num_workers", 4)
    pin_memory = config.get("device", {}).get("pin_memory", True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=multimodal_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=multimodal_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=multimodal_collate_fn)

    model = MultimodalModel(config=config, genomics_input_dim=genomics_input_dim)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        logger=logger,
        # No extract_fn needed, multimodal model processes the batch dict directly
    )

    results = trainer.train()

    # Inference again on test set to extract fusion weights correctly (the Trainer does not store them all by default)
    logger.log_section("EVALUATION & PLOTS")
    model.eval()
    all_fusion_weights = {}
    if model.fusion is not None and config.get("fusion", {}).get("return_weights", True):
        for batch in test_loader:
             batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
             with torch.no_grad():
                  outputs = model(batch_gpu)
             f_weights = outputs.get("fusion_weights", {})
             for m, w in f_weights.items():
                  if m not in all_fusion_weights:
                       all_fusion_weights[m] = []
                  all_fusion_weights[m].extend(w.cpu().numpy().tolist())

        avg_weights = {m: sum(w)/len(w) for m, w in all_fusion_weights.items()}
        plot_modality_contributions(
            avg_weights,
            Path(config.get("evaluation", {}).get("figures_dir", "outputs/figures")),
            exp_name,
            config.get("evaluation", {}).get("figure_format", ["png", "pdf"])
        )

    eval_cfg = config.get("evaluation", {})
    metrics_dir = Path(eval_cfg.get("metrics_dir", "outputs/metrics"))
    preds_dir = Path(eval_cfg.get("predictions_dir", "outputs/predictions"))
    figs_dir = Path(eval_cfg.get("figures_dir", "outputs/figures"))
    formats = eval_cfg.get("figure_format", ["png", "pdf"])
    class_names = dataset_cfg.get("class_names")

    test_metrics = results["test_metrics"]
    test_preds = results["test_predictions"]

    save_metrics(test_metrics, metrics_dir, exp_name)
    save_predictions(test_preds, preds_dir, exp_name, fusion_weights=all_fusion_weights if all_fusion_weights else None)
    
    plot_training_curves(results["history"], figs_dir, exp_name, formats)
    plot_confusion_matrix(test_preds["true_labels"], test_preds["predicted_labels"], figs_dir, exp_name, class_names, formats)

    if num_classes == 2:
        plot_roc_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats)
        plot_pr_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats)
        plot_calibration_curve(test_preds["true_labels"], test_preds["prediction_probabilities"], figs_dir, exp_name, formats=formats)

    logger.info("Multimodal experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal/multimodal.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.smoke_test:
        overrides["experiment"] = {"smoke_test": True}

    run_experiment(args.config, overrides)
