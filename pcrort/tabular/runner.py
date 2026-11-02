"""Repeated-holdout federated evaluation over any set of binary tabular datasets.

For every dataset, client regime and repetition the runner splits the data,
partitions the training rows into clients, fits LOCAL, POOLED and FedAvg
models, selects mixture weights on the pilot split, locks them, and only then
scores the test split. Each repetition is written to disk as it finishes, so
an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import METHODS, ExperimentConfig
from .datasets import DatasetSpec, load_dataset, resolve_datasets
from .models import (classification_metrics, fit_logistic, mixture_candidates, select_mixture,
                     specificity_threshold, train_fedavg)
from .preprocessing import FederatedPreprocessor
from .splits import assign_clients, split_dataset


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def repetition_seed(config: ExperimentConfig, dataset_index: int, repetition: int) -> int:
    return config.seed + dataset_index * 100000 + repetition


def run_repetition(config: ExperimentConfig, data, dataset_index: int, regime: str,
                   repetition: int) -> dict:
    seed = repetition_seed(config, dataset_index, repetition)
    regime_index = config.clients.regimes.index(regime)
    (train, pilot, test), split_info = split_dataset(
        data, seed, config.split.ratios, config.split.min_train_class, config.split.min_eval_class)
    client_id, client_info = assign_clients(
        data.y[train], regime, seed + 20000000 + regime_index * 10000, config.clients.count,
        config.clients.dirichlet_alpha, config.clients.min_client_class)
    clients = config.clients.count

    prep = FederatedPreprocessor().fit(data.x.iloc[train], client_id)
    x_train, x_pilot, x_test = (prep.transform(data.x.iloc[part]) for part in (train, pilot, test))
    y_train, y_pilot, y_test = data.y[train], data.y[pilot], data.y[test]

    local = [fit_logistic(x_train[client_id == k], y_train[client_id == k], config.l2)[0]
             for k in range(clients)]
    pooled, _ = fit_logistic(x_train, y_train, config.l2)
    fed, fed_history = train_fedavg(x_train, y_train, client_id, config.fedavg.rounds,
                                    config.fedavg.local_epochs, config.fedavg.learning_rate, config.l2)

    pilot_local = np.column_stack([expit(x_pilot @ b) for b in local])
    candidates, adaptive = mixture_candidates(pilot_local, y_pilot, config.mixture.adaptive_folds,
                                              config.mixture.adaptive_penalty)
    pilot_shared = {"POOLED": expit(x_pilot @ pooled), "FedAvg": expit(x_pilot @ fed),
                    "UNIFORM": pilot_local.mean(1)}

    # Weights and thresholds for every recipient are fixed before any test prediction.
    weights, thresholds, diagnostics = {}, {}, {}
    for recipient in range(clients):
        for method in METHODS:
            info = {}
            if method == "LOCAL":
                v = np.eye(clients)[recipient]
                pilot_p = pilot_local[:, recipient]
            elif method in candidates:
                v, info = select_mixture(candidates[method], recipient,
                                         config.mixture.external_mass_cap,
                                         config.mixture.radius_fraction,
                                         config.mixture.max_iterations)
                pilot_p = pilot_local @ v
            elif method == "UNIFORM":
                v = np.ones(clients) / clients
                pilot_p = pilot_shared[method]
            else:
                v = None
                pilot_p = pilot_shared[method]
            weights[recipient, method] = v
            diagnostics[recipient, method] = info
            thresholds[recipient, method] = specificity_threshold(
                y_pilot, pilot_p, config.threshold.target_specificity,
                config.threshold.min_pilot_per_class)
    lock = hashlib.sha256(json.dumps(
        {f"{r}:{m}": None if v is None else v.tolist() for (r, m), v in weights.items()},
        sort_keys=True).encode()).hexdigest()

    test_local = np.column_stack([expit(x_test @ b) for b in local])
    test_shared = {"POOLED": expit(x_test @ pooled), "FedAvg": expit(x_test @ fed),
                   "UNIFORM": test_local.mean(1)}
    local_brier = [np.mean((test_local[:, k] - y_test) ** 2) for k in range(clients)]
    rows, weight_rows, arrays = [], [], {}
    for recipient in range(clients):
        for method in METHODS:
            v = weights[recipient, method]
            p = test_local @ v if v is not None else test_shared[method]
            row = dict(dataset=data.name, regime=regime, rep=repetition, seed=seed,
                       recipient=recipient, method=method, train_n=len(train), pilot_n=len(pilot),
                       **classification_metrics(y_test, p, config.threshold.fixed))
            row["paired_brier_vs_local"] = row["brier"] - local_brier[recipient]
            row["negative_transfer"] = int(row["paired_brier_vs_local"] > 1e-12)
            row["external_weight"] = float(1 - v[recipient]) if v is not None else None
            threshold = thresholds[recipient, method]
            row["threshold_pilot"] = threshold
            if threshold is not None:
                at = classification_metrics(y_test, p, threshold)
                for key in ("sensitivity", "specificity", "fn", "fp", "tp", "tn"):
                    row[f"{key}_at_pilot_threshold"] = at[key]
            row["selection_converged"] = diagnostics[recipient, method].get("converged", True)
            rows.append(row)
            if v is not None:
                weight_rows.append(dict(recipient=recipient, method=method, weights=v.tolist(),
                                        **diagnostics[recipient, method]))
            arrays[f"test_{recipient}_{method}"] = p

    record = dict(metrics=rows, weights=weight_rows, lock_digest=lock, split=split_info,
                  clients=client_info, adaptive_weights=adaptive.weights.tolist(),
                  adaptive_cv_losses=adaptive.cv_losses.tolist(),
                  fedavg_history=fed_history, preprocessing=prep.config())
    if config.save_arrays:
        record["_arrays"] = dict(train_indices=train, pilot_indices=pilot, test_indices=test,
                                 client_id=client_id, pilot_labels=y_pilot, test_labels=y_test,
                                 pilot_local=pilot_local, local_parameters=np.stack(local),
                                 pooled_parameters=pooled, fedavg_parameters=fed, **arrays)
    return record


def run_cell(config: ExperimentConfig, spec: DatasetSpec, dataset_index: int,
             regime: str) -> tuple[str, str, int]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    data = load_dataset(spec)
    folder = config.run_dir / "records" / data.name / regime
    rows = []
    for repetition in range(config.repetitions):
        path = folder / f"rep_{repetition:03d}.json"
        if path.exists():
            rows += json.loads(path.read_text(encoding="utf-8"))["metrics"]
            continue
        record = run_repetition(config, data, dataset_index, regime, repetition)
        arrays = record.pop("_arrays", None)
        if arrays is not None:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"rep_{repetition:03d}.npz", **arrays)
        write_json(path, record)
        rows += record["metrics"]
    return data.name, regime, len(rows)


def run_experiment(config: ExperimentConfig, log=print) -> Path:
    out = config.run_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    protocol = json.loads(json.dumps(config.protocol(), default=_json_default))
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != protocol:
            raise RuntimeError(f"{protocol_path} differs from this configuration; "
                               "use a new experiment name instead of overwriting results")
    write_json(protocol_path, protocol)

    specs = resolve_datasets(config.datasets, config.dataset_root)
    summary = []
    for spec in specs:
        data = load_dataset(spec)
        write_json(out / "datasets" / f"{data.name}.json", data.metadata)
        summary.append({k: data.metadata[k] for k in
                        ("name", "title", "rows", "positives", "prevalence", "duplicate_feature_rows")}
                       | {"features": len(data.metadata["feature_columns"])})
    pd.DataFrame(summary).to_csv(out / "datasets.csv", index=False)

    log(f"{len(specs)} dataset(s): " + ", ".join(row["name"] for row in summary))
    cells = [(i, regime) for i in range(len(specs)) for regime in config.clients.regimes]
    if config.workers > 1:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            jobs = [pool.submit(run_cell, config, specs[i], i, regime) for i, regime in cells]
            for job in as_completed(jobs):
                name, regime, count = job.result()
                log(f"finished {name} {regime}: {count} metric rows")
    else:
        for i, regime in cells:
            name, regime, count = run_cell(config, specs[i], i, regime)
            log(f"finished {name} {regime}: {count} metric rows")

    records = sorted((out / "records").rglob("rep_*.json"))
    metrics = pd.DataFrame([row for path in records
                            for row in json.loads(path.read_text(encoding="utf-8"))["metrics"]])
    metrics.to_csv(out / "metrics.csv", index=False)
    log(f"wrote {out / 'metrics.csv'} ({len(metrics)} rows)")
    return out
