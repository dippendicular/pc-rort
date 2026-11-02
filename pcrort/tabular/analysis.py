"""Statistics and tables for a completed run directory.

Recipients share one test split per repetition, so metrics are averaged over
recipients first and each repetition is one paired observation. Intervals use
the corrected repeated-holdout variance of Nadeau and Bengio,
SE = sqrt((1/R + n_test/n_train) * s^2), because overlapping resamples make
the ordinary s/sqrt(R) far too small.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .config import METHODS, MIXTURES

METRICS = ["brier", "auroc", "auprc", "log_loss", "sensitivity", "specificity", "fn", "fp",
           "accuracy", "balanced_accuracy", "sensitivity_at_pilot_threshold",
           "specificity_at_pilot_threshold", "fn_at_pilot_threshold", "fp_at_pilot_threshold"]


def holm(p_values) -> np.ndarray:
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def corrected_interval(delta, test_train_ratio: float, level: float = 0.95):
    d = np.asarray(delta, float)
    n = len(d)
    mean = d.mean()
    if n < 2:
        return mean, np.nan, np.nan, np.nan, np.nan
    se = np.sqrt((1 / n + test_train_ratio) * d.var(ddof=1))
    half = student_t.ppf(0.5 + level / 2, n - 1) * se
    p = float(2 * student_t.sf(abs(mean / se), n - 1)) if se > 0 else (1.0 if mean == 0 else 0.0)
    return mean, mean - half, mean + half, se, p


def ordinary_interval(delta, level: float = 0.95):
    d = np.asarray(delta, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    half = student_t.ppf(0.5 + level / 2, n - 1) * se
    p = float(2 * student_t.sf(abs(d.mean() / se), n - 1)) if se > 0 else 1.0
    return d.mean() - half, d.mean() + half, p


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines) + "\n"


def save_table(frame: pd.DataFrame, folder: Path, name: str, digits: int = 4) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.md").write_text(markdown_table(frame, digits), encoding="utf-8")


def load_records(run_dir: Path):
    for path in sorted((run_dir / "records").rglob("rep_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        yield path.parents[1].name, path.parent.name, int(path.stem[-3:]), record


def repetition_means(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in METRICS + ["negative_transfer", "external_weight", "train_n", "test_n",
                                      "test_positive"] if c in metrics]
    return metrics.groupby(["dataset", "regime", "rep", "method"], as_index=False)[columns].mean()


def paired_comparisons(rep: pd.DataFrame, methods, comparators, metrics=("brier",)) -> pd.DataFrame:
    rows = []
    for (dataset, regime), group in rep.groupby(["dataset", "regime"]):
        ratio = float((group.test_n / group.train_n).mean())
        for metric in metrics:
            pivot = group.pivot(index="rep", columns="method", values=metric)
            for method in methods:
                for comparator in comparators:
                    if method == comparator or method not in pivot or comparator not in pivot:
                        continue
                    delta = (pivot[method] - pivot[comparator]).dropna()
                    if len(delta) < 2:
                        continue
                    estimate, low, high, se, p = corrected_interval(delta, ratio)
                    naive_low, naive_high, naive_p = ordinary_interval(delta)
                    rows.append(dict(dataset=dataset, regime=regime, metric=metric, method=method,
                                     comparator=comparator, repetitions=len(delta), estimate=estimate,
                                     ci_low=low, ci_high=high, corrected_se=se, p_value=p,
                                     naive_ci_low=naive_low, naive_ci_high=naive_high,
                                     naive_p=naive_p, test_train_ratio=ratio))
    return pd.DataFrame(rows)


def analyze(run_dir: str | Path, primary_methods=None, comparators=None, alpha=None) -> Path:
    """Write ``statistics/`` and ``tables/`` for a run and return the tables folder."""
    run_dir = Path(run_dir)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    settings = protocol.get("analysis", {})
    primary_methods = primary_methods or settings.get("primary_methods", ["FULL", "ADAPTIVE"])
    comparators = comparators or settings.get("comparators", ["LOCAL", "FedAvg", "POOLED"])
    alpha = alpha or settings.get("alpha", 0.05)
    stats, tables = run_dir / "statistics", run_dir / "tables"
    stats.mkdir(exist_ok=True)

    metrics = pd.read_csv(run_dir / "metrics.csv")
    rep = repetition_means(metrics)
    rep.to_csv(stats / "repetition_means.csv", index=False)
    datasets = pd.read_csv(run_dir / "datasets.csv")
    order = list(datasets.name)
    regimes = list(dict.fromkeys(metrics.regime))

    summary = rep.groupby(["dataset", "regime", "method"])[
        [c for c in METRICS + ["negative_transfer", "external_weight"] if c in rep]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(stats / "summary.csv", index=False)

    # Primary family: every primary method against every comparator, all cells, Holm once.
    primary = paired_comparisons(rep, primary_methods, comparators)
    primary["p_holm"] = holm(primary.p_value)
    primary["naive_p_holm"] = holm(primary.naive_p)
    primary["significant"] = primary.p_holm < alpha
    primary.to_csv(stats / "primary_comparisons.csv", index=False)
    secondary = paired_comparisons(rep, METHODS, comparators, [m for m in METRICS if m in rep])
    secondary.to_csv(stats / "all_comparisons.csv", index=False)

    datasets_table = datasets.rename(columns={"name": "dataset"})
    splits = []
    for dataset, regime, repetition, record in load_records(run_dir):
        if regime == regimes[0]:
            splits.append(dict(dataset=dataset, train=record["split"]["sizes"][0],
                               pilot=record["split"]["sizes"][1], test=record["split"]["sizes"][2]))
    if splits:
        sizes = pd.DataFrame(splits).groupby("dataset").agg(["min", "max"])
        for part in ("train", "pilot", "test"):
            datasets_table[part] = [
                f"{sizes.loc[d, (part, 'min')]}" if sizes.loc[d, (part, 'min')] == sizes.loc[d, (part, 'max')]
                else f"{sizes.loc[d, (part, 'min')]}-{sizes.loc[d, (part, 'max')]}" for d in datasets_table.dataset]
    save_table(datasets_table, tables, "datasets", 3)

    brier = rep.groupby(["dataset", "regime", "method"]).brier.mean().unstack()
    brier = brier.reindex(columns=[m for m in METHODS if m in brier]).reset_index()
    brier["dataset"] = pd.Categorical(brier.dataset, order)
    save_table(brier.sort_values(["dataset", "regime"]), tables, "brier_by_method")

    main = []
    for (dataset, regime, method), g in rep.groupby(["dataset", "regime", "method"]):
        main.append(dict(dataset=dataset, regime=regime, method=method,
                         brier_mean=g.brier.mean(), brier_sd=g.brier.std(),
                         auroc_mean=g.auroc.mean(), auroc_sd=g.auroc.std(),
                         auprc_mean=g.auprc.mean(), log_loss_mean=g.log_loss.mean()))
    save_table(pd.DataFrame(main), tables, "main_results")

    vs_local = []
    for (dataset, regime), g in rep.groupby(["dataset", "regime"]):
        pivot = g.pivot(index="rep", columns="method", values="brier")
        ratio = float((g.test_n / g.train_n).mean())
        local, pooled = pivot.LOCAL.mean(), pivot.POOLED.mean()
        for method in [m for m in METHODS if m != "LOCAL" and m in pivot]:
            estimate, low, high, _, p = corrected_interval(pivot[method] - pivot.LOCAL, ratio)
            vs_local.append(dict(
                dataset=dataset, regime=regime, method=method, delta_brier=estimate, ci_low=low,
                ci_high=high, relative_reduction_pct=-100 * estimate / local,
                gap_closed_pct=100 * (local - pivot[method].mean()) / (local - pooled) if local > pooled else np.nan,
                wins=int((pivot[method] < pivot.LOCAL).sum()), repetitions=len(pivot), p_value=p))
    vs_local = pd.DataFrame(vs_local)
    holm_lookup = primary[primary.comparator == "LOCAL"].set_index(["dataset", "regime", "method"]).p_holm
    vs_local["p_holm_primary"] = [holm_lookup.get((d, r, m), np.nan)
                                  for d, r, m in zip(vs_local.dataset, vs_local.regime, vs_local.method)]
    vs_local.to_csv(stats / "versus_local.csv", index=False)
    save_table(vs_local, tables, "versus_local")

    save_table(primary[["dataset", "regime", "method", "comparator", "estimate", "ci_low", "ci_high",
                        "p_value", "p_holm", "significant"]], tables, "primary_comparisons")

    family = []
    for label, subset in [("all", primary)] + [(f"vs {c}", primary[primary.comparator == c]) for c in comparators]:
        for column, kind in (("p_holm", "corrected"), ("naive_p_holm", "ordinary (ignores overlap)")):
            family.append(dict(subset=label, interval=kind, tests=len(subset),
                               lower_brier=int((subset.estimate < 0).sum()),
                               significant_favoring_method=int(((subset.estimate < 0) & (subset[column] < alpha)).sum()),
                               significant_favoring_comparator=int(((subset.estimate > 0) & (subset[column] < alpha)).sum())))
    save_table(pd.DataFrame(family), tables, "significance_counts")

    selection = rep[rep.method.isin(MIXTURES + ["UNIFORM"])].groupby(
        ["dataset", "regime", "method"], as_index=False)[["external_weight", "negative_transfer"]].mean()
    selection.to_csv(stats / "selection.csv", index=False)
    save_table(selection, tables, "weights_and_negative_transfer")

    adaptive, covariance, clients = [], [], []
    for dataset, regime, repetition, record in load_records(run_dir):
        alpha_diag, alpha_structured, alpha_full = record["adaptive_weights"]
        adaptive.append(dict(dataset=dataset, regime=regime, rep=repetition, alpha_diag=alpha_diag,
                             alpha_structured=alpha_structured, alpha_full=alpha_full))
        for k, counts in enumerate(record["clients"]["class_counts"]):
            clients.append(dict(dataset=dataset, regime=regime, rep=repetition, client=k,
                                negative=counts[0], positive=counts[1]))
        recipients = {row["recipient"] for row in record["weights"]}
        for recipient in recipients:
            w = {row["method"]: np.asarray(row["weights"]) for row in record["weights"]
                 if row["recipient"] == recipient}
            risk = {row["method"]: row["brier"] for row in record["metrics"] if row["recipient"] == recipient}
            covariance.append(dict(dataset=dataset, regime=regime, rep=repetition, recipient=recipient,
                                   weight_l1_full_diag=float(np.abs(w["FULL"] - w["DIAG"]).sum()),
                                   brier_full_minus_diag=risk["FULL"] - risk["DIAG"]))
    adaptive = pd.DataFrame(adaptive).groupby(["dataset", "regime"], as_index=False)[
        ["alpha_diag", "alpha_structured", "alpha_full"]].mean()
    save_table(adaptive, tables, "adaptive_covariance_weights")
    covariance = pd.DataFrame(covariance).groupby(["dataset", "regime"], as_index=False)[
        ["weight_l1_full_diag", "brier_full_minus_diag"]].mean()
    save_table(covariance, tables, "full_versus_diag")
    pd.DataFrame(clients).to_csv(stats / "client_class_counts.csv", index=False)

    points = []
    for (dataset, regime, method), g in rep[rep.method.isin(["LOCAL", "FedAvg", "POOLED", "ADAPTIVE"])].groupby(
            ["dataset", "regime", "method"]):
        base = dict(dataset=dataset, regime=regime, method=method, auroc=g.auroc.mean(),
                    auprc=g.auprc.mean(), brier=g.brier.mean(), test_positive=g.test_positive.mean())
        points.append(base | dict(operating_point="fixed", sensitivity=g.sensitivity.mean(),
                                  specificity=g.specificity.mean(), fn=g.fn.mean(), fp=g.fp.mean()))
        if "sensitivity_at_pilot_threshold" in g:
            points.append(base | dict(operating_point="pilot specificity target",
                                      sensitivity=g.sensitivity_at_pilot_threshold.mean(),
                                      specificity=g.specificity_at_pilot_threshold.mean(),
                                      fn=g.fn_at_pilot_threshold.mean(), fp=g.fp_at_pilot_threshold.mean()))
    save_table(pd.DataFrame(points), tables, "operating_points", 3)
    return tables
