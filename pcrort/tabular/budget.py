"""Choose the FedAvg communication budget from training and pilot data only.

For each dataset and regime the smallest candidate budget is kept whose mean
pilot log loss is within ``tolerance`` of the largest budget's and whose
remaining training-objective decrease is at most ``tolerance``. One budget is
used for the whole experiment: the largest of the per-cell choices. Test
Brier is recorded for reporting but never used in the choice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .analysis import save_table
from .config import ExperimentConfig
from .datasets import load_dataset, resolve_datasets
from .models import fit_logistic, objective, train_fedavg
from .preprocessing import FederatedPreprocessor
from .runner import repetition_seed
from .splits import assign_clients, split_dataset


def _log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def budget_study(config: ExperimentConfig, budgets=(20, 50, 100), tolerance=1e-3,
                 repetitions: int | None = None, log=print) -> dict:
    budgets = sorted(budgets)
    repetitions = repetitions or config.repetitions
    out = config.run_dir / "fedavg_budget"
    rows = []
    for index, spec in enumerate(resolve_datasets(config.datasets, config.dataset_root)):
        data = load_dataset(spec)
        for regime_index, regime in enumerate(config.clients.regimes):
            for repetition in range(repetitions):
                seed = repetition_seed(config, index, repetition)
                (train, pilot, test), _ = split_dataset(data, seed, config.split.ratios,
                                                        config.split.min_train_class,
                                                        config.split.min_eval_class)
                client_id, _ = assign_clients(data.y[train], regime, seed + 20000000 + regime_index * 10000,
                                              config.clients.count, config.clients.dirichlet_alpha,
                                              config.clients.min_client_class)
                prep = FederatedPreprocessor().fit(data.x.iloc[train], client_id)
                xt, xp, xe = (prep.transform(data.x.iloc[p]) for p in (train, pilot, test))
                yt, yp, ye = data.y[train], data.y[pilot], data.y[test]
                pooled, _ = fit_logistic(xt, yt, config.l2)
                optimum = objective(pooled, xt, yt, config.l2)[0]
                for rounds in budgets:
                    beta, _ = train_fedavg(xt, yt, client_id, rounds, config.fedavg.local_epochs,
                                           config.fedavg.learning_rate, config.l2)
                    value = objective(beta, xt, yt, config.l2)[0]
                    rows.append(dict(dataset=data.name, regime=regime, rep=repetition, rounds=rounds,
                                     train_objective=value, gap_to_pooled=value - optimum,
                                     pilot_log_loss=_log_loss(yp, expit(xp @ beta)),
                                     test_brier=float(np.mean((expit(xe @ beta) - ye) ** 2))))
            log(f"budget study: {data.name} {regime} done")
    raw = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw.csv", index=False)
    table = raw.groupby(["dataset", "regime", "rounds"], as_index=False)[
        ["train_objective", "gap_to_pooled", "pilot_log_loss", "test_brier"]].mean()
    save_table(table, out, "convergence", 5)

    choices = {}
    largest = budgets[-1]
    for (dataset, regime), g in table.groupby(["dataset", "regime"]):
        g = g.set_index("rounds")
        ok = [b for b in budgets
              if abs(g.loc[b, "pilot_log_loss"] - g.loc[largest, "pilot_log_loss"]) <= tolerance
              and g.loc[b, "train_objective"] - g.loc[largest, "train_objective"] <= tolerance]
        choices[f"{dataset}/{regime}"] = min(ok) if ok else largest
    selected = max(choices.values())
    result = dict(budgets=budgets, tolerance=tolerance, cell_choices=choices, selected_rounds=selected,
                  largest_budget_gap_to_pooled=float(table[table.rounds == largest].gap_to_pooled.max()))
    (out / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot(table, selected, out)
    return result


def _plot(table, selected, out: Path):
    from .figures import _pyplot
    plt = _pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for (dataset, regime), g in table.groupby(["dataset", "regime"]):
        style = dict(ls="-" if regime == "iid" else "--", marker="o", ms=3, lw=1)
        ax[0].plot(g.rounds, g.gap_to_pooled, **style)
        ax[1].plot(g.rounds, g.pilot_log_loss, **style)
        ax[2].plot(g.rounds, g.test_brier, label=f"{dataset} {regime}", **style)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Train objective - pooled optimum")
    ax[1].set_ylabel("Pilot log loss (used for the choice)")
    ax[2].set_ylabel("Test Brier (reporting only)")
    for a in ax:
        a.set_xticks(sorted(table.rounds.unique()))
        a.set_xlabel("Rounds")
        a.axvline(selected, color="k", lw=0.6)
    ax[2].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=600)
    plt.close(fig)
