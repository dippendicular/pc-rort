"""Publication figures for an analyzed run (PNG, 600 DPI)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLORS = {"LOCAL": "#7F7F7F", "UNIFORM": "#B07AA1", "DIAG": "#9CC3E4", "STRUCTURED": "#59A14F",
          "FULL": "#4E79A7", "ADAPTIVE": "#E15759", "FedAvg": "#F28E2B", "POOLED": "#222222"}
ORDER = ["LOCAL", "UNIFORM", "DIAG", "STRUCTURED", "FULL", "ADAPTIVE", "FedAvg", "POOLED"]
REGIME_LABEL = {"iid": "IID", "non_iid": "Non-IID"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})
    return plt


def _cells(rep: pd.DataFrame, datasets: list[str]) -> list[tuple[str, str]]:
    regimes = list(dict.fromkeys(rep.regime))
    return [(d, r) for d in datasets for r in regimes if ((rep.dataset == d) & (rep.regime == r)).any()]


def _cell_labels(cells):
    return [f"{d}\n{REGIME_LABEL.get(r, r)}" for d, r in cells]


def _grouped_bars(ax, cells, methods, value):
    x = np.arange(len(cells))
    width = 0.8 / len(methods)
    for k, method in enumerate(methods):
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, [value(c, method) for c in cells], width,
               color=COLORS[method], label=method)
    ax.set_xticks(x, _cell_labels(cells), fontsize=6)
    return x


def make_figures(run_dir: str | Path) -> Path:
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    run_dir = Path(run_dir)
    stats, tables, out = run_dir / "statistics", run_dir / "tables", run_dir / "figures"
    out.mkdir(exist_ok=True)
    rep = pd.read_csv(stats / "repetition_means.csv")
    datasets = list(pd.read_csv(run_dir / "datasets.csv").name)
    cells = _cells(rep, datasets)
    methods = [m for m in ORDER if m in set(rep.method)]
    vs_local = pd.read_csv(stats / "versus_local.csv").set_index(["dataset", "regime", "method"])
    selection = pd.read_csv(stats / "selection.csv").set_index(["dataset", "regime", "method"])
    primary = pd.read_csv(stats / "primary_comparisons.csv")
    wide = max(7.2, 0.6 * len(cells))

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=600)
        plt.close(fig)

    # Datasets and client heterogeneity.
    info = pd.read_csv(run_dir / "datasets.csv").set_index("name").loc[datasets]
    clients = pd.read_csv(stats / "client_class_counts.csv")
    clients["rate"] = clients.positive / (clients.positive + clients.negative)
    fig, ax = plt.subplots(1, 2, figsize=(wide, 2.6))
    x = np.arange(len(datasets))
    ax[0].bar(x, info.rows, color="#4E79A7")
    ax[0].set_xticks(x, datasets, rotation=30, ha="right")
    ax[0].set_ylabel("Records")
    twin = ax[0].twinx()
    twin.plot(x, info.prevalence, "o", color="#E15759")
    twin.set_ylim(0, 1)
    twin.set_ylabel("Prevalence", color="#E15759")
    ax[0].set_title("a  Datasets", loc="left")
    regimes = list(dict.fromkeys(rep.regime))
    for j, regime in enumerate(regimes):
        offset = (j - (len(regimes) - 1) / 2) * 0.36
        color = ["#4E79A7", "#E15759", "#59A14F"][j % 3]
        boxes = ax[1].boxplot([clients[(clients.dataset == d) & (clients.regime == regime)].rate for d in datasets],
                              positions=x + offset, widths=0.3, showfliers=False, patch_artist=True)
        for box in boxes["boxes"]:
            box.set(facecolor=color, alpha=0.5)
    ax[1].set_xticks(x, datasets, rotation=30, ha="right")
    ax[1].set_ylabel("Client positive rate")
    ax[1].legend([Line2D([], [], color=["#4E79A7", "#E15759", "#59A14F"][j % 3], lw=6, alpha=0.5)
                  for j in range(len(regimes))], [REGIME_LABEL.get(r, r) for r in regimes])
    ax[1].set_title("b  Client label heterogeneity", loc="left")
    fig.tight_layout()
    save(fig, "datasets_and_clients")

    # Mean Brier for every method in every cell.
    fig, axes = plt.subplots(len(regimes), len(datasets), figsize=(max(7.2, 1.2 * len(datasets)), 1.8 * len(regimes)),
                             squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, regime in enumerate(regimes):
            a = axes[j, i]
            g = rep[(rep.dataset == dataset) & (rep.regime == regime)].groupby("method").brier
            if g.ngroups == 0:
                a.axis("off")
                continue
            mean, sem = g.mean().reindex(methods), g.std().reindex(methods) / np.sqrt(g.size().reindex(methods))
            a.barh(range(len(methods)), mean, xerr=sem, color=[COLORS[m] for m in methods], height=0.7)
            a.set_yticks(range(len(methods)), methods if i == 0 else [])
            a.invert_yaxis()
            a.set_xlim(mean.min() * 0.85, mean.max() * 1.05)
            a.xaxis.set_major_locator(MaxNLocator(3))
            a.tick_params(axis="x", labelsize=6)
            if j == 0:
                a.set_title(dataset)
            if i == 0:
                a.set_ylabel(REGIME_LABEL.get(regime, regime))
    fig.supxlabel("Mean test Brier (lower is better)", fontsize=8)
    fig.tight_layout()
    save(fig, "brier_by_method")

    # Primary mixtures minus LOCAL with corrected intervals.
    shown = [m for m in ("FULL", "ADAPTIVE") if m in methods]
    fig, ax = plt.subplots(figsize=(4.2, 0.33 * len(cells) + 1))
    for y, cell in enumerate(cells):
        for k, method in enumerate(shown):
            row = vs_local.loc[cell + (method,)]
            ax.errorbar(row.delta_brier, y + (k - 0.5) * 0.3,
                        xerr=[[row.delta_brier - row.ci_low], [row.ci_high - row.delta_brier]],
                        fmt="o", color=COLORS[method], ms=3.5, capsize=2, lw=1)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(cells)), [f"{d} {REGIME_LABEL.get(r, r)}" for d, r in cells])
    ax.invert_yaxis()
    ax.set_xlabel("Brier difference vs LOCAL (corrected 95% CI)\n< 0 favors the mixture")
    ax.legend([Line2D([], [], marker="o", color=COLORS[m], ls="") for m in shown], shown,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(shown))
    save(fig, "difference_vs_local")

    # Share of the LOCAL-to-POOLED gap closed.
    gap_methods = [m for m in ("UNIFORM", "DIAG", "FULL", "ADAPTIVE", "FedAvg") if m in methods]
    fig, ax = plt.subplots(figsize=(wide, 2.6))
    _grouped_bars(ax, cells, gap_methods, lambda c, m: vs_local.loc[c + (m,)].gap_closed_pct)
    ax.axhline(100, color="k", lw=0.6, ls="--")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("% of LOCAL-to-POOLED gap closed")
    ax.legend(ncol=len(gap_methods), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "gap_closed")

    # Repetitions in which each mixture beats LOCAL.
    mixtures = [m for m in ("UNIFORM", "DIAG", "STRUCTURED", "FULL", "ADAPTIVE") if m in methods]
    fig, ax = plt.subplots(figsize=(wide, 2.4))
    _grouped_bars(ax, cells, mixtures,
                  lambda c, m: 100 * vs_local.loc[c + (m,)].wins / vs_local.loc[c + (m,)].repetitions)
    ax.axhline(50, color="k", lw=0.6, ls="--")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Repetitions better\nthan LOCAL (%)")
    ax.legend(ncol=len(mixtures), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "wins_vs_local")

    # External weight and negative transfer.
    fig, ax = plt.subplots(2, 1, figsize=(wide, 4.6))
    _grouped_bars(ax[0], cells, mixtures, lambda c, m: selection.loc[c + (m,)].external_weight)
    _grouped_bars(ax[1], cells, mixtures, lambda c, m: 100 * selection.loc[c + (m,)].negative_transfer)
    ax[0].set_ylabel("Mean external weight")
    ax[1].set_ylabel("Recipients worse than LOCAL (%)")
    ax[0].set_title("a  Weight placed on other clients", loc="left")
    ax[1].set_title("b  Negative transfer", loc="left")
    ax[0].legend(ncol=len(mixtures), fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "weights_and_negative_transfer")

    # ADAPTIVE weights on the three covariance estimators.
    adaptive = pd.read_csv(tables / "adaptive_covariance_weights.csv").set_index(["dataset", "regime"])
    fig, ax = plt.subplots(figsize=(wide, 2.3))
    bottom = np.zeros(len(cells))
    for column, label, color in (("alpha_diag", "Diagonal", "#9CC3E4"),
                                 ("alpha_structured", "Exchangeable", "#59A14F"),
                                 ("alpha_full", "Full", "#4E79A7")):
        values = adaptive.loc[cells, column].to_numpy()
        ax.bar(np.arange(len(cells)), values, bottom=bottom, color=color, label=label, width=0.7)
        bottom += values
    ax.set_xticks(np.arange(len(cells)), _cell_labels(cells), fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ADAPTIVE aggregation weight")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "adaptive_covariance_weights")

    # FULL vs DIAG: weight disagreement against risk difference.
    full_diag = pd.read_csv(tables / "full_versus_diag.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for marker, regime in zip("os^", regimes):
        g = full_diag[full_diag.regime == regime]
        ax.scatter(g.weight_l1_full_diag, g.brier_full_minus_diag, marker=marker, s=26,
                   label=REGIME_LABEL.get(regime, regime))
        for _, row in g.iterrows():
            ax.annotate(row.dataset, (row.weight_l1_full_diag, row.brier_full_minus_diag),
                        fontsize=5, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Mean L1 weight difference, FULL vs DIAG")
    ax.set_ylabel("Test Brier, FULL - DIAG")
    ax.legend()
    save(fig, "full_versus_diag")

    # AUROC.
    auroc_methods = [m for m in ("LOCAL", "FULL", "ADAPTIVE", "FedAvg", "POOLED") if m in methods]
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(7.2, 1.2 * len(datasets)), 2.4), squeeze=False)
    for i, dataset in enumerate(datasets):
        for marker, regime in zip("os^", regimes):
            g = rep[(rep.dataset == dataset) & (rep.regime == regime)].groupby("method").auroc.mean()
            axes[0, i].scatter(range(len(auroc_methods)), g.reindex(auroc_methods), marker=marker, s=14,
                               color=[COLORS[m] for m in auroc_methods])
        axes[0, i].set_xticks(range(len(auroc_methods)), auroc_methods, rotation=90, fontsize=6)
        axes[0, i].set_title(dataset)
    axes[0, 0].set_ylabel("Mean test AUROC")
    fig.tight_layout()
    save(fig, "auroc")

    # Sensitivity at the fixed and pilot-selected operating points.
    points = pd.read_csv(tables / "operating_points.csv")
    point_methods = [m for m in ("LOCAL", "FedAvg", "POOLED", "ADAPTIVE") if m in methods]
    fig, axes = plt.subplots(len(regimes), len(datasets), figsize=(max(7.2, 1.3 * len(datasets)), 2.0 * len(regimes)),
                             squeeze=False, sharey=True)
    for i, dataset in enumerate(datasets):
        for j, regime in enumerate(regimes):
            a = axes[j, i]
            for k, method in enumerate(point_methods):
                for s, point in enumerate(("fixed", "pilot specificity target")):
                    hit = points[(points.dataset == dataset) & (points.regime == regime) &
                                 (points.method == method) & (points.operating_point == point)]
                    if hit.empty:
                        continue
                    a.bar(k + (s - 0.5) * 0.38, hit.sensitivity.iloc[0], 0.38, color=COLORS[method],
                          alpha=0.45 if s == 0 else 1, hatch="//" if s == 0 else None, edgecolor="white")
            a.set_xticks(range(len(point_methods)), point_methods, rotation=90, fontsize=6)
            if j == 0:
                a.set_title(dataset)
            if i == 0:
                a.set_ylabel(f"{REGIME_LABEL.get(regime, regime)}\nsensitivity")
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc="#999", alpha=0.45, hatch="//"),
                plt.Rectangle((0, 0), 1, 1, fc="#999")],
               ["Fixed threshold", "Pilot specificity target"], loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "operating_points")

    # Corrected versus ordinary intervals for the first primary method against LOCAL.
    method = shown[-1] if shown else None
    if method is not None:
        sub = primary[(primary.method == method) & (primary.comparator == "LOCAL")].set_index(["dataset", "regime"])
        sub = sub.loc[[c for c in cells if c in sub.index]]
        fig, ax = plt.subplots(figsize=(4.2, 0.3 * len(sub) + 1))
        for y, (_, row) in enumerate(sub.iterrows()):
            ax.plot([row.naive_ci_low, row.naive_ci_high], [y - 0.15] * 2, color="#BBBBBB", lw=2.5)
            ax.plot([row.ci_low, row.ci_high], [y + 0.15] * 2, color=COLORS[method], lw=1.5)
            ax.plot(row.estimate, y, "ko", ms=2.5)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(range(len(sub)), [f"{d} {REGIME_LABEL.get(r, r)}" for d, r in sub.index])
        ax.invert_yaxis()
        ax.set_xlabel(f"{method} - LOCAL Brier, 95% interval")
        ax.legend([Line2D([], [], color="#BBBBBB", lw=2.5), Line2D([], [], color=COLORS[method])],
                  ["Ordinary (ignores overlap)", "Corrected repeated-holdout"],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
        save(fig, "corrected_vs_ordinary_intervals")
    return out
