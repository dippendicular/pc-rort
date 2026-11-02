"""Command line interface: ``pcrort <command> --help`` for details."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path


def _config(args):
    from .tabular import ExperimentConfig
    overrides = dict(name=args.name, repetitions=args.repetitions, workers=args.workers,
                     dataset_root=args.datasets, output_dir=args.output)
    if args.config:
        return ExperimentConfig.from_yaml(args.config, **overrides)
    return ExperimentConfig.from_dict({k: v for k, v in overrides.items() if v is not None})


def _experiment_arguments(parser):
    parser.add_argument("--config", help="experiment YAML (defaults are used when omitted)")
    parser.add_argument("--name", help="run name; results go to <output>/<name>")
    parser.add_argument("--datasets", help="folder searched for datasets (default: datasets)")
    parser.add_argument("--output", help="output folder (default: outputs)")
    parser.add_argument("--repetitions", type=int, help="repeated holdout splits per dataset")
    parser.add_argument("--workers", type=int, help="parallel processes")


def cmd_datasets(args):
    from .tabular import discover
    results = discover(args.root)
    if not results:
        print(f"No table files found under {args.root}")
        return
    for r in results:
        if r.usable:
            m = r.metadata
            print(f"[ok]   {r.name:<28} {m['rows']:>6} rows  {len(m['feature_columns']):>3} features  "
                  f"target={m['target']!r} positive={m['positive_label']} prevalence={m['prevalence']:.3f}")
            dropped = {c: why for c, why in m["excluded_columns"].items() if why != "target"}
            if dropped:
                print(f"       excluded: " + ", ".join(f"{c} ({why})" for c, why in dropped.items()))
        else:
            print(f"[skip] {r.file}: {r.reason}")


def cmd_run(args):
    from .tabular import analyze, make_figures, run_experiment
    config = _config(args)
    run_dir = run_experiment(config)
    if not args.no_analysis:
        analyze(run_dir)
        make_figures(run_dir)
        print(f"tables and figures written under {run_dir}")


def cmd_analyze(args):
    from .tabular import analyze, make_figures
    analyze(args.run)
    if not args.no_figures:
        make_figures(args.run)
    print(f"analysis written under {args.run}")


def cmd_budget(args):
    from .tabular import budget_study
    result = budget_study(_config(args), args.budgets, args.tolerance, args.budget_repetitions)
    print(f"selected FedAvg rounds: {result['selected_rounds']} "
          f"(set fedavg.rounds in the experiment config)")


def cmd_demo(args):
    import numpy as np
    from .matrix import two_candidate_robust_selection
    full = np.array([[1.0, 1.5], [1.5, 9.0]])
    diagonal = np.diag([1.0, 9.0])
    for name, matrix in (("full covariance", full), ("diagonal only", diagonal)):
        fit = two_candidate_robust_selection(matrix, radius=0.0, rho=0.5)
        print(f"{name}: external weight {fit.external_weight:.4f}, contrast {fit.robust_contrast:.4f}")


def cmd_web(args):
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Install the web extras first: pip install -e '.[web]'")
    uvicorn.run("pcrort.web.main:app", host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pcrort", description="PC-RORT federated evaluation tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("datasets", help="list the datasets found in a folder and what was inferred")
    p.add_argument("root", nargs="?", default="datasets")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("run", help="run the repeated-holdout evaluation, then analyze it")
    _experiment_arguments(p)
    p.add_argument("--no-analysis", action="store_true", help="skip tables and figures")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="recompute statistics, tables and figures for a run")
    p.add_argument("run", type=Path, help="run directory, e.g. outputs/experiment")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("fedavg-budget", help="choose FedAvg rounds from pilot and training loss")
    _experiment_arguments(p)
    p.add_argument("--budgets", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--budget-repetitions", type=int, help="repetitions for the study (default: all)")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("demo", help="two-candidate covariance example")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("web", help="start the web portal (needs .env and the web extras)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_web)

    sub.add_parser("version", help="print the installed version").set_defaults(
        func=lambda args: print(version("pcrort")))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
