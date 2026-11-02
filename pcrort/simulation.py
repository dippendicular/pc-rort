"""Reproducible matrix-level statistical power and uncertainty comparison.

Run from the repository root:
  python -m pcrort.simulation --reps 25 --bootstrap 100 --out results/pcrort_matrix.csv
No patient data, neural training, or clinical inference is performed.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from pcrort.robust import robust_select, UncertaintySet
from pcrort.uncertainty import pilot_risk_matrix, bootstrap_matrices, proposals, contains


def covariance_template(j: int, p: int, common: float) -> np.ndarray:
    """Shared encoder factor plus independent site residual scores."""
    loadings = np.linspace(1., 1.5, j)
    site = common * np.outer(loadings, loadings) + (1. - common) * np.eye(j)
    return np.kron(site, np.eye(p))


def oracle_matrix(j: int, p: int, mismatch: float, common: float, n_final: int) -> np.ndarray:
    b = np.zeros((j, p))
    b[1:, 0] = mismatch
    covariance = covariance_template(j, p, common)
    block_trace = np.array([
        [np.trace(covariance[i*p:(i+1)*p, k*p:(k+1)*p]) / (p * n_final)
         for k in range(j)] for i in range(j)
    ])
    return b @ b.T / p + block_trace


def one_rep(
    rng: np.random.Generator, *, n: int, j: int, p: int, mismatch: float,
    common: float, alpha: float, bootstrap: int, rho: float,
    n_final: int, solver: str, iteration_limit: int
) -> list[dict]:
    true_q = oracle_matrix(j, p, mismatch, common, n_final)
    bias = np.zeros((j, p))
    bias[1:, 0] = mismatch
    covariance = covariance_template(j, p, common)
    x = rng.multivariate_normal(bias.ravel(), covariance, size=n).reshape(n, j, p)
    center = pilot_risk_matrix(x, n_final)
    boot = bootstrap_matrices(x, n_final, bootstrap, rng)
    regions = proposals(center, boot, alpha)
    rows = []
    for name, region in regions.items():
        t0 = time.perf_counter()
        try:
            result = robust_select(region, rho, tolerance=1e-5,
                                   max_iterations=iteration_limit, solver=solver)
            oracle_gain = float(result.weights @ true_q @ result.weights - true_q[0, 0])
            rows.append({
                "method": name, "covered": int(contains(region, true_q)),
                "external_mass": float(1 - result.weights[0]),
                "nonlocal": int(1 - result.weights[0] > 1e-4),
                "robust_contrast": result.objective,
                "true_contrast": oracle_gain, "harmful": int(oracle_gain > 1e-8),
                "iterations": result.iterations, "gap": result.gap,
                "converged": int(result.converged),
                "seconds": time.perf_counter() - t0, "status": "ok",
            })
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            rows.append({
                "method": name, "covered": int(contains(region, true_q)),
                "external_mass": "", "nonlocal": "", "robust_contrast": "",
                "true_contrast": "", "harmful": "", "iterations": "",
                "gap": "", "converged": 0, "seconds": time.perf_counter() - t0,
                "status": str(exc)[:200],
            })
    # Plug-in comparison: one fixed matrix, no uncertainty.
    result = robust_select(UncertaintySet(center, "frobenius", 0.), rho, solver=solver,
                           max_iterations=iteration_limit)
    gain = float(result.weights @ true_q @ result.weights - true_q[0, 0])
    rows.append({
        "method": "plug_in", "covered": "", "external_mass": float(1 - result.weights[0]),
        "nonlocal": int(1 - result.weights[0] > 1e-4),
        "robust_contrast": result.objective, "true_contrast": gain,
        "harmful": int(gain > 1e-8), "iterations": result.iterations,
        "gap": result.gap, "converged": int(result.converged),
        "seconds": "", "status": "ok",
    })
    return rows


def run(args: argparse.Namespace) -> Path:
    root = np.random.SeedSequence(args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "rep", "N", "J0", "p", "mismatch", "common",
              "n_final", "alpha", "method", "covered", "external_mass",
              "nonlocal", "robust_contrast", "true_contrast", "harmful",
              "iterations", "gap", "converged", "seconds", "status"]
    scenario_grid = [(int(n), int(j), int(p), float(b))
                     for n in args.samples for j in args.candidates
                     for p in args.dimensions for b in args.mismatches]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scenario, (n, j, p, mismatch) in enumerate(scenario_grid):
            rngs = [np.random.default_rng(s) for s in root.spawn(args.reps)]
            for rep, rng in enumerate(rngs):
                rows = one_rep(rng, n=n, j=j, p=p, mismatch=mismatch,
                               common=args.shared, alpha=args.alpha,
                               bootstrap=args.bootstrap, rho=args.rho,
                               n_final=args.n_final, solver=args.solver,
                               iteration_limit=args.max_iterations)
                for row in rows:
                    writer.writerow(dict(
                        scenario=scenario, rep=rep, N=n, J0=j, p=p,
                        mismatch=mismatch, common=args.shared,
                        n_final=args.n_final, alpha=args.alpha, **row
                    ))
    metadata = vars(args).copy()
    metadata["notes"] = ("Experimental bootstrap regions. Coverage is measured, not assumed. "
                         "No formal finite-sample coverage guarantee or solver certification.")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--candidates", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--mismatches", type=float, nargs="+", default=[0., .3])
    parser.add_argument("--shared", type=float, default=.5)
    parser.add_argument("--n-final", type=int, default=50)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--rho", type=float, default=.5)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--out", default="results/pcrort_matrix.csv")
    args = parser.parse_args()
    if args.reps < 1 or args.bootstrap < 10 or not 0 <= args.shared < 1:
        parser.error("reps >= 1, bootstrap >= 10 and 0 <= shared < 1 required")
    print(run(args))


if __name__ == "__main__":
    main()
