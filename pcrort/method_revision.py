"""Pilot-only methodological extensions for PC-RORT.

The routines here preserve the recipient-relative contrast and local vertex.
They do not replace the manuscript's joint-gamma confidence construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _psd(a: np.ndarray) -> np.ndarray:
    a = (np.asarray(a, float) + np.asarray(a, float).T) / 2
    d, u = np.linalg.eigh(a)
    return (u * np.maximum(d, 0.0)) @ u.T


def structured_exchangeable(q: np.ndarray) -> np.ndarray:
    """PSD exchangeable-correlation projection preserving marginal scales."""
    q = _psd(q)
    s = np.sqrt(np.maximum(np.diag(q), 1e-12))
    r = q / np.outer(s, s)
    j = len(q)
    off = r[np.triu_indices(j, 1)]
    rho = float(np.clip(off.mean() if len(off) else 0.0,
                        -1.0 / max(j - 1, 1) + 1e-8, 1.0))
    corr = (1 - rho) * np.eye(j) + rho * np.ones((j, j))
    return _psd(np.outer(s, s) * corr)


@dataclass(frozen=True)
class AdaptiveCovariance:
    matrix: np.ndarray
    weights: np.ndarray
    cv_losses: np.ndarray
    loss_se: np.ndarray
    labels: tuple[str, ...] = ("DIAG", "STRUCTURED", "FULL")


def covariance_adaptive(outer_products: np.ndarray, folds: int = 5,
                        penalty: float = 1.0) -> AdaptiveCovariance:
    """Cross-fitted exponential aggregation using pilot observations only.

    ``outer_products[i]`` is a symmetric PSD contribution whose expectation is
    the target second-moment matrix.  Each validation loss predicts a fresh
    contribution, so no final-fit or test observation is used.
    """
    z = np.asarray(outer_products, float)
    if z.ndim != 3 or z.shape[1] != z.shape[2] or len(z) < 6:
        raise ValueError("outer_products must have shape (n>=6,J,J)")
    folds = min(max(2, int(folds)), len(z) // 2)
    losses = np.zeros((len(z), 3))
    indices = np.arange(len(z))
    for f in range(folds):
        va = indices[indices % folds == f]
        tr = indices[indices % folds != f]
        full = _psd(z[tr].mean(axis=0))
        candidates = (np.diag(np.diag(full)), structured_exchangeable(full), full)
        for k, q in enumerate(candidates):
            losses[va, k] = np.sum((z[va] - q) ** 2, axis=(1, 2))
    mean = losses.mean(axis=0)
    se = losses.std(axis=0, ddof=1) / math.sqrt(len(z))
    scale = max(float(np.median(se[se > 1e-12])) if np.any(se > 1e-12) else 0.0,
                1e-10)
    score = mean + float(penalty) * se
    weights = np.exp(-(score - score.min()) / scale)
    weights /= weights.sum()
    full = _psd(z.mean(axis=0))
    candidates = (np.diag(np.diag(full)), structured_exchangeable(full), full)
    matrix = _psd(sum(w * q for w, q in zip(weights, candidates)))
    return AdaptiveCovariance(matrix, weights, mean, se)


def contrast_error_bound(q_est: np.ndarray, q_true: np.ndarray) -> float:
    """Uniform deterministic |estimated contrast - true contrast| bound."""
    return math.sqrt(2.0) * float(np.linalg.norm(q_est - q_true, "fro"))


def simplex_net(j: int, rho: float, denominator: int) -> np.ndarray:
    """Finite rational net of the simplex with local coordinate zero."""
    if j < 1 or not (0 <= rho <= 1) or denominator < 1:
        raise ValueError("invalid simplex-net arguments")
    out = []
    def rec(prefix, remaining, slots):
        if slots == 1:
            counts = prefix + [remaining]
            if sum(counts[1:]) <= rho * denominator + 1e-12:
                out.append(np.asarray(counts, float) / denominator)
            return
        for x in range(remaining + 1):
            rec(prefix + [x], remaining - x, slots - 1)
    rec([], denominator, j)
    return np.asarray(out)


@dataclass(frozen=True)
class ContrastSelection:
    weights: np.ndarray
    empirical_contrast: float
    upper_bound: float
    variance: float
    net_size: int
    epsilon_l1: float


def contrast_net_select(contributions: np.ndarray, rho: float = 0.5,
                        delta: float = 0.05, denominator: int = 40,
                        bound: float | None = None) -> ContrastSelection:
    """Uniform finite-net empirical-Bernstein selection.

    Assumes independent pilot contributions Z_i that are PSD and satisfy
    ||Z_i||_op <= bound almost surely.  The selected upper confidence bound is
    simultaneous over the finite net; ``4*bound*epsilon_l1`` extends it to the
    continuous feasible simplex.  Selection returns LOCAL unless the best
    upper bound is strictly negative.
    """
    z = np.asarray(contributions, float)
    n, j, j2 = z.shape
    if n < 3 or j != j2 or not (0 < delta < 1):
        raise ValueError("invalid contributions or delta")
    observed = np.linalg.eigvalsh((z + z.transpose(0, 2, 1)) / 2)
    if observed.min() < -1e-8:
        raise ValueError("contributions must be PSD")
    m = float(bound) if bound is not None else float(np.max(observed))
    if m <= 0 or np.max(observed) > m + 1e-8:
        raise ValueError("bound must dominate every contribution operator norm")
    net = simplex_net(j, rho, denominator)
    # Floor the first j-1 coordinates and put the remainder in the last one:
    # the resulting l1 error is at most 2(j-1)/denominator.  The mass-cap
    # boundary is on the grid in the executed design (rho*denominator integer).
    eps = 2 * (j - 1) / denominator
    logterm = math.log(2 * len(net) / delta)
    values = np.einsum("ki,nij,kj->nk", net, z, net) - z[:, 0, 0, None]
    means = values.mean(axis=0)
    variances = values.var(axis=0, ddof=1)
    uppers = (means + np.sqrt(2 * variances * logterm / n)
              + 14 * m * logterm / (3 * (n - 1)) + 4 * m * eps)
    index = int(np.argmin(uppers))
    best = ContrastSelection(net[index], float(means[index]),
                             float(uppers[index]), float(variances[index]),
                             len(net), eps)
    local = np.zeros(j); local[0] = 1.0
    if best is None or best.upper_bound >= 0:
        return ContrastSelection(local, 0.0, 0.0, 0.0, len(net), eps)
    return best


def empirical_bernstein_gate(values: np.ndarray, delta: float = 0.05,
                             loss_bound: float = 1.0) -> tuple[float, float]:
    """Manuscript Eq. (29): return sample mean and upper bound."""
    x = np.asarray(values, float)
    if len(x) <= 1 or np.max(np.abs(x)) > loss_bound + 1e-12:
        raise ValueError("invalid bounded paired losses")
    logterm = math.log(2 / delta)
    upper = (x.mean() + math.sqrt(2 * x.var(ddof=1) * logterm / len(x))
             + 14 * loss_bound * logterm / (3 * (len(x) - 1)))
    return float(x.mean()), float(upper)


def gate_population_sufficient_gap(n: int, delta: float = 0.05,
                                   eta: float = 0.05,
                                   loss_bound: float = 1.0) -> float:
    """Conservative gap sufficient for gate acceptance with probability 1-eta.

    It combines a Hoeffding bound for the sample mean with the deterministic
    bound s^2 <= n L^2/(n-1).  The condition is sufficient, not necessary.
    """
    if n <= 1:
        return math.inf
    l = float(loss_bound)
    mean_dev = l * math.sqrt(2 * math.log(1 / eta) / n)
    gate_var = l * math.sqrt(2 * n * math.log(2 / delta) / (n - 1) / n)
    correction = 14 * l * math.log(2 / delta) / (3 * (n - 1))
    return mean_dev + gate_var + correction
