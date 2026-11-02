"""Logistic models, FedAvg, probability mixtures and operating thresholds."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from ..method_revision import covariance_adaptive, structured_exchangeable
from ..robust import UncertaintySet, robust_select


def objective(beta, x, y, l2=0.01):
    """Mean log loss plus (l2/2)||beta||^2; the last coordinate is an unpenalized intercept."""
    logits = x @ beta
    penalty = beta.copy()
    penalty[-1] = 0.0
    value = float(np.mean(np.logaddexp(0, logits) - y * logits) + l2 / 2 * (penalty @ penalty))
    gradient = x.T @ (expit(logits) - y) / len(y) + l2 * penalty
    return value, gradient


def fit_logistic(x, y, l2=0.01, max_iterations=1000, tolerance=1e-5):
    result = minimize(objective, np.zeros(x.shape[1]), args=(x, y, l2), jac=True,
                      method="L-BFGS-B",
                      options={"maxiter": max_iterations, "gtol": 1e-8, "ftol": 1e-12})
    residual = float(np.linalg.norm(objective(result.x, x, y, l2)[1], np.inf))
    if residual > tolerance:
        raise RuntimeError(f"logistic fit did not converge (gradient {residual:.2e})")
    return result.x, dict(objective=float(result.fun), gradient_inf=residual,
                          iterations=int(result.nit))


def train_fedavg(x, y, client_id, rounds=100, local_epochs=3, learning_rate=0.1, l2=0.01):
    """Full-participation FedAvg from a zero start with sample-weighted averaging."""
    clients = np.unique(client_id)
    beta = np.zeros(x.shape[1])
    history = []
    for round_index in range(rounds):
        updates = []
        for k in clients:
            xk, yk = x[client_id == k], y[client_id == k]
            local = beta.copy()
            for _ in range(local_epochs):
                local -= learning_rate * objective(local, xk, yk, l2)[1]
            updates.append((local, len(yk)))
        total = sum(n for _, n in updates)
        aggregated = np.zeros_like(beta)
        for local, n in updates:
            aggregated += local * (n / total)
        beta = aggregated
        loss = sum(np.sum(client_id == k) / len(y) * objective(beta, x[client_id == k], y[client_id == k], l2)[0]
                   for k in clients)
        history.append(dict(round=round_index + 1, global_train_loss=loss))
    return beta, history


def mixture_candidates(pilot_predictions, y, folds=5, penalty=1.0):
    """Candidate pilot error second-moment matrices for DIAG, STRUCTURED, FULL and ADAPTIVE.

    Row i of ``pilot_predictions`` holds every client's probability for pilot
    record i; v'Qv is then exactly the pilot Brier score of mixture weights v.
    """
    errors = pilot_predictions - y[:, None]
    outer = np.einsum("ni,nj->nij", errors, errors)
    full = outer.mean(0)
    adaptive = covariance_adaptive(outer, folds, penalty)
    candidates = {"DIAG": np.diag(np.diag(full)), "FULL": full,
                  "STRUCTURED": structured_exchangeable(full), "ADAPTIVE": adaptive.matrix}
    return candidates, adaptive


def select_mixture(q, recipient, external_mass_cap=0.5, radius_fraction=0.03, max_iterations=30):
    """Robust recipient-relative weights over the recipient-preserving simplex."""
    clients = len(q)
    order = [recipient] + [k for k in range(clients) if k != recipient]
    local_q = q[np.ix_(order, order)]
    radius = radius_fraction * np.linalg.norm(local_q, "fro")
    fit = robust_select(UncertaintySet(local_q, "frobenius", float(radius)), external_mass_cap,
                        max_iterations=max_iterations)
    weights = np.empty(clients)
    weights[order] = fit.weights
    if not (np.isfinite(weights).all() and weights.min() >= -1e-8 and abs(weights.sum() - 1) < 1e-7):
        raise RuntimeError("robust selector returned invalid weights")
    return weights, dict(radius=float(radius), numerical_gap=float(fit.gap),
                         converged=bool(fit.converged), iterations=fit.iterations,
                         objective=float(fit.objective))


def specificity_threshold(y, p, target=0.9, min_per_class=5):
    """Lowest threshold t (predict p >= t) whose specificity on (y, p) is at least ``target``."""
    if min(np.bincount(y, minlength=2)) < min_per_class:
        return None
    for threshold in np.r_[np.unique(p), np.nextafter(np.max(p), np.inf)]:
        if np.mean(p[y == 0] < threshold) >= target:
            return float(threshold)
    return None


def classification_metrics(y, p, threshold=0.5):
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("invalid probabilities")
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                sensitivity=float(sensitivity), specificity=float(specificity),
                fn=int(fn), fp=int(fp), tn=int(tn), tp=int(tp),
                brier=float(np.mean((p - y) ** 2)), accuracy=float(np.mean(predicted == y)),
                balanced_accuracy=float((sensitivity + specificity) / 2),
                log_loss=float(-np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe))),
                test_n=len(y), test_positive=int(y.sum()), test_negative=int(len(y) - y.sum()))
