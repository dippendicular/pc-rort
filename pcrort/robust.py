"""General J-candidate PSD robust contrast via constraint generation.

The oracle returns numerical bounds and residuals, not interval-verified
certificates. Any claim of a rigorous strict-improvement certificate requires
a verified solver/dual bound with the residuals accounted for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cvxpy as cp


@dataclass(frozen=True)
class UncertaintySet:
    """PSD Frobenius, PSD entrywise box, or PSD affine ellipsoid in Q space."""
    center: np.ndarray
    kind: str = "frobenius"
    radius: float = 0.
    widths: Optional[np.ndarray] = None
    directions: Optional[np.ndarray] = None
    # For affine ellipsoid, directions shape (r, J, J) and ||u||_2 <= radius.

    def __post_init__(self):
        q = np.asarray(self.center, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.isfinite(q).all():
            raise ValueError("center must be finite square")
        if not np.allclose(q, q.T, atol=1e-8) or np.linalg.eigvalsh(q).min() < -1e-8:
            raise ValueError("center must be symmetric PSD")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("radius must be finite nonnegative")
        if self.kind not in ("frobenius", "box_psd", "affine_psd"):
            raise ValueError("unknown uncertainty kind")
        object.__setattr__(self, "center", (q + q.T) / 2)
        j = len(q)
        if self.kind == "box_psd":
            w = np.asarray(self.widths, dtype=float)
            if w.shape != (j, j) or not np.isfinite(w).all() or np.min(w) < 0:
                raise ValueError("widths must be nonnegative finite J x J")
            if not np.allclose(w, w.T, atol=1e-8):
                raise ValueError("widths must be symmetric")
            object.__setattr__(self, "widths", (w + w.T) / 2)
        if self.kind == "affine_psd":
            d = np.asarray(self.directions, dtype=float)
            if d.ndim != 3 or d.shape[1:] != (j, j) or not np.isfinite(d).all():
                raise ValueError("directions must be finite r x J x J")
            if not np.allclose(d, d.transpose(0, 2, 1), atol=1e-8):
                raise ValueError("directions must be symmetric")
            object.__setattr__(self, "directions", (d + d.transpose(0, 2, 1)) / 2)


@dataclass(frozen=True)
class OracleOutcome:
    worst_case_value: float
    matrix: np.ndarray
    solver_status: str
    # CVXPY primal values are approximate; this is NOT a verified upper bound.
    set_residual: float = 0.
    repair_distance: float = 0.
    certificate_status: str = 'numerical-only'


@dataclass(frozen=True)
class SelectionOutcome:
    weights: np.ndarray
    objective: float
    iterations: int
    converged: bool
    master_lower: float
    oracle_value: float
    gap: float
    uncertainty_kind: str
    certificate_status: str = 'numerical-only'


def _repair_oracle_matrix(matrix: np.ndarray, uncertainty: UncertaintySet) -> np.ndarray:
    """Repair floating-point cuts without discarding the other set constraints.

    This is numerical feasibility repair, not interval arithmetic certification.
    Frobenius/box regions are convex and contain the center. For affine sets,
    work in the original coefficient space and contract toward the center.
    """
    q0=uncertainty.center
    q=(matrix+matrix.T)/2
    if uncertainty.kind=='affine_psd':
        d=uncertainty.directions.reshape(len(uncertainty.directions),-1).T
        u=np.linalg.lstsq(d,(q-q0).ravel(),rcond=None)[0]
        if np.linalg.norm(u)>uncertainty.radius:
            u*=uncertainty.radius/max(np.linalg.norm(u),1e-300)
        direction=(d@u).reshape(q.shape)
        step=1.
        for _ in range(60):
            candidate=q0+step*direction
            if np.linalg.eigvalsh(candidate).min()>=0:return candidate
            step*=.5
        return q0.copy()
    w,v=np.linalg.eigh(q)
    q=(v*np.maximum(w,0))@v.T
    direction=q-q0
    if uncertainty.kind=='frobenius':
        step=min(1.,uncertainty.radius/max(np.linalg.norm(direction,'fro'),1e-300))
    else:
        active=np.abs(direction)>0
        step=min(1.,float(np.min(uncertainty.widths[active]/np.abs(direction[active])))) if active.any() else 1.
    if step<1:step*=1-1e-12
    return q0+step*direction


def _set_residual(q: np.ndarray, uncertainty: UncertaintySet) -> float:
    psd=max(0.,-float(np.linalg.eigvalsh(q).min()))
    delta=q-uncertainty.center
    if uncertainty.kind=='frobenius':other=max(0.,np.linalg.norm(delta,'fro')-uncertainty.radius)
    elif uncertainty.kind=='box_psd':other=max(0.,float(np.max(np.abs(delta)-uncertainty.widths)))
    else:
        d=uncertainty.directions.reshape(len(uncertainty.directions),-1).T
        u=np.linalg.lstsq(d,delta.ravel(),rcond=None)[0]
        other=max(np.linalg.norm(d@u-delta.ravel()),np.linalg.norm(u)-uncertainty.radius,0.)
    return float(max(psd,other))


def _solve(problem: cp.Problem, solver: str, **solver_options) -> str:
    problem.solve(solver=solver, warm_start=True, **solver_options)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Optimization failed: {problem.status}")
    return str(problem.status)


def separation_oracle(
    weights: np.ndarray, uncertainty: UncertaintySet,
    *, solver: str = "CLARABEL", **solver_options
) -> OracleOutcome:
    """Numerically solve max_Q <Q, vv^T-e0e0^T> over a PSD uncertainty set."""
    v = np.asarray(weights, dtype=float)
    q0 = uncertainty.center
    j = q0.shape[0]
    if v.shape != (j,) or np.min(v) < -1e-8 or abs(v.sum() - 1) > 1e-7:
        raise ValueError("weights must lie on the J-candidate simplex")
    g = np.outer(v, v)
    g[0, 0] -= 1
    if uncertainty.kind == "frobenius" and uncertainty.radius == 0:
        return OracleOutcome(float(np.sum(q0 * g)), q0.copy(), "exact-singleton")
    if uncertainty.kind == "affine_psd" and uncertainty.radius == 0:
        return OracleOutcome(float(np.sum(q0 * g)), q0.copy(), "exact-singleton")
    if uncertainty.kind == "box_psd" and np.all(uncertainty.widths == 0):
        return OracleOutcome(float(np.sum(q0 * g)), q0.copy(), "exact-singleton")
    q = cp.Variable((j, j), symmetric=True)
    constraints = [q >> 0]
    if uncertainty.kind == "frobenius":
        constraints += [cp.norm(q - q0, "fro") <= uncertainty.radius]
    elif uncertainty.kind == "box_psd":
        constraints += [cp.abs(q - q0) <= uncertainty.widths]
    else:
        d = uncertainty.directions
        u = cp.Variable(d.shape[0])
        affine_q = q0 + sum(u[i] * d[i] for i in range(len(d)))
        constraints += [q == affine_q, cp.norm(u, 2) <= uncertainty.radius]
    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(g, q))), constraints)
    status = _solve(problem, solver, **solver_options)
    q_sol = np.asarray(q.value, dtype=float)
    # Validate numerical output and never call a solver optimum a verified bound.
    q_sol = (q_sol + q_sol.T) / 2
    eigenvalues = np.linalg.eigvalsh(q_sol)
    if eigenvalues.min() < -2e-5:
        raise RuntimeError("Oracle returned a significantly non-PSD matrix")
    # PSD projection alone can leave the ball, box or affine set.
    original=q_sol.copy()
    q_sol=_repair_oracle_matrix(q_sol,uncertainty)
    residual=_set_residual(q_sol,uncertainty)
    if residual>2e-7:raise RuntimeError('oracle matrix fails uncertainty-set feasibility')
    return OracleOutcome(float(np.sum(g*q_sol)),q_sol,status,residual,
                         float(np.linalg.norm(q_sol-original,'fro')))


def robust_select(
    uncertainty: UncertaintySet, rho: float, *,
    tolerance: float = 1e-5, max_iterations: int = 60,
    solver: str = "CLARABEL", **solver_options
) -> SelectionOutcome:
    """Cutting-plane robust minimization of Eq. 26 for any J >= 1.

    Each stored PSD matrix defines one convex quadratic cut. The termination
    gap is purely numerical; verified interval/dual bounds are separate work.
    """
    j = len(uncertainty.center)
    if not (0 <= rho <= 1) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid mass cap, tolerance or max_iterations")
    ek = np.zeros(j)
    ek[0] = 1
    if j == 1 or rho == 0:
        return SelectionOutcome(ek, 0., 0, True, 0., 0., 0., uncertainty.kind)
    cuts = [uncertainty.center.copy()]
    best_upper = 0.
    best_weights = ek.copy()
    last_lb = -np.inf
    last_oracle = 0.
    for iteration in range(1, max_iterations + 1):
        v = cp.Variable(j)
        t = cp.Variable()
        constraints = [v >= 0, cp.sum(v) == 1, v[0] >= 1 - rho]
        for q in cuts:
            constraints.append(t >= cp.quad_form(v, cp.psd_wrap(q)) - q[0, 0])
        master = cp.Problem(cp.Minimize(t), constraints)
        _solve(master, solver, **solver_options)
        v_candidate = np.maximum(np.asarray(v.value, dtype=float), 0.)
        v_candidate /= v_candidate.sum()
        lower = float(master.value)
        oracle = separation_oracle(v_candidate, uncertainty, solver=solver, **solver_options)
        last_lb, last_oracle = lower, oracle.worst_case_value
        if oracle.worst_case_value < best_upper:
            best_upper = oracle.worst_case_value
            best_weights = v_candidate.copy()
        gap = max(0., best_upper - lower)
        if gap <= tolerance:
            return SelectionOutcome(best_weights, best_upper, iteration, True,
                                    lower, oracle.worst_case_value, gap, uncertainty.kind)
        cuts.append(oracle.matrix)
    return SelectionOutcome(best_weights, best_upper, max_iterations, False,
                            last_lb, last_oracle, max(0., best_upper - last_lb),
                            uncertainty.kind)
