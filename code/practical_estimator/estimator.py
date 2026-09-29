"""Noise-aware all-candidate-edge unknown-topology fit; separate from the exact inverse."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares


SUPPORT_THRESHOLD = 0.25  # Frozen in normalized stiffness units before the test suite.


@dataclass
class FitResult:
    L: np.ndarray
    C: np.ndarray
    edge_weights: np.ndarray
    candidate_edges: list[tuple[int, int]]
    success: bool
    status: int
    message: str
    normalized_residual: float
    relative_residual: float
    nfev: int
    jacobian_condition: float


def all_candidate_edges(n: int) -> list[tuple[int, int]]:
    return list(combinations(range(n), 2))


def descriptor_from_parameters(n: int, candidate_edges: list[tuple[int, int]], theta: np.ndarray):
    m = len(candidate_edges)
    grounding = np.exp(theta[:n])
    weights = np.asarray(theta[n:n + m], dtype=float)
    storage = np.exp(theta[n + m:n + m + n])
    incidence = np.zeros((n, m), dtype=float)
    for k, (u, v) in enumerate(candidate_edges):
        incidence[u, k] = -1.0
        incidence[v, k] = 1.0
    L = np.diag(grounding) + incidence @ np.diag(weights) @ incidence.T
    return L, np.diag(storage), weights


def threshold_schur(L: np.ndarray, C: np.ndarray, growth_rate: float, q: np.ndarray,
                    retained: tuple[int, ...] | None, root: int) -> float:
    """Evaluate a measured root threshold without querying true topology."""
    n = len(L)
    retained = tuple(range(n)) if retained is None else tuple(retained)
    B = L + growth_rate * C + np.diag(q)
    block = B[np.ix_(retained, retained)]
    ri = retained.index(root)
    hidden = [k for k in range(len(retained)) if k != ri]
    if not hidden:
        return float(block[ri, ri])
    kblock = block[np.ix_(hidden, hidden)]
    b = -block[hidden, ri]
    return float(block[ri, ri] - b @ np.linalg.solve(kblock, b))


def threshold_finite_clamp(L: np.ndarray, C: np.ndarray, growth_rate: float, q: np.ndarray,
                           retained: tuple[int, ...] | None, root: int, penalty: float) -> float:
    """Finite-penalty approximation used only for intervention sensitivity tests."""
    n = len(L)
    retained = tuple(range(n)) if retained is None else tuple(retained)
    omitted = [i for i in range(n) if i not in retained]
    q_finite = np.asarray(q, dtype=float).copy()
    q_finite[omitted] += penalty
    return threshold_schur(L, C, growth_rate, q_finite, None, root)


def fit_all_candidate_edges(n: int, root: int, records: list[dict], *, max_nfev: int = 1200) -> FitResult:
    """Fit positive groundings/storage and nonnegative weights on all n choose 2 edges.

    Inputs are only n, the predeclared accessible root, and recorded queries.
    No true edge support or parameter is accepted by this estimator.
    """
    candidate_edges = all_candidate_edges(n)
    m = len(candidate_edges)
    yobs = np.asarray([float(row["observed_threshold"]) for row in records], dtype=float)
    scales = np.maximum(np.abs(yobs), max(0.02 * float(np.median(np.abs(yobs))), 1e-10))
    lambdas = [float(row["growth_rate"]) for row in records]
    qs = [np.asarray(row["q"], dtype=float) for row in records]
    retained = [row["retained"] for row in records]

    def predict(theta):
        L, C, weights = descriptor_from_parameters(n, candidate_edges, theta)
        values = np.array([threshold_schur(L, C, lam, q, keep, root)
                           for lam, q, keep in zip(lambdas, qs, retained)])
        return values, L, C, weights

    def residual(theta):
        values, _, _, _ = predict(theta)
        return (values - yobs) / scales

    x0 = np.r_[np.full(n, np.log(0.8)), np.full(m, 0.6), np.full(n, np.log(1.0))]
    lower = np.r_[np.full(n, np.log(0.05)), np.zeros(m), np.full(n, np.log(0.05))]
    upper = np.r_[np.full(n, np.log(10.0)), np.full(m, 20.0), np.full(n, np.log(10.0))]
    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev,
                           xtol=1e-9, ftol=1e-9, gtol=1e-9, x_scale="jac")
    yhat, Lhat, Chat, weights = predict(result.x)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    positive = singular[singular > np.finfo(float).eps * max(singular[0], 1.0)] if len(singular) else np.array([])
    condition = float(positive[0] / positive[-1]) if len(positive) else float("inf")
    return FitResult(
        L=Lhat, C=Chat, edge_weights=weights, candidate_edges=candidate_edges,
        success=bool(result.success), status=int(result.status), message=str(result.message),
        normalized_residual=float(np.linalg.norm(result.fun)),
        relative_residual=float(np.linalg.norm(yhat - yobs) / max(np.linalg.norm(yobs), 1e-14)),
        nfev=int(result.nfev), jacobian_condition=condition,
    )


def inferred_support(fit: FitResult, threshold: float = SUPPORT_THRESHOLD) -> set[tuple[int, int]]:
    return {edge for edge, weight in zip(fit.candidate_edges, fit.edge_weights) if weight >= threshold}
