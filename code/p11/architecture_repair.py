from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .known_tree import KnownTreeObservations
from .scaling import L_to_J


@dataclass(frozen=True)
class YTreeFixture:
    J: np.ndarray
    edges: list[tuple[int, int]]
    local_tangent: np.ndarray
    coupling: np.ndarray
    C: np.ndarray


@dataclass(frozen=True)
class CyclicFixture:
    L: np.ndarray
    C: np.ndarray
    edges: list[tuple[int, int]]
    local_tangent: np.ndarray
    coupling: np.ndarray


def incidence_matrix(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    B = np.zeros((n, len(edges)), dtype=float)
    for j, (u, v) in enumerate(edges):
        B[u, j] = -1.0
        B[v, j] = 1.0
    return B


def make_y_tree_fixture() -> YTreeFixture:
    """Four labelled nodes: root 0, junction 1, daughters 2 and 3."""
    edges = [(0, 1), (1, 2), (1, 3)]
    local = np.array([0.72, 0.91, 0.83, 1.07])
    coupling = np.array([0.68, 0.49, 0.31])
    C = np.diag([0.8, 1.0, 1.2, 0.9])
    B = incidence_matrix(4, edges)
    L = np.diag(local) + B @ np.diag(coupling) @ B.T
    J = L_to_J(L, C)
    return YTreeFixture(J, edges, local, coupling, C)


def make_cyclic_network_fixture() -> CyclicFixture:
    """Five-node connected graph with a 1-3-4-1 cycle and two leaves."""
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 1)]
    local = np.array([0.62, 0.78, 0.69, 0.91, 0.73])
    coupling = np.array([0.34, 0.29, 0.25, 0.38, 0.21])
    C = np.diag([0.74, 1.12, 0.91, 1.27, 0.83])
    B = incidence_matrix(5, edges)
    L = np.diag(local) + B @ np.diag(coupling) @ B.T
    return CyclicFixture(L, C, edges, local, coupling)


def permute_y_daughters(J: np.ndarray) -> np.ndarray:
    """Swap labelled daughter states 2 and 3 while leaving root/junction fixed."""
    permutation = np.eye(4)[[0, 1, 3, 2]]
    return permutation @ np.asarray(J, dtype=float) @ permutation.T


def local_edge_parameters(L: np.ndarray, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Read local groundings and positive edge weights from a tree-form L."""
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    coupling = np.array([-L[u, v] for u, v in edges], dtype=float)
    degree_weight = np.zeros(n, dtype=float)
    for (u, v), value in zip(edges, coupling):
        degree_weight[u] += value
        degree_weight[v] += value
    local = np.diag(L) - degree_weight
    return local, coupling


def fit_known_tree(
    observations: KnownTreeObservations,
    C: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    root: int = 0,
    max_nfev: int = 3000,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Exploratory positive-parameter least-squares fit for a known tree.

    This is a numerical noise probe, not part of the exact known-tree theorem.
    Unknowns are the positive local tangents and edge stiffnesses in L.
    """
    from scipy.optimize import least_squares

    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    w = np.asarray(observations.frequencies, dtype=float)
    P = np.asarray(observations.profiles, dtype=float)
    h_obs = np.asarray(observations.coherent_root, dtype=complex)
    y_obs = np.asarray(observations.psd_differences, dtype=float)
    B = incidence_matrix(n, edges)
    scales_h = np.maximum(1e-3, np.abs(h_obs))
    scales_y = np.maximum(1e-5, np.abs(y_obs))

    def predict(log_parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local = np.exp(log_parameters[:n])
        coupling = np.exp(log_parameters[n:])
        L = np.diag(local) + B @ np.diag(coupling) @ B.T
        J = L_to_J(L, C)
        h = np.empty(n, dtype=complex)
        y = np.empty((n - 2, n), dtype=float)
        for k, omega in enumerate(w):
            G = np.linalg.inv(1j * omega * np.eye(n) - J)
            h[k] = G[root, root]
            y[:, k] = P @ (np.abs(G[root, :]) ** 2)
        return J, h, y

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        _, h, y = predict(log_parameters)
        rh = (h - h_obs) / scales_h
        ry = (y - y_obs) / scales_y
        return np.concatenate((rh.real, rh.imag, ry.ravel()))

    x0 = np.log(np.concatenate((np.full(n, 0.8), np.full(len(edges), 0.5))))
    result = least_squares(residual, x0, max_nfev=max_nfev, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    Jhat, _, _ = predict(result.x)
    diagnostics: dict[str, float | int | bool | str] = {
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "weighted_residual_norm": float(np.linalg.norm(result.fun)),
        "evaluations": int(result.nfev),
    }
    return Jhat, diagnostics


def recover_jacobi_from_spectral_measure(eigenvalues: np.ndarray, root_weights: np.ndarray) -> np.ndarray:
    """Rebuild an endpoint Jacobi matrix from its exact root spectral measure.

    The positive Lanczos subdiagonals fix the sign convention. This routine
    documents the path-only coherent-transfer control; branching trees do not
    share this scalar-measure reconstruction property.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(root_weights, dtype=float)
    if eigenvalues.ndim != 1 or weights.shape != eigenvalues.shape or len(weights) < 2:
        raise ValueError("Need matching one-dimensional eigenvalue and root-weight arrays.")
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(weights)):
        raise ValueError("Spectral data must be finite.")
    if np.any(weights <= 0) or not np.isclose(np.sum(weights), 1.0, atol=1e-10):
        raise ValueError("An irreducible Jacobi endpoint measure has positive unit-sum weights.")
    if len(np.unique(eigenvalues)) != len(eigenvalues):
        raise ValueError("An irreducible Jacobi matrix has simple spectrum.")
    n = len(weights)
    diagonal = np.zeros(n)
    off_diagonal = np.zeros(n - 1)
    q_prev = np.zeros(n)
    q = np.sqrt(weights)
    beta_prev = 0.0
    for j in range(n):
        v = eigenvalues * q - beta_prev * q_prev
        alpha = float(q @ v)
        v = v - alpha * q
        # Full re-orthogonalization preserves numerical accuracy for this small
        # diagnostic; it does not add data beyond the spectral measure.
        basis = [q_prev, q]
        for basis_vector in basis:
            v -= (basis_vector @ v) * basis_vector
        diagonal[j] = alpha
        if j < n - 1:
            beta = float(np.linalg.norm(v))
            if beta <= 1e-12:
                raise ValueError("The spectral measure is not cyclic at the root.")
            off_diagonal[j] = beta
            q_prev, q = q, v / beta
            beta_prev = beta
    J = np.diag(diagonal)
    J[np.arange(n - 1), np.arange(1, n)] = off_diagonal
    J[np.arange(1, n), np.arange(n - 1)] = off_diagonal
    return J
