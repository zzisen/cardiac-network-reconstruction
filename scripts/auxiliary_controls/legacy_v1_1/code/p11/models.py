from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaling import L_to_J

from .errors import InputContractError, TheoremBoundaryFailure


@dataclass(frozen=True)
class PathSystem:
    """Synthetic serial myofibril in normalized small-signal coordinates."""

    C: np.ndarray
    L: np.ndarray
    local_tangent: np.ndarray
    coupling: np.ndarray
    label: str

    @property
    def n(self) -> int:
        return int(self.C.shape[0])


def incidence_path(n: int) -> np.ndarray:
    if n < 2:
        raise InputContractError("A path benchmark requires n >= 2.")
    B = np.zeros((n, n - 1), dtype=float)
    for edge in range(n - 1):
        B[edge, edge] = -1.0
        B[edge + 1, edge] = 1.0
    return B


def make_path_system(
    n: int,
    seed: int,
    *,
    near_ill_conditioned: bool = False,
    label: str | None = None,
) -> PathSystem:
    """Create a heterogeneous, stable path with explicitly synthetic scales.

    The Washio mechanics motivate positive local transverse tangent stiffness,
    positive friction, and positive nearest-neighbour lattice-alignment terms.
    The numerical values here are dimensionless synthetic stress-test values;
    they are not estimates fitted to human cells.
    """
    if n < 2:
        raise InputContractError("A path benchmark requires n >= 2.")
    rng = np.random.default_rng(seed)
    if near_ill_conditioned:
        local = np.linspace(0.012, 0.022, n)
        coupling = np.linspace(0.65, 1.15, n - 1)
        storage = np.geomspace(0.65, 1.6, n)
        tag = label or "near_ill_conditioned"
    else:
        local = rng.uniform(0.10, 0.32, n)
        coupling = rng.uniform(0.12, 0.72, n - 1)
        storage = rng.uniform(0.70, 1.45, n)
        tag = label or "heterogeneous_path"
    B = incidence_path(n)
    L = np.diag(local) + B @ np.diag(coupling) @ B.T
    C = np.diag(storage)
    system = PathSystem(C=C, L=L, local_tangent=local, coupling=coupling, label=tag)
    validate_path_system(system)
    return system


def make_star_J(n: int, *, hidden_twin: bool = False) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Small exact-class K8 controls, returned in J coordinates."""
    if n < 2:
        raise InputContractError("A star benchmark requires n >= 2.")
    diag = -np.linspace(3.0, 3.0 + n - 1, n)
    if hidden_twin and n >= 4:
        diag[2] = diag[1]
    J = np.diag(diag)
    edges: list[tuple[int, int]] = []
    for v in range(1, n):
        b = 0.16 + 0.035 * v
        J[0, v] = J[v, 0] = b
        edges.append((0, v))
    if np.max(np.linalg.eigvalsh(J)) >= 0:
        raise TheoremBoundaryFailure("K8 fixture J must be negative definite.")
    return J, edges


def validate_path_system(system: PathSystem, *, strict: bool = True) -> None:
    C, L = np.asarray(system.C), np.asarray(system.L)
    n = system.n
    if C.shape != (n, n) or L.shape != (n, n):
        raise InputContractError("C and L must be square matrices of equal size.")
    if not np.allclose(C, C.T, atol=1e-12) or np.any(np.diag(C) <= 0):
        raise TheoremBoundaryFailure("C must be positive diagonal storage/friction.")
    if not np.allclose(C, np.diag(np.diag(C)), atol=1e-12):
        raise TheoremBoundaryFailure("The declared descriptor contract requires diagonal C.")
    if not np.allclose(L, L.T, atol=1e-12):
        raise TheoremBoundaryFailure("L must be symmetric reciprocal stiffness.")
    if np.min(np.linalg.eigvalsh(L)) <= 0:
        raise TheoremBoundaryFailure("L must be positive definite at the operating point.")
    off = L.copy()
    np.fill_diagonal(off, 0.0)
    if np.any(off > 1e-12):
        raise TheoremBoundaryFailure("L must have non-positive off-diagonal entries.")
    expected = {(i, i + 1) for i in range(n - 1)}
    observed = {(i, j) for i in range(n) for j in range(i + 1, n) if L[i, j] < -1e-12}
    if strict and observed != expected:
        raise TheoremBoundaryFailure("The K8 cardiac family requires a connected path graph.")


def normalized_J(system: PathSystem) -> np.ndarray:
    """Return the K8 generator J=-C^-1/2 L C^-1/2."""
    validate_path_system(system)
    return L_to_J(system.L, system.C)


def normalized_condition(system: PathSystem) -> float:
    return float(np.linalg.cond(-L_to_J(system.L, system.C)))
