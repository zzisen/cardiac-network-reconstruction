from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np

from .errors import (
    InputContractError,
    PrecisionFailure,
    TheoremBoundaryFailure,
    UnidentifiableNode,
)

ThresholdFn = Callable[[float, np.ndarray, tuple[int, ...] | None], float]


@lru_cache(maxsize=512)
def _validate_class_matrices(n: int, l_bytes: bytes, c_bytes: bytes) -> None:
    """Validate an immutable matrix value once across repeated oracle calls."""
    L = np.frombuffer(l_bytes, dtype=np.float64).reshape(n, n)
    C = np.frombuffer(c_bytes, dtype=np.float64).reshape(n, n)
    if not np.allclose(L, L.T, rtol=1e-12, atol=1e-14) or not np.allclose(C, C.T, rtol=1e-12, atol=1e-14):
        raise TheoremBoundaryFailure("unknown-topology requires reciprocal symmetric L and C.")
    if np.any(np.diag(C) <= 0) or not np.allclose(C, np.diag(np.diag(C)), rtol=0.0, atol=1e-14):
        raise TheoremBoundaryFailure("unknown-topology requires a positive diagonal storage matrix C.")
    if np.any(L - np.diag(np.diag(L)) > 0) or np.min(np.linalg.eigvalsh(L)) <= 0:
        raise TheoremBoundaryFailure("unknown-topology requires an SPD M-matrix L.")
    reached = {0}
    frontier = [0]
    while frontier:
        u = frontier.pop()
        for v in range(n):
            if v not in reached and L[u, v] < 0:
                reached.add(v)
                frontier.append(v)
    if len(reached) != n:
        raise TheoremBoundaryFailure("unknown-topology requires an irreducible (connected) M-matrix L.")


@dataclass(frozen=True)
class UnknownTopologyBRecovery:
    B: np.ndarray
    query_count: int


@dataclass(frozen=True)
class UnknownTopologyJointRecovery:
    L: np.ndarray
    C: np.ndarray
    query_count: int
    growth_rate_storage: float


def threshold_from_descriptor(
    L: np.ndarray,
    C: np.ndarray,
    growth_rate: float,
    q: np.ndarray,
    retained_nodes: Iterable[int] | None = None,
    *,
    root: int = 0,
) -> float:
    """Ideal unknown-topology threshold oracle for a reciprocal descriptor network.

    Clamps retain the principal L_PP and C_PP matrices. Diagonal stiffness
    contributions from edges to clamped nodes remain in L_PP, as required by
    the ideal zero-state/Dirichlet intervention contract.
    """
    L = np.asarray(L, dtype=float)
    C = np.asarray(C, dtype=float)
    q = np.asarray(q, dtype=float)
    n = L.shape[0]
    if L.shape != (n, n) or C.shape != (n, n) or q.shape != (n,):
        raise InputContractError("unknown-topology oracle requires equal n-by-n L,C and length-n q.")
    if n < 2 or not np.isfinite(growth_rate) or growth_rate < 0 or not np.all(np.isfinite(L)) or not np.all(np.isfinite(C)) or not np.all(np.isfinite(q)):
        raise InputContractError("unknown-topology matrices and growth level must be finite, with n>=2 and lambda>=0.")
    if np.any(q < 0):
        raise InputContractError("unknown-topology growth levels and calibrated shunts must be nonnegative.")
    if not 0 <= root < n:
        raise InputContractError("The unknown-topology root must be a valid node label.")
    if q[root] != 0:
        raise InputContractError("Calibrated unknown-topology shunts are allowed only on hidden nodes; root loading is feedback.")
    L = np.ascontiguousarray(L, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    _validate_class_matrices(n, L.tobytes(), C.tobytes())
    retained = tuple(range(n)) if retained_nodes is None else tuple(int(i) for i in retained_nodes)
    if not retained or root not in retained or len(set(retained)) != len(retained):
        raise InputContractError("The retained clamp set must be unique and include the root.")
    if any(i < 0 or i >= n for i in retained):
        raise InputContractError("A retained unknown-topology node label is out of range.")
    B = (L + growth_rate * C)[np.ix_(retained, retained)].copy()
    B += np.diag(q[list(retained)])
    if len(retained) == 1:
        return float(B[retained.index(root), retained.index(root)])
    ri = retained.index(root)
    other = [i for i in range(len(retained)) if i != ri]
    K = B[np.ix_(other, other)]
    b = -B[other, ri]
    if np.min(np.linalg.eigvalsh(K)) <= 0:
        raise TheoremBoundaryFailure("The hidden unknown-topology block must remain positive definite.")
    return float(B[ri, ri] - b @ np.linalg.solve(K, b))


class CountingThresholdOracle:
    """Wrap a threshold function and expose the exact scalar-query count."""

    def __init__(self, fn: ThresholdFn):
        self.fn = fn
        self.query_count = 0

    def __call__(self, growth_rate: float, q: np.ndarray, retained_nodes=None) -> float:
        self.query_count += 1
        return float(self.fn(growth_rate, np.asarray(q, dtype=float), retained_nodes))


def _check_recovered_B(B: np.ndarray, *, root: int = 0, tol: float | None = None) -> None:
    n = B.shape[0]
    tol = 2e-7 * max(1.0, float(np.linalg.norm(B, ord=np.inf))) if tol is None else tol
    if not np.all(np.isfinite(B)) or not np.allclose(B, B.T, atol=tol):
        raise PrecisionFailure("Recovered unknown-topology descriptor block is nonfinite or asymmetric.")
    if np.min(np.linalg.eigvalsh(B)) <= 0:
        raise PrecisionFailure("Recovered unknown-topology descriptor block is not positive definite.")
    off = B.copy()
    np.fill_diagonal(off, 0.0)
    if np.any(off > tol):
        raise PrecisionFailure("Recovered unknown-topology block violates the M-matrix sign condition.")
    unseen = set(range(n))
    reached = {root}
    weak_edges = False
    while reached:
        u = reached.pop()
        unseen.discard(u)
        for v in list(unseen):
            if B[u, v] < -tol:
                reached.add(v)
            elif B[u, v] != 0.0:
                weak_edges = True
    if unseen:
        if weak_edges:
            raise PrecisionFailure("Recovered unknown-topology topology has edge signals below double-precision resolution.")
        raise UnidentifiableNode(f"Recovered unknown-topology network has nodes not connected to root: {sorted(unseen)}")


def recover_B_from_thresholds(
    n: int,
    oracle: ThresholdFn,
    growth_rate: float,
    *,
    root: int = 0,
    load: float = 1.0,
) -> UnknownTopologyBRecovery:
    """unknown-topology-T1 finite-load protocol for one B_lambda using ideal thresholds."""
    if n < 2 or not np.isfinite(load) or load <= 0 or not np.isfinite(growth_rate) or growth_rate < 0:
        raise InputContractError("unknown-topology-T1 requires n>=2, positive load, and lambda>=0.")
    if not 0 <= root < n:
        raise InputContractError("Invalid unknown-topology root label.")
    hidden = [i for i in range(n) if i != root]
    zero = np.zeros(n, dtype=float)
    T0 = float(oracle(growth_rate, zero, None))
    m = n - 1
    G = np.zeros((m, m), dtype=float)
    v = np.zeros(m, dtype=float)
    for i, node in enumerate(hidden):
        q1 = zero.copy()
        q2 = zero.copy()
        q1[node], q2[node] = load, 2.0 * load
        T1 = float(oracle(growth_rate, q1, None))
        T2 = float(oracle(growth_rate, q2, None))
        r1, r2 = T1 - T0, T2 - T0
        gap = r2 - r1
        resolution = 64 * np.finfo(float).eps * max(abs(T0), abs(T1), abs(T2), np.finfo(float).tiny)
        if r1 < 0 or gap < 0:
            raise PrecisionFailure(f"unknown-topology axis thresholds are inconsistent at hidden node {node}.")
        if r1 <= resolution or gap <= resolution:
            raise PrecisionFailure(f"unknown-topology axis response for hidden node {node} is below double-precision resolution.")
        G[i, i] = (2.0 * r1 - r2) / (2.0 * load * gap)
        vi2 = r1 * (1.0 + load * G[i, i]) / load
        if G[i, i] <= 0 or vi2 <= 0:
            raise PrecisionFailure(f"unknown-topology axis inversion produced a nonpositive Green entry at node {node}.")
        v[i] = np.sqrt(vi2)

    for i in range(m):
        for j in range(i + 1, m):
            gi, gj = G[i, i], G[j, j]
            c = v[i] * v[j]
            U = v[i] ** 2 + v[j] ** 2
            eps = c / (8.0 * (gi + gj) * U)
            if not np.isfinite(eps) or eps <= 0:
                raise PrecisionFailure("unknown-topology pair-load scale is not representable at the current precision.")
            q = zero.copy()
            q[hidden[i]], q[hidden[j]] = eps, eps
            measured = float(oracle(growth_rate, q, None)) - T0
            A = eps * v[i] ** 2 * (1.0 + eps * gj) + eps * v[j] ** 2 * (1.0 + eps * gi)
            B = (1.0 + eps * gi) * (1.0 + eps * gj)

            def response(x: float) -> float:
                return (A - 2.0 * eps**2 * c * x) / (B - eps**2 * x**2)

            upper = np.nextafter(np.sqrt(gi * gj), 0.0)
            y0, y1 = response(0.0), response(upper)
            response_tolerance = 256 * np.finfo(float).eps * max(abs(y0), abs(y1), abs(y0 - y1), np.finfo(float).tiny)
            if measured > y0 + response_tolerance or measured < y1 - response_tolerance:
                raise PrecisionFailure("unknown-topology paired threshold is outside its unique admissible Green interval.")
            measured = min(y0, max(y1, measured))
            lo, hi = 0.0, upper
            for _ in range(90):
                mid = (lo + hi) / 2.0
                if response(mid) > measured:
                    lo = mid
                else:
                    hi = mid
            estimate = (lo + hi) / 2.0
            if abs(response(estimate) - measured) > 8 * response_tolerance:
                raise PrecisionFailure("unknown-topology pair inversion did not resolve its threshold residual at double precision.")
            G[i, j] = G[j, i] = estimate
    if np.min(np.linalg.eigvalsh(G)) <= 0 or np.any(G < -1e-9):
        raise PrecisionFailure("unknown-topology reconstructed Green matrix is not entrywise nonnegative positive definite.")
    try:
        K = np.linalg.inv(G)
    except np.linalg.LinAlgError as exc:
        raise PrecisionFailure("unknown-topology Green matrix is numerically singular.") from exc
    b_hidden = K @ v
    if np.any(b_hidden < -2e-7):
        raise PrecisionFailure("unknown-topology reconstructed root-coupling vector has a negative entry.")
    a = T0 + b_hidden @ v
    Bmat = np.zeros((n, n), dtype=float)
    Bmat[root, root] = a
    Bmat[np.ix_(hidden, hidden)] = K
    Bmat[root, hidden] = -b_hidden
    Bmat[hidden, root] = -b_hidden
    _check_recovered_B(Bmat, root=root)
    return UnknownTopologyBRecovery(Bmat, 1 + 2 * m + m * (m - 1) // 2)


def _graph_paths(L: np.ndarray, root: int, tol: float | None = None) -> tuple[dict[int, int], dict[int, int]]:
    n = len(L)
    tol = 2e-7 * max(1.0, float(np.linalg.norm(L, ord=np.inf))) if tol is None else tol
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if L[i, j] < -tol:
                adj[i].append(j)
                adj[j].append(i)
    dist = {root: 0}
    parent = {root: -1}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    if len(dist) != n:
        weak_edges = any(0.0 < abs(L[i, j]) <= tol for i in range(n) for j in range(i + 1, n))
        if weak_edges:
            raise PrecisionFailure("Recovered unknown-topology topology contains an edge below double-precision resolution.")
        raise UnidentifiableNode("unknown-topology recovered L contains a component disconnected from the observed root.")
    return dist, parent


def _path_to(node: int, parent: dict[int, int]) -> tuple[int, ...]:
    path = [node]
    while parent[path[-1]] != -1:
        path.append(parent[path[-1]])
    return tuple(reversed(path))


def recover_unknown_topology(
    n: int,
    oracle: ThresholdFn,
    *,
    root: int = 0,
    growth_rate_storage: float = 1.0,
    load: float = 1.0,
) -> UnknownTopologyJointRecovery:
    """unknown-topology-T3 sharp joint recovery of L and diagonal C.

    The first d=n(n+1)/2 queries recover L at lambda=0. Then n calibrated
    ideal state-clamp queries at lambda>0 recover one storage coefficient per
    state. A caller-supplied oracle is the designed in-silico intervention
    interface; the function never receives the true matrices.
    """
    if n < 2 or not np.isfinite(growth_rate_storage) or growth_rate_storage <= 0:
        raise InputContractError("unknown-topology-T3 requires n>=2 and a strictly positive storage growth level.")
    if not 0 <= root < n:
        raise InputContractError("Invalid unknown-topology root label.")
    base = recover_B_from_thresholds(n, oracle, 0.0, root=root, load=load)
    L = base.B.copy()
    if not np.allclose(L, L.T, atol=2e-7) or np.min(np.linalg.eigvalsh(L)) <= 0:
        raise PrecisionFailure("unknown-topology zero-growth recovery did not produce SPD L.")
    Cdiag = np.full(n, np.nan, dtype=float)
    zero = np.zeros(n, dtype=float)
    first = float(oracle(growth_rate_storage, zero, (root,)))
    Cdiag[root] = (first - L[root, root]) / growth_rate_storage
    if Cdiag[root] <= 0:
        raise PrecisionFailure("unknown-topology root-only clamp returned nonpositive storage.")
    dist, parent = _graph_paths(L, root)
    for node in sorted((i for i in range(n) if i != root), key=lambda x: (dist[x], x)):
        path = _path_to(node, parent)
        threshold_value = float(oracle(growth_rate_storage, zero, path))
        f = threshold_value
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            a_u = L[u, u] + growth_rate_storage * Cdiag[u]
            edge = -L[u, v]
            denom = a_u - f
            if edge <= 0 or denom <= 1e-13:
                raise PrecisionFailure(f"unknown-topology path continued-fraction inversion failed before node {v}.")
            f = edge**2 / denom
        Cdiag[node] = (f - L[node, node]) / growth_rate_storage
        if Cdiag[node] <= 0 or not np.isfinite(Cdiag[node]):
            raise PrecisionFailure(f"unknown-topology recovered nonpositive storage for node {node}.")
    if np.any(~np.isfinite(Cdiag)):
        raise UnidentifiableNode("unknown-topology did not recover every storage coefficient.")
    return UnknownTopologyJointRecovery(np.asarray(L), np.diag(Cdiag), base.query_count + n, growth_rate_storage)
