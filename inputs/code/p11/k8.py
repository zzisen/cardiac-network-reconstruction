from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
from numpy.polynomial import Polynomial

from .errors import (
    InconsistentObservation,
    InputContractError,
    PrecisionFailure,
    ProfileRankDeficient,
    TheoremBoundaryFailure,
    UnidentifiableNode,
)


@dataclass(frozen=True)
class K8Observations:
    frequencies: np.ndarray
    coherent_root: np.ndarray
    profiles: np.ndarray
    psd_differences: np.ndarray


@dataclass(frozen=True)
class K8Recovery:
    J: np.ndarray
    reduced_denominator: np.ndarray
    row_numerators: dict[int, np.ndarray]
    distances: dict[int, int]
    coherent_fit_error: float
    profile_fit_error: float


def _tree_distances(n: int, edges: list[tuple[int, int]], root: int) -> tuple[dict[int, int], dict[int, int]]:
    if not 0 <= root < n:
        raise InputContractError("The root label must lie in 0,...,n-1.")
    adj = [[] for _ in range(n)]
    edge_set: set[tuple[int, int]] = set()
    for a, b in edges:
        if a == b or not (0 <= a < n and 0 <= b < n):
            raise InputContractError("Tree edges must join two distinct labelled nodes.")
        key = tuple(sorted((int(a), int(b))))
        if key in edge_set:
            raise InputContractError("Duplicate tree edge.")
        edge_set.add(key)
        adj[a].append(b)
        adj[b].append(a)
    if len(edge_set) != n - 1:
        raise TheoremBoundaryFailure("K8 requires a supplied labelled tree with n-1 edges.")
    parent = {root: -1}
    distance = {root: 0}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in distance:
                parent[v] = u
                distance[v] = distance[u] + 1
                queue.append(v)
    if len(distance) != n:
        raise TheoremBoundaryFailure("The supplied K8 graph is disconnected.")
    return distance, parent


def default_profiles(n: int, root: int = 0, omitted: int | None = None) -> np.ndarray:
    hidden = [v for v in range(n) if v != root]
    if omitted is None:
        omitted = hidden[-1]
    if omitted not in hidden:
        raise InputContractError("The omitted K8 noise profile must be a hidden node.")
    included = [v for v in hidden if v != omitted]
    profiles = np.zeros((n - 2, n), dtype=float)
    for row, v in enumerate(included):
        profiles[row, v] = 1.0
    return profiles


def generate_observations(
    J: np.ndarray,
    edges: list[tuple[int, int]],
    frequencies: np.ndarray,
    profiles: np.ndarray | None = None,
    *,
    root: int = 0,
    background_scale: float = 0.4,
) -> K8Observations:
    """Generate ideal K8 coherent and added-noise spectral values.

    A shared, frequency-dependent ambient spectrum is added to every run and
    subtracted. The stored PSD values are the exact-contract differences.
    """
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    _tree_distances(n, edges, root)
    validate_positive_tree_generator(J, edges)
    w = np.asarray(frequencies, dtype=float)
    if w.shape != (n,) or np.any(w <= 0) or len(np.unique(w)) != n:
        raise InputContractError("K8 requires n distinct positive frequencies.")
    P = default_profiles(n, root) if profiles is None else np.asarray(profiles, dtype=float)
    if P.shape != (n - 2, n) or np.any(P < 0):
        raise InputContractError("K8 requires n-2 known nonnegative fixed spatial profiles.")
    coherent = np.empty(n, dtype=complex)
    diffs = np.empty((n - 2, n), dtype=float)
    base = background_scale * (1.0 + w / (1.0 + w))
    for k, omega in enumerate(w):
        G = np.linalg.inv(1j * omega * np.eye(n) - J)
        coherent[k] = G[root, root]
        row_power = np.abs(G[root, :]) ** 2
        baseline = base[k]
        observed_runs = baseline + P @ row_power
        diffs[:, k] = observed_runs - baseline
    return K8Observations(w, coherent, P, diffs)


def validate_positive_tree_generator(J: np.ndarray, edges: list[tuple[int, int]]) -> None:
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    if J.shape != (n, n) or not np.all(np.isfinite(J)):
        raise InputContractError("J must be a finite square real matrix.")
    if not np.allclose(J, J.T, atol=1e-10):
        raise TheoremBoundaryFailure("K8 requires a symmetric generator.")
    if np.max(np.linalg.eigvalsh(J)) >= 0:
        raise TheoremBoundaryFailure("K8 requires a negative-definite generator.")
    edge_set = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    for i in range(n):
        for j in range(i + 1, n):
            if ((i, j) in edge_set and J[i, j] <= 0) or ((i, j) not in edge_set and abs(J[i, j]) > 1e-10):
                raise TheoremBoundaryFailure("J must have strictly positive off-diagonals exactly on the supplied tree.")


def _fit_reduced_rational(
    frequencies: np.ndarray,
    values: np.ndarray,
    *,
    minimum_degree: int = 1,
    tolerance: float = 1e-13,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit the lowest-degree real rational h=N/D allowed by K8 finite samples."""
    n = len(frequencies)
    s = 1j * np.asarray(frequencies, dtype=float)
    h = np.asarray(values, dtype=complex)
    if frequencies.shape != (n,) or h.shape != (n,) or not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(h)):
        raise InputContractError("K8 coherent frequencies and samples must be finite vectors of length n.")
    for degree in range(max(1, minimum_degree), n + 1):
        powers = np.column_stack([s**ell for ell in range(degree)])
        A_complex = np.column_stack((h[:, None] * powers, -powers))
        b_complex = -h * s**degree
        A = np.vstack((A_complex.real, A_complex.imag))
        b = np.concatenate((b_complex.real, b_complex.imag))
        if np.linalg.matrix_rank(A) < 2 * degree:
            continue
        coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        residual = float(np.linalg.norm(A @ coeff - b) / max(1.0, np.linalg.norm(b)))
        if residual <= tolerance:
            den = np.concatenate(([1.0], coeff[:degree][::-1]))
            num = coeff[degree:][::-1]
            poles = np.roots(den)
            if np.any(~np.isfinite(poles)) or np.any(np.real(poles) >= -1e-8):
                continue
            if np.linalg.cond(A) > 1e14:
                raise PrecisionFailure("Coherent rational interpolation is ill-conditioned at double precision.")
            return num, den, residual
    raise PrecisionFailure("Finite coherent samples did not yield a stable real rational transfer.")


def _spectral_factor_from_power(
    frequencies: np.ndarray,
    row_power: np.ndarray,
    denominator: np.ndarray,
    numerator_degree: int,
) -> tuple[np.ndarray, float]:
    s = 1j * frequencies
    dvals = np.polyval(denominator, s)
    samples = np.asarray(row_power, dtype=float) * np.abs(dvals) ** 2
    x = frequencies**2
    poly = Polynomial.fit(x, samples, deg=numerator_degree).convert()
    r = np.zeros(numerator_degree + 1, dtype=float)
    r[: min(len(poly.coef), len(r))] = poly.coef[: len(r)]
    fitted = np.polynomial.polynomial.polyval(x, r)
    fit_error = float(np.linalg.norm(fitted - samples) / max(1.0, np.linalg.norm(samples)))
    if not np.all(np.isfinite(r)) or fit_error > 2e-5:
        raise PrecisionFailure("Row-power interpolation is inconsistent at the declared precision.")
    if r[numerator_degree] <= 1e-14 * max(1.0, float(np.max(np.abs(r)))):
        raise PrecisionFailure("A K8 row has zero or unresolved leading path gain at double precision.")
    if numerator_degree == 0:
        return np.array([np.sqrt(r[0])]), fit_error
    # R(-s^2) = N(s)N(-s). Its real roots occur in +/- pairs; the stable
    # minimum-phase factor uses the negative member of each pair.
    even = np.zeros(2 * numerator_degree + 1, dtype=float)
    for j, value in enumerate(r):
        even[2 * j] = ((-1.0) ** j) * value
    roots = np.roots(even[::-1])
    scale = max(1.0, float(np.max(np.abs(roots))))
    root_tolerance = 2e-5 * scale
    if np.any(~np.isfinite(roots)) or np.any(np.abs(np.imag(roots)) > root_tolerance):
        raise PrecisionFailure("The measured row power does not have a fully real K8 spectral factor.")
    real_roots = np.real(roots)
    positive = np.sort(real_roots[real_roots > root_tolerance])
    negative_magnitudes = np.sort(-real_roots[real_roots < -root_tolerance])
    if len(positive) != numerator_degree or len(negative_magnitudes) != numerator_degree:
        raise PrecisionFailure("The measured row power lacks matched real +/- roots required by K8.")
    if np.max(np.abs(positive - negative_magnitudes)) > root_tolerance:
        raise PrecisionFailure("The measured row-power roots are not matched +/- pairs.")
    negative = np.sort(real_roots[real_roots < -root_tolerance])
    numerator = np.sqrt(r[numerator_degree]) * np.poly(negative)
    reflected = numerator * ((-1.0) ** np.arange(numerator_degree, -1, -1))
    reconstructed_even = np.polymul(numerator, reflected)
    polynomial_error = float(np.linalg.norm(reconstructed_even - even[::-1]) / max(np.linalg.norm(even), np.finfo(float).tiny))
    if polynomial_error > max(2e-5, 10.0 * fit_error):
        raise PrecisionFailure("The K8 spectral factor does not reproduce the fitted row-power polynomial.")
    return np.asarray(numerator, dtype=float), fit_error


def _high_frequency_path_coefficients(
    numerator: np.ndarray,
    denominator: np.ndarray,
    distance: int,
) -> tuple[float, float]:
    k = len(denominator) - 1
    q = np.asarray(denominator, dtype=float)  # 1 + d_(k-1)t + ... + d_0t^k
    p = np.zeros(k + 1, dtype=float)
    degree = len(numerator) - 1
    start = k - degree
    p[start : start + len(numerator)] = numerator
    series = np.zeros(distance + 3, dtype=float)
    for m in range(len(series)):
        correction = sum(q[j] * series[m - j] for j in range(1, min(m, k) + 1))
        p_value = p[m] if m < len(p) else 0.0
        series[m] = (p_value - correction) / q[0]
    order = distance + 1
    gain = float(series[order])
    if not np.isfinite(gain) or abs(gain) <= 1e-12:
        raise PrecisionFailure("A row transfer has no resolvable shortest-path coefficient at double precision.")
    tau = float(series[order + 1] / gain)
    return gain, tau


def reconstruct(
    observations: K8Observations,
    edges: list[tuple[int, int]],
    *,
    root: int = 0,
) -> K8Recovery:
    """Recover the labelled positive-tree generator from K8 observations."""
    w = np.asarray(observations.frequencies, dtype=float)
    h = np.asarray(observations.coherent_root, dtype=complex)
    P = np.asarray(observations.profiles, dtype=float)
    y = np.asarray(observations.psd_differences, dtype=float)
    n = len(w)
    dist, parent = _tree_distances(n, edges, root)
    if w.shape != (n,) or len(np.unique(w)) != n or np.any(w <= 0):
        raise InputContractError("K8 requires n distinct positive frequencies and n coherent values.")
    if P.shape != (n - 2, n) or y.shape != (n - 2, n):
        raise InputContractError("K8 requires n-2 fixed profiles and one PSD difference per frequency.")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(y)) or np.any(P < 0):
        raise InputContractError("K8 profiles and PSD differences must be finite and nonnegative profiles.")

    num_root, den, coherent_error = _fit_reduced_rational(
        w, h, minimum_degree=max(dist.values()) + 1
    )
    if len(den) - 1 > n:
        raise PrecisionFailure("The recovered coherent denominator exceeds the state dimension.")
    total = -np.imag(h) / w
    root_power = np.abs(h) ** 2
    if np.any(total <= 0) or np.any(total + 1e-9 < root_power):
        raise InconsistentObservation("The coherent samples violate the K8 resolvent power identity.")
    hidden = [v for v in range(n) if v != root]
    profile_matrix = np.vstack((np.ones(n - 1), P[:, hidden]))
    if np.linalg.matrix_rank(profile_matrix, tol=1e-10) != n - 1:
        raise ProfileRankDeficient("The all-ones row plus hidden profile restrictions must have rank n-1.")
    right = np.vstack((total - root_power, y - P[:, root, None] * root_power[None, :]))
    hidden_power = np.linalg.solve(profile_matrix, right)
    if np.any(hidden_power <= 0):
        raise InconsistentObservation("A reconstructed hidden row power is nonpositive.")
    powers = {root: root_power}
    powers.update({v: hidden_power[j] for j, v in enumerate(hidden)})

    numerators: dict[int, np.ndarray] = {root: num_root}
    power_fit_errors: list[float] = []
    denominator_degree = len(den) - 1
    for v in hidden:
        numerator_degree = denominator_degree - dist[v] - 1
        if numerator_degree < 0:
            raise PrecisionFailure("The reduced root denominator is too small for a supplied tree distance.")
        nv, err = _spectral_factor_from_power(w, powers[v], den, numerator_degree)
        numerators[v] = nv
        power_fit_errors.append(err)

    gains: dict[int, float] = {}
    taus: dict[int, float] = {}
    for v in range(n):
        gains[v], taus[v] = _high_frequency_path_coefficients(numerators[v], den, dist[v])
    # The root leading coefficient is fixed by the resolvent normalization.
    if not np.isclose(gains[root], 1.0, rtol=1e-5, atol=1e-7):
        raise PrecisionFailure("The recovered root transfer violates its known high-frequency gain.")
    J = np.zeros((n, n), dtype=float)
    J[root, root] = taus[root]
    for v in range(n):
        if v == root:
            continue
        p = parent[v]
        edge_gain = gains[v] / gains[p]
        if not np.isfinite(edge_gain) or edge_gain <= 0:
            raise UnidentifiableNode(f"Node {v} has no positive recovered edge gain.")
        J[p, v] = J[v, p] = edge_gain
        J[v, v] = taus[v] - taus[p]
    try:
        validate_positive_tree_generator(J, edges)
    except TheoremBoundaryFailure as exc:
        raise PrecisionFailure(f"Recovered generator violates the K8 class: {exc}") from exc
    profile_error = float(np.mean(power_fit_errors)) if power_fit_errors else 0.0
    return K8Recovery(J, den, numerators, dist, coherent_error, profile_error)
