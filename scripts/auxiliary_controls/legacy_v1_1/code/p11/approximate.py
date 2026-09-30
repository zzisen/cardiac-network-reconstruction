from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .errors import InputContractError, PrecisionFailure, ReconstructionError
from .k8 import K8Observations
from .k9 import threshold_from_descriptor
from .models import incidence_path
from .scaling import L_to_J


def _path_matrices(n: int, theta: np.ndarray, offset: int = 0):
    local = np.exp(theta[offset : offset + n])
    coupling = np.exp(theta[offset + n : offset + n + n - 1])
    B = incidence_path(n)
    return np.diag(local) + B @ np.diag(coupling) @ B.T, local, coupling


def fit_k8_path(
    observations: K8Observations,
    C: np.ndarray,
    *,
    max_nfev: int = 3000,
) -> tuple[np.ndarray, dict]:
    """Constrained finite-noise fit for a known path and calibrated C.

    This is a numerical extension for noisy synthetic observations, not part of
    the exact K8 theorem. It fits positive local grounding and edge terms.
    """
    w = np.asarray(observations.frequencies, dtype=float)
    n = len(w)
    C = np.asarray(C, dtype=float)
    P = np.asarray(observations.profiles, dtype=float)
    hobs = np.asarray(observations.coherent_root, dtype=complex)
    yobs = np.asarray(observations.psd_differences, dtype=float)
    if C.shape != (n, n) or P.shape != (n - 2, n) or yobs.shape != (n - 2, n):
        raise InputContractError("Noisy K8 path fit received inconsistent dimensions.")
    if np.any(np.diag(C) <= 0) or not np.allclose(C, np.diag(np.diag(C))):
        raise InputContractError("Noisy K8 fit requires a calibrated positive diagonal C.")
    hscale = max(float(np.median(np.abs(hobs))), 1e-12)
    pscale = max(float(np.median(np.abs(yobs))), 1e-12)
    initials = [
        np.concatenate((np.full(n, np.log(g)), np.full(n - 1, np.log(k))))
        for g, k in ((0.18, 0.35), (0.02, 0.8), (0.30, 0.75))
    ]

    def predict(theta):
        L, _, _ = _path_matrices(n, theta)
        J = L_to_J(L, C)
        coh = np.empty(n, dtype=complex)
        diff = np.empty((n - 2, n), dtype=float)
        for k, omega in enumerate(w):
            G = np.linalg.inv(1j * omega * np.eye(n) - J)
            coh[k] = G[0, 0]
            diff[:, k] = P @ (np.abs(G[0, :]) ** 2)
        return L, J, coh, diff

    def residual(theta):
        _, _, hhat, phat = predict(theta)
        return np.concatenate((
            ((hhat - hobs).real / hscale),
            ((hhat - hobs).imag / hscale),
            ((phat - yobs) / pscale).ravel(),
        ))

    fits = [least_squares(
        residual,
        initial,
        bounds=(np.full(2 * n - 1, -9.0), np.full(2 * n - 1, 3.0)),
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    ) for initial in initials]
    fit = min(fits, key=lambda x: float(np.linalg.norm(x.fun)))
    Lhat, _, hhat, phat = predict(fit.x)
    diagnostics = {
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "normalized_residual_norm": float(np.linalg.norm(fit.fun)),
        "nfev": int(fit.nfev),
        "coherent_relative_residual": float(np.linalg.norm(hhat - hobs) / max(np.linalg.norm(hobs), 1e-15)),
        "profile_relative_residual": float(np.linalg.norm(phat - yobs) / max(np.linalg.norm(yobs), 1e-15)),
    }
    if not np.all(np.isfinite(Lhat)):
        raise PrecisionFailure("Noisy K8 constrained fit returned nonfinite coefficients.")
    return Lhat, diagnostics


@dataclass(frozen=True)
class NoisyK9Run:
    records: list[dict]
    design_status: str
    exact_inverse_status: str


def record_noisy_k9_schedule(
    L: np.ndarray,
    C: np.ndarray,
    *,
    growth_rate_storage: float,
    rng: np.random.Generator,
    noise_sd: float,
    root: int = 0,
) -> NoisyK9Run:
    """Execute the theorem query schedule with noisy returned scalar values.

    Pair-load magnitudes are computed from noisy baseline/axis responses, so
    this function retains the adaptive design dependence of K9-T1. The path
    family is known only to this finite-noise benchmark for its storage-clamp
    schedule; K9-T3 itself recovers the topology from L.
    """
    L, C = np.asarray(L), np.asarray(C)
    n = len(L)
    records: list[dict] = []
    zero = np.zeros(n)

    def read(lam, q, retained=None):
        y = threshold_from_descriptor(L, C, lam, q, retained, root=root)
        y += float(rng.normal(0.0, noise_sd))
        records.append({
            "growth_rate": float(lam),
            "q": np.asarray(q, dtype=float).copy(),
            "retained": tuple(range(n)) if retained is None else tuple(retained),
            "observed_threshold": float(y),
        })
        return float(y)

    try:
        T0 = read(0.0, zero)
        hidden = [v for v in range(n) if v != root]
        m = n - 1
        diagonal = {}
        gains = {}
        load = 1.0
        for node in hidden:
            q1, q2 = zero.copy(), zero.copy()
            q1[node], q2[node] = load, 2 * load
            T1, T2 = read(0.0, q1), read(0.0, q2)
            r1, r2 = T1 - T0, T2 - T0
            gap = r2 - r1
            if r1 <= 0 or gap <= 0:
                return NoisyK9Run(records, "axis_design_refused", "not_run")
            gii = (2 * r1 - r2) / (2 * load * gap)
            vi2 = r1 * (1 + load * gii) / load
            if gii <= 0 or vi2 <= 0:
                return NoisyK9Run(records, "axis_design_refused", "not_run")
            diagonal[node] = gii
            gains[node] = np.sqrt(vi2)
        for i in range(m):
            for j in range(i + 1, m):
                ni, nj = hidden[i], hidden[j]
                gi, gj = diagonal[ni], diagonal[nj]
                vi, vj = gains[ni], gains[nj]
                eps = vi * vj / (8 * (gi + gj) * (vi**2 + vj**2))
                if not np.isfinite(eps) or eps <= 0:
                    return NoisyK9Run(records, "pair_design_refused", "not_run")
                q = zero.copy()
                q[ni] = q[nj] = eps
                read(0.0, q)
        # In this finite-noise benchmark the known path topology supplies the
        # clamp sets. Exact K9-T3 instead builds them from recovered L.
        read(growth_rate_storage, zero, (root,))
        order = sorted((v for v in range(n) if v != root), key=lambda v: abs(v - root))
        for v in order:
            path = tuple(range(root, v + 1)) if root == 0 else tuple(range(root, v - 1, -1))
            read(growth_rate_storage, zero, path)
        return NoisyK9Run(records, "complete", "pending")
    except ReconstructionError as exc:
        return NoisyK9Run(records, f"{type(exc).__name__}: {exc}", "not_run")


def fit_k9_path_records(
    n: int,
    records: list[dict],
    *,
    max_nfev: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit positive local stiffness, path couplings and diagonal C to K9 reads.

    Known-path fitting is an explicit finite-noise benchmark restriction. It
    is separate from the exact topology-unknown K9 recovery theorem.
    """
    if len(records) < 3 * n - 1:
        raise InputContractError("Too few noisy K9 thresholds for the path-restricted fit.")
    if any(len(record["retained"]) == 0 for record in records):
        raise InputContractError("Invalid retained-state clamp record.")
    initial = np.concatenate((
        np.full(n, np.log(0.18)),
        np.full(n - 1, np.log(0.35)),
        np.full(n, np.log(1.0)),
    ))
    bounds = (np.full(3 * n - 1, -9.0), np.full(3 * n - 1, 3.0))
    yobs = np.array([r["observed_threshold"] for r in records], dtype=float)
    scale = max(float(np.std(yobs)), 0.1)

    def predict(theta):
        L, _, _ = _path_matrices(n, theta)
        Cdiag = np.exp(theta[2 * n - 1 :])
        C = np.diag(Cdiag)
        values = []
        for r in records:
            values.append(threshold_from_descriptor(
                L, C, r["growth_rate"], r["q"], r["retained"], root=0
            ))
        return L, C, np.asarray(values)

    def residual(theta):
        _, _, yhat = predict(theta)
        return (yhat - yobs) / scale

    fit = least_squares(
        residual,
        initial,
        bounds=bounds,
        max_nfev=max_nfev,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    Lhat, Chat, yhat = predict(fit.x)
    diagnostics = {
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "normalized_residual_norm": float(np.linalg.norm(fit.fun)),
        "nfev": int(fit.nfev),
        "threshold_relative_residual": float(np.linalg.norm(yhat - yobs) / max(np.linalg.norm(yobs), 1e-15)),
    }
    if not np.all(np.isfinite(Lhat)) or not np.all(np.isfinite(Chat)):
        raise PrecisionFailure("Noisy K9 constrained fit returned nonfinite coefficients.")
    return Lhat, Chat, diagnostics
