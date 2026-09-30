from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

from .errors import PrecisionFailure
from .k8 import K8Observations, default_profiles, reconstruct


def _load_model(vendor_src: Path):
    vendor_text = str(vendor_src.resolve())
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    module = importlib.import_module("model.model")
    return module.Model, module.ModelParams, module.SimParams


def _model_run(
    vendor_src: Path,
    *,
    seed: int,
    extra_force: np.ndarray | None = None,
    extra_white_site: int | None = None,
    added_noise_intensity: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    Model, ModelParams, SimParams = _load_model(vendor_src)
    n = 6
    dt, duration = 0.002, 6.0
    steps = int(duration / dt)
    np.random.seed(seed)
    params = ModelParams(
        N=n,
        mu=0.006,
        eta=[0.05, 0.05],
        k_l=1.5,
        k_s=[3.5, 3.0],
        u_s=0.1,
        o_s=None,
        f_s=[-0.77, -0.06, 0.10, 0.01],
        h=0.0,
    )
    initial = np.zeros(2 * n, dtype=float)
    sim = SimParams(
        t1=duration,
        dt=dt,
        a_mode="sine",
        a=[1.0, 0.0, 1.0],
        p=0.0,
        y0=initial,
    )
    model = Model(params, sim, verbose=False)
    t = np.arange(steps) * dt
    if extra_force is not None:
        model.eta_thermal[0, :] += np.asarray(extra_force, dtype=float)
    if extra_white_site is not None:
        rng = np.random.default_rng(seed + 9000 + extra_white_site)
        model.eta_thermal[extra_white_site, :] += rng.normal(
            0.0, np.sqrt(added_noise_intensity / dt), size=steps
        )
    model.integrate()
    x = np.asarray(model.data["length"], dtype=float)
    if x.shape != (n, steps) or not np.all(np.isfinite(x)):
        raise PrecisionFailure("SarcomereModel returned a nonfinite or unexpected trajectory.")
    return t, x, dt


def run_sarcomere_model_mismatch(vendor_src: Path) -> tuple[list[dict], dict]:
    """Run a bounded out-of-class K8 stress test using the supplied model.

    The source model includes inertia, nonlinear force-velocity mechanics,
    activation-dependent stochasticity, and a global load term. The returned
    observations are deliberately treated as a mismatch probe, not as K8 data.
    """
    n, root, seed = 6, 0, 78031
    t, baseline, dt = _model_run(vendor_src, seed=seed)
    duration = float(t[-1] + dt)
    fs = 1.0 / dt
    # Six distinct Fourier bins, avoiding the 1-Hz activation fundamental.
    bins = np.array([2, 3, 4, 5, 7, 8], dtype=int)
    f_hz = bins / duration
    omega = 2.0 * np.pi * f_hz
    profiles = default_profiles(n, root=root)
    q_intensity = 1e-5
    _, p_base = periodogram(
        baseline[root], fs=fs, window="hann", detrend="linear", scaling="density"
    )
    profile_differences = np.zeros((n - 2, n), dtype=float)
    rows: list[dict] = []
    for j, profile in enumerate(profiles):
        site = int(np.argmax(profile))
        _, treated, _ = _model_run(
            vendor_src, seed=seed, extra_white_site=site, added_noise_intensity=q_intensity
        )
        p_treat = periodogram(
            treated[root], fs=fs, window="hann", detrend="linear", scaling="density"
        )[1]
        for k, f in enumerate(f_hz):
            idx = int(bins[k])
            profile_differences[j, k] = p_treat[idx] - p_base[idx]
            rows.append({
                "profile_index": j,
                "injected_site": site,
                "frequency_hz": float(f),
                "omega_rad_s": float(omega[k]),
                "baseline_root_psd": float(p_base[idx]),
                "treated_root_psd": float(p_treat[idx]),
                "psd_difference": float(profile_differences[j, k]),
                "added_noise_intensity": q_intensity,
                "model": "Haertter_SarcomereModel_v0.2.0",
            })

    coherent = np.empty(n, dtype=complex)
    probe_amplitude = 0.2
    for k, f in enumerate(f_hz):
        force = -probe_amplitude * np.sin(2.0 * np.pi * f * t)
        _, forced, _ = _model_run(vendor_src, seed=seed, extra_force=force)
        demod = np.exp(-2j * np.pi * f * t)
        numerator = np.sum((forced[root] - baseline[root]) * demod)
        denominator = np.sum(force * demod)
        coherent[k] = numerator / denominator
    observations = K8Observations(omega, coherent, profiles, profile_differences)
    try:
        recovered = reconstruct(observations, [(i, i + 1) for i in range(n - 1)], root=root)
        recovery_status = "recovered_out_of_class_surrogate"
        recovery_error = None
        recovered_eigen_max = float(np.max(np.linalg.eigvalsh(recovered.J)))
        coherent_fit = recovered.coherent_fit_error
        profile_fit = recovered.profile_fit_error
    except Exception as exc:  # Typed refusal/error is itself a prespecified outcome.
        recovery_status = f"{type(exc).__name__}: {exc}"
        recovery_error = type(exc).__name__
        recovered_eigen_max = None
        coherent_fit = None
        profile_fit = None
    summary = {
        "model": "Haertter SarcomereModel v0.2.0",
        "seed": seed,
        "n_sarcomeres": n,
        "duration_s": duration,
        "dt_s": dt,
        "activation_hz": 1.0,
        "added_noise_intensity": q_intensity,
        "n_profiles": n - 2,
        "n_coherent_bins": n,
        "nonlinear_model_is_theorem_class": False,
        "mismatch_features": [
            "inertial second-order state",
            "nonlinear and piecewise force-length/force-velocity terms",
            "activation-dependent stochastic drive",
            "global load coupling rather than a nearest-neighbour path",
        ],
        "recovery_status": recovery_status,
        "recovery_error_type": recovery_error,
        "recovered_max_eigenvalue": recovered_eigen_max,
        "coherent_fit_error": coherent_fit,
        "profile_fit_error": profile_fit,
        "baseline_active_cycles": "source-model activation cycles; no human data",
    }
    return rows, summary
