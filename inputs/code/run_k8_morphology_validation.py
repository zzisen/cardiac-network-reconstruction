"""K8 morphology motif checks and reproducible multi-seed profile sensitivities."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from p11.architecture_repair import fit_k8_tree  # noqa: E402
from p11.k8 import K8Observations, default_profiles, generate_observations, reconstruct  # noqa: E402
from p11.scaling import J_to_L, L_to_J  # noqa: E402


SEEDS = tuple(range(91001, 91021))
FREQUENCIES = np.array([0.35, 0.75, 1.20, 1.75])
FIT_MAX_NFEV = 3000


def load_motif() -> dict | None:
    selected = json.loads((ROOT / "results" / "public_subgraph_selection.json").read_text(encoding="utf-8"))
    return selected.get("k8_selected_frozen")


def mechanics_for_motif(motif: dict, mapping: str):
    n = 4
    edges = [tuple(map(int, edge)) for edge in motif["edges"]]
    xy = np.asarray(motif["node_positions_px"], dtype=float)
    edge_lengths = np.array([np.linalg.norm(xy[u] - xy[v]) for u, v in edges], dtype=float)
    median_length = float(np.median(edge_lengths))
    rng = np.random.default_rng(41001)
    local = np.clip(rng.normal(16.0, 1.3, size=n) / 16.0, 0.65, 1.35)
    if mapping == "inverse_length":
        coupling = 5.0 * median_length / edge_lengths
    elif mapping == "constant":
        coupling = np.full(len(edges), 5.0)
    else:
        raise ValueError(mapping)
    B = np.zeros((n, len(edges)))
    for k, (u, v) in enumerate(edges):
        B[u, k], B[v, k] = -1.0, 1.0
    L = np.diag(local) + B @ np.diag(coupling) @ B.T
    C = np.diag([0.92, 1.08, 0.86, 1.15])
    J = L_to_J(L, C)
    return J, L, C, edges, edge_lengths, local, coupling


def forward_with_profile(J, actual_profiles, frequencies, *, seed: int, noise_fraction: float, repeats: int):
    rng = np.random.default_rng(seed)
    n = len(J)
    root = 0
    hs, ys = [], []
    power_scales = []
    for _ in range(repeats):
        h = np.empty(n, dtype=complex)
        y = np.empty((n - 2, n), dtype=float)
        for k, omega in enumerate(frequencies):
            G = np.linalg.inv(1j * omega * np.eye(n) - J)
            row_power = np.abs(G[root, :]) ** 2
            power_scales.append(float(np.sum(row_power)))
            h[k] = G[root, root]
            y[:, k] = actual_profiles @ row_power
        hscale = max(float(np.median(np.abs(h))), 1e-8)
        h += noise_fraction * hscale * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2.0)
        # Noise remains referenced to the untreated response scale, so lower
        # dose also lowers the profile-response signal-to-noise ratio.
        y += noise_fraction * max(float(np.median(power_scales)), 1e-6) * rng.normal(size=y.shape)
        hs.append(h)
        ys.append(y)
    return np.mean(hs, axis=0), np.mean(ys, axis=0)


def relative_L_error(Jhat, true_L, C) -> float:
    return float(np.linalg.norm(J_to_L(Jhat, C) - true_L) / np.linalg.norm(true_L))


def fit_and_score_case(observations, C, edges, true_L, true_J, *, root=0):
    """Fit normalized J, then use the single canonical transform to score L."""
    Jhat, diagnostics = fit_k8_tree(observations, C, edges, root=root, max_nfev=FIT_MAX_NFEV)
    L_error = relative_L_error(Jhat, true_L, C)
    J_error = float(np.linalg.norm(Jhat - true_J) / np.linalg.norm(true_J))
    return Jhat, diagnostics, L_error, J_error


def make_sensitivity_settings(nominal: np.ndarray, leak: np.ndarray) -> list[dict]:
    settings = []
    for dose in (0.25, 0.50, 1.0):
        settings.append({"condition": "profile_amplitude_limited_known", "profile_amplitude": dose,
                         "profile_calibration_error": 0.0, "crosstalk_fraction": 0.0,
                         "noise_fraction": 1e-3, "repeats": 5,
                         "actual_profiles": dose * nominal, "reported_profiles": dose * nominal})
    for calibration in (-0.15, -0.05, 0.05, 0.15):
        settings.append({"condition": "profile_strength_miscalibrated", "profile_amplitude": 1.0 + calibration,
                         "profile_calibration_error": calibration, "crosstalk_fraction": 0.0,
                         "noise_fraction": 1e-3, "repeats": 5,
                         "actual_profiles": (1.0 + calibration) * nominal, "reported_profiles": nominal})
    for leakage in (0.05, 0.15, 0.30):
        settings.append({"condition": "unmodelled_spatial_crosstalk", "profile_amplitude": 1.0,
                         "profile_calibration_error": 0.0, "crosstalk_fraction": leakage,
                         "noise_fraction": 1e-3, "repeats": 5,
                         "actual_profiles": (1.0 - leakage) * nominal + leakage * leak,
                         "reported_profiles": nominal})
    for repeats in (1, 3, 10):
        settings.append({"condition": "repeated_measurement_noise", "profile_amplitude": 1.0,
                         "profile_calibration_error": 0.0, "crosstalk_fraction": 0.0,
                         "noise_fraction": 3e-3, "repeats": repeats,
                         "actual_profiles": nominal, "reported_profiles": nominal})
    return settings


def summarize_sensitivities(replicates: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["condition", "profile_amplitude", "profile_calibration_error", "crosstalk_fraction",
                     "noise_fraction", "repeats", "query_or_read_count"]
    rows = []
    for key, group in replicates.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, key))
        l_values = group["relative_L_error"].dropna().to_numpy(dtype=float)
        j_values = group["relative_J_error"].dropna().to_numpy(dtype=float)
        attempted = int(len(group))
        converged = int(group["fit_success"].astype(bool).sum())
        row.update({"seed_count": attempted, "converged_count": converged,
                    "failure_count": attempted - converged,
                    "convergence_rate": converged / attempted if attempted else np.nan,
                    "failure_rate": 1.0 - converged / attempted if attempted else np.nan,
                    "finite_L_error_count": int(len(l_values)), "finite_J_error_count": int(len(j_values))})
        for name, values in (("relative_L_error", l_values), ("relative_J_error", j_values)):
            for suffix, value in (("median", np.median(values) if len(values) else np.nan),
                                  ("q25", np.quantile(values, 0.25) if len(values) else np.nan),
                                  ("q75", np.quantile(values, 0.75) if len(values) else np.nan),
                                  ("p90", np.quantile(values, 0.90) if len(values) else np.nan)):
                row[f"{name}_{suffix}"] = float(value)
            row[f"{name}_iqr"] = float(row[f"{name}_q75"] - row[f"{name}_q25"])
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    motif = load_motif()
    if motif is None:
        raise RuntimeError("The morphology-only frozen K8 motif is missing.")
    edges = [tuple(map(int, edge)) for edge in motif["edges"]]
    n, root = 4, 0
    nominal = default_profiles(n, root=root)
    sensitivity_rows = []
    mechanics_rows = []
    exact_checks = []

    # Exact/theorem-status and generated zero-imperfection fit checks.
    for mapping in ("inverse_length", "constant"):
        J, L, C, current_edges, lengths, local, coupling = mechanics_for_motif(motif, mapping)
        observations = generate_observations(J, current_edges, FREQUENCIES, nominal, root=root)
        try:
            exact = reconstruct(observations, current_edges, root=root)
            exact_L_error = relative_L_error(exact.J, L, C)
            exact_status = "exact_inverse_recovered"
        except Exception as exc:
            exact_L_error = np.nan
            exact_status = f"{type(exc).__name__}: {exc}"
        try:
            _, fit_diagnostics, fit_L_error, fit_J_error = fit_and_score_case(
                observations, C, current_edges, L, J, root=root)
            fit_success = bool(fit_diagnostics["optimizer_success"])
        except Exception as exc:
            fit_diagnostics = {"optimizer_success": False, "optimizer_message": f"{type(exc).__name__}: {exc}"}
            fit_L_error = fit_J_error = np.nan
            fit_success = False
        swap = np.eye(n)[[0, 1, 3, 2]]
        swapped = generate_observations(swap @ J @ swap.T, current_edges, FREQUENCIES, nominal, root=root)
        exact_checks.append({"mapping": mapping, "exact_inverse_status": exact_status,
                             "exact_inverse_relative_L_error": (exact_L_error if np.isfinite(exact_L_error) else None),
                             "known_tree_fit_success": fit_success,
                             "known_tree_fit_relative_L_error": (fit_L_error if np.isfinite(fit_L_error) else None),
                             "known_tree_fit_relative_J_error": (fit_J_error if np.isfinite(fit_J_error) else None),
                             "known_tree_fit_message": fit_diagnostics.get("optimizer_message", ""),
                             "coherent_daughter_swap_max_abs": float(np.max(np.abs(
                                 observations.coherent_root - swapped.coherent_root))),
                             "labelled_profile_swap_max_abs": float(np.max(np.abs(
                                 observations.psd_differences - swapped.psd_differences))),
                             "edge_length_min_px": float(np.min(lengths)), "edge_length_max_px": float(np.max(lengths)),
                             "coupling_min": float(np.min(coupling)), "coupling_max": float(np.max(coupling)),
                             "local_grounding_min": float(np.min(local)), "local_grounding_max": float(np.max(local))})
        mechanics_rows.append({"mapping": mapping, "known_tree_fit_relative_L_error": fit_L_error,
                               "known_tree_fit_relative_J_error": fit_J_error,
                               "known_tree_fit_success": fit_success,
                               "exact_inverse_relative_L_error": exact_L_error,
                               "exact_inverse_status": exact_status,
                               "edge_length_min_px": float(np.min(lengths)), "edge_length_max_px": float(np.max(lengths)),
                               "coupling_min": float(np.min(coupling)), "coupling_max": float(np.max(coupling)),
                               "local_grounding_min": float(np.min(local)), "local_grounding_max": float(np.max(local))})

    J, L, C, edges, _, _, _ = mechanics_for_motif(motif, "inverse_length")
    leak = np.zeros_like(nominal)
    for row_index, profile in enumerate(nominal):
        target = int(np.flatnonzero(profile)[0])
        adjacent = [v for v in range(n) if v not in (target, root) and
                    tuple(sorted((target, v))) in {tuple(sorted(edge)) for edge in edges}]
        leak[row_index, min(adjacent) if adjacent else root] = 1.0

    settings = make_sensitivity_settings(nominal, leak)
    for setting_index, setting in enumerate(settings):
        read_count = int(setting["repeats"] * n * (1 + n - 2))
        for seed in SEEDS:
            h, y = forward_with_profile(J, setting["actual_profiles"], FREQUENCIES, seed=seed,
                                        noise_fraction=setting["noise_fraction"], repeats=setting["repeats"])
            observations = K8Observations(FREQUENCIES, h, setting["reported_profiles"], y)
            try:
                Jhat, diagnostics, L_error, J_error = fit_and_score_case(
                    observations, C, edges, L, J, root=root)
                fit_success = bool(diagnostics["optimizer_success"])
                status = "fit_complete" if fit_success else "optimizer_not_converged"
                residual_norm = float(diagnostics.get("weighted_residual_norm", np.nan))
                message = str(diagnostics.get("optimizer_message", ""))
            except Exception as exc:
                fit_success, L_error, J_error = False, np.nan, np.nan
                residual_norm = np.nan
                message = f"{type(exc).__name__}: {exc}"
                status = message
            sensitivity_rows.append({"setting_id": f"S{setting_index + 1:02d}",
                                     "condition": setting["condition"], "seed": seed,
                                     "profile_amplitude": setting["profile_amplitude"],
                                     "profile_calibration_error": setting["profile_calibration_error"],
                                     "crosstalk_fraction": setting["crosstalk_fraction"],
                                     "noise_fraction": setting["noise_fraction"], "repeats": setting["repeats"],
                                     "query_or_read_count": read_count, "fit_success": fit_success,
                                     "relative_L_error": L_error, "relative_J_error": J_error,
                                     "weighted_residual_norm": residual_norm, "status": status,
                                     "optimizer_message": message})

    replicates = pd.DataFrame(sensitivity_rows)
    summary = summarize_sensitivities(replicates)
    replicates.to_csv(ROOT / "results" / "k8_morphology_imperfections.csv", index=False, float_format="%.10g")
    replicates.to_csv(ROOT / "results" / "k8_morphology_sensitivity_replicates.csv", index=False, float_format="%.10g")
    summary.to_csv(ROOT / "results" / "k8_morphology_sensitivity_summary.csv", index=False, float_format="%.10g")
    pd.DataFrame(mechanics_rows).to_csv(ROOT / "results" / "k8_mechanics_mapping_sensitivity.csv", index=False,
                                        float_format="%.10g")
    (ROOT / "results" / "k8_exact_checks.json").write_text(
        json.dumps({"status": "complete", "morphology_motif": motif, "checks": exact_checks},
                   indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "results" / "k8_morphology_status.json").write_text(
        json.dumps({"status": "REAL_MORPHOLOGY_MOTIF_USED", "movie": motif["movie"],
                    "node_order": motif["node_order"], "root": motif["root_node"],
                    "junction": motif["junction_node"], "detection_persistence": motif["detection_persistence"],
                    "selection_rule": "results/k8_motif_selection_freeze.json",
                    "mapping_alternative": "inverse edge length versus constant edge weights",
                    "sensitivity_seed_list": list(SEEDS), "seeds_per_setting": len(SEEDS),
                    "sensitivity_setting_count": len(settings), "sensitivity_fit_count": len(replicates),
                    "metrics_are_model_generated_not_empirical_parameter_recovery": True},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
