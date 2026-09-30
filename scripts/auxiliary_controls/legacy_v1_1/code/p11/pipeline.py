from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import clamp_boundary_control, run_exact_recoveries, run_noise_trials
from .figures import make_figures
from .fixtures import exact_k9_path_fixture
from .architecture_repair import (
    fit_k8_tree,
    local_edge_parameters,
    make_cyclic_k9_fixture,
    make_y_tree_fixture,
    permute_y_daughters,
    recover_jacobi_from_spectral_measure,
)
from .errors import PrecisionFailure
from .k8 import K8Observations, default_profiles, generate_observations, reconstruct
from .k9 import CountingThresholdOracle, recover_joint_LC, threshold_from_descriptor
from .mismatch import run_sarcomere_model_mismatch
from .models import make_path_system, normalized_J
from .scaling import J_to_L


BUILD_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, data: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (np.floating, float)):
            number = float(value)
            return number if np.isfinite(number) else None
        return value

    path.write_text(json.dumps(clean(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _example_matrices() -> pd.DataFrame:
    n, seed = 6, 6206
    system = make_path_system(n, seed)
    edges = [(i, i + 1) for i in range(n - 1)]
    omega = np.linspace(0.35, 1.75, n)
    k8obs = generate_observations(normalized_J(system), edges, omega, default_profiles(n))
    k8rec = reconstruct(k8obs, edges)
    L8 = J_to_L(k8rec.J, system.C)
    oracle = CountingThresholdOracle(
        lambda lam, q, retained: threshold_from_descriptor(system.L, system.C, lam, q, retained)
    )
    k9rec = recover_joint_LC(n, oracle, growth_rate_storage=0.8)
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "n": n,
                "i": i,
                "j": j,
                "L_true": float(system.L[i, j]),
                "L_k8_recovered": float(L8[i, j]),
                "L_k9_recovered": float(k9rec.L[i, j]),
                "C_true_diagonal": float(system.C[i, i]) if i == j else np.nan,
                "C_k9_recovered_diagonal": float(k9rec.C[i, i]) if i == j else np.nan,
                "C_k8_assumed_diagonal": float(system.C[i, i]) if i == j else np.nan,
            })
    return pd.DataFrame(rows)


def _architecture_repair_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce the Direction-C branch/cycle evidence and bounded noise probes."""
    branch = make_y_tree_fixture()
    omega = np.linspace(0.35, 1.75, 4)
    profiles = default_profiles(4, root=0)
    observations = generate_observations(branch.J, branch.edges, omega, profiles, root=0)
    recovered = reconstruct(observations, branch.edges, root=0)
    swapped = permute_y_daughters(branch.J)
    L_swapped = J_to_L(swapped, branch.C)
    swapped_local, swapped_coupling = local_edge_parameters(L_swapped, branch.edges)
    swapped_is_passive = bool(np.all(swapped_local > 0) and np.all(swapped_coupling > 0))
    alias_rows: list[dict] = []
    coherent_alias_error = 0.0
    for k, frequency in enumerate(omega):
        G = np.linalg.inv(1j * frequency * np.eye(4) - branch.J)
        Gswap = np.linalg.inv(1j * frequency * np.eye(4) - swapped)
        coherent_alias_error = max(coherent_alias_error, float(abs(G[0, 0] - Gswap[0, 0])))
        alias_rows.append({
            "frequency": float(frequency),
            "coherent_original_real": float(G[0, 0].real),
            "coherent_original_imag": float(G[0, 0].imag),
            "coherent_daughter_swapped_real": float(Gswap[0, 0].real),
            "coherent_daughter_swapped_imag": float(Gswap[0, 0].imag),
            "profile_node_3_power_original": float(abs(G[0, 2]) ** 2),
            "profile_node_3_power_after_swap": float(abs(Gswap[0, 2]) ** 2),
            "profile_power_difference": float(abs(G[0, 2]) ** 2 - abs(Gswap[0, 2]) ** 2),
        })

    cycle = make_cyclic_k9_fixture()
    cycle_oracle = CountingThresholdOracle(
        lambda lam, q, retained: threshold_from_descriptor(cycle.L, cycle.C, lam, q, retained)
    )
    cycle_recovered = recover_joint_LC(5, cycle_oracle, growth_rate_storage=0.8)
    true_edge_set = {tuple(sorted(edge)) for edge in cycle.edges}
    recovered_edge_set = {
        (i, j)
        for i in range(5)
        for j in range(i + 1, 5)
        if cycle_recovered.L[i, j] < -1e-8
    }
    cycle_rows: list[dict] = []
    for i in range(5):
        for j in range(5):
            cycle_rows.append({
                "i": i,
                "j": j,
                "L_true": float(cycle.L[i, j]),
                "L_recovered": float(cycle_recovered.L[i, j]),
                "edge_true": int(tuple(sorted((i, j))) in true_edge_set) if i != j else 0,
                "edge_recovered": int(cycle_recovered.L[i, j] < -1e-8) if i != j else 0,
                "C_true_diagonal": float(cycle.C[i, i]) if i == j else np.nan,
                "C_recovered_diagonal": float(cycle_recovered.C[i, i]) if i == j else np.nan,
            })

    # Coherent-only negative control: the endpoint spectral measure determines
    # a Jacobi path, so the K8 spatial-profile intervention is redundant there.
    path_system = make_path_system(6, 6286)
    Jpath = normalized_J(path_system)
    eigenvalues, eigenvectors = np.linalg.eigh(Jpath)
    path_from_coherent = recover_jacobi_from_spectral_measure(eigenvalues, eigenvectors[0, :] ** 2)
    summary_rows = [
        {
            "case": "K8_Y_tree_labelled",
            "method": "K8",
            "n": 4,
            "edge_count": 3,
            "cycle_rank": 0,
            "query_or_profile_count": 2,
            "expected_count": 2,
            "J_or_L_relative_error": float(np.linalg.norm(recovered.J - branch.J) / np.linalg.norm(branch.J)),
            "C_relative_error": np.nan,
            "coherent_alias_max_abs": coherent_alias_error,
            "profile_node3_max_abs_difference": float(max(abs(r["profile_power_difference"]) for r in alias_rows)),
            "daughter_swap_J_relative_separation": float(np.linalg.norm(swapped - branch.J) / np.linalg.norm(branch.J)),
            "daughter_swap_remains_positive_passive_model": swapped_is_passive,
            "notes": "known labelled Y tree; synthetic motivated fixture; n-2 profiles",
        },
        {
            "case": "K9_cyclic_general_graph",
            "method": "K9",
            "n": 5,
            "edge_count": 5,
            "cycle_rank": 1,
            "query_or_profile_count": int(cycle_recovered.query_count),
            "expected_count": 5 * 8 // 2,
            "J_or_L_relative_error": float(np.linalg.norm(cycle_recovered.L - cycle.L) / np.linalg.norm(cycle.L)),
            "C_relative_error": float(np.linalg.norm(cycle_recovered.C - cycle.C) / np.linalg.norm(cycle.C)),
            "graph_edge_support_match": bool(recovered_edge_set == true_edge_set),
            "recovered_edge_count": len(recovered_edge_set),
            "coherent_alias_max_abs": np.nan,
            "profile_node3_max_abs_difference": np.nan,
            "notes": "unknown-topology connected graph with one cycle; ideal thresholds",
        },
        {
            "case": "K8_endpoint_path_coherent_only_control",
            "method": "coherent-only control",
            "n": 6,
            "edge_count": 5,
            "cycle_rank": 0,
            "query_or_profile_count": 0,
            "expected_count": 0,
            "J_or_L_relative_error": float(np.linalg.norm(path_from_coherent - Jpath) / np.linalg.norm(Jpath)),
            "C_relative_error": np.nan,
            "coherent_alias_max_abs": np.nan,
            "profile_node3_max_abs_difference": np.nan,
            "notes": "endpoint Weyl spectral measure uniquely reconstructs Jacobi path; profiles redundant",
        },
    ]

    rng = np.random.default_rng(20260929)
    noise_rows: list[dict] = []
    for sigma in (0.0, 1e-10, 1e-8, 1e-6):
        for replicate in range(8):
            h = observations.coherent_root.copy()
            y = observations.psd_differences.copy()
            if sigma:
                h += sigma * np.maximum(np.abs(h), 1e-12) * (
                    rng.normal(size=4) + 1j * rng.normal(size=4)
                )
                y += sigma * np.maximum(np.abs(y), 1e-12) * rng.normal(size=y.shape)
            noisy = K8Observations(omega, h, profiles, y)
            try:
                exact = reconstruct(noisy, branch.edges, root=0)
                exact_status = "accepted"
                exact_error = float(np.linalg.norm(exact.J - branch.J) / np.linalg.norm(branch.J))
            except (PrecisionFailure, ValueError, np.linalg.LinAlgError) as exc:
                exact_status = type(exc).__name__
                exact_error = np.nan
            Jfit, diagnostics = fit_k8_tree(noisy, branch.C, branch.edges, root=0)
            noise_rows.append({
                "method": "K8",
                "network": "known_labelled_Y_tree",
                "relative_noise_scale": sigma,
                "replicate": replicate,
                "exact_inverse_status": exact_status,
                "exact_relative_parameter_error": exact_error,
                "fit_success": diagnostics["optimizer_success"],
                "fit_relative_parameter_error": float(np.linalg.norm(Jfit - branch.J) / np.linalg.norm(branch.J)),
                "fit_weighted_residual": diagnostics["weighted_residual_norm"],
                "seed": 20260929,
            })

            for sigma_k9 in (sigma,):
                def noisy_threshold(lam, q, retained):
                    exact_value = threshold_from_descriptor(cycle.L, cycle.C, lam, q, retained)
                    sd = sigma_k9 * max(1.0, float(np.linalg.norm(cycle.L, ord=2)))
                    return exact_value + (float(rng.normal(scale=sd)) if sd else 0.0)

                oracle = CountingThresholdOracle(noisy_threshold)
                try:
                    estimate = recover_joint_LC(5, oracle, growth_rate_storage=0.8)
                    k9_status = "accepted"
                    l_error = float(np.linalg.norm(estimate.L - cycle.L) / np.linalg.norm(cycle.L))
                    c_error = float(np.linalg.norm(estimate.C - cycle.C) / np.linalg.norm(cycle.C))
                except (PrecisionFailure, ValueError, np.linalg.LinAlgError) as exc:
                    k9_status = type(exc).__name__
                    l_error = np.nan
                    c_error = np.nan
                noise_rows.append({
                    "method": "K9",
                    "network": "unknown_topology_cyclic_graph",
                    "relative_noise_scale": sigma_k9,
                    "replicate": replicate,
                    "exact_inverse_status": k9_status,
                    "exact_relative_parameter_error": l_error,
                    "C_relative_parameter_error": c_error,
                    "fit_success": False,
                    "fit_relative_parameter_error": np.nan,
                    "fit_weighted_residual": np.nan,
                    "seed": 20260929,
                })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(alias_rows),
        pd.DataFrame(cycle_rows),
        pd.DataFrame(noise_rows),
    )


def run_v1(*, replicates: int = 10) -> dict:
    started = time.perf_counter()
    result_dir = BUILD_ROOT / "results"
    figure_dir = BUILD_ROOT / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    exact, family, scaling = run_exact_recoveries()
    raw_noise, noise_summary = run_noise_trials(replicates=replicates)
    clamp = clamp_boundary_control()
    model_source = BUILD_ROOT / "support" / "SarcomereModel-main" / "src"
    mismatch_rows, mismatch_summary = run_sarcomere_model_mismatch(model_source)
    matrices = _example_matrices()
    arch_summary, branch_alias, cycle_matrices, arch_noise = _architecture_repair_results()

    exact.to_csv(result_dir / "exact_recovery.csv", index=False, float_format="%.12g")
    family.to_csv(result_dir / "synthetic_path_family.csv", index=False, float_format="%.12g")
    scaling.to_csv(result_dir / "scaling.csv", index=False, float_format="%.12g")
    raw_noise.to_csv(result_dir / "noise_replicates.csv", index=False, float_format="%.12g")
    noise_summary.to_csv(result_dir / "noise_summary.csv", index=False, float_format="%.12g")
    clamp.to_csv(result_dir / "clamp_boundary_control.csv", index=False, float_format="%.12g")
    pd.DataFrame(mismatch_rows).to_csv(result_dir / "mismatch_observations.csv", index=False, float_format="%.12g")
    matrices.to_csv(result_dir / "example_matrices.csv", index=False, float_format="%.12g")
    arch_summary.to_csv(result_dir / "architecture_repair_exact.csv", index=False, float_format="%.12g")
    branch_alias.to_csv(result_dir / "branch_alias_frequency.csv", index=False, float_format="%.12g")
    cycle_matrices.to_csv(result_dir / "cyclic_k9_matrices.csv", index=False, float_format="%.12g")
    arch_noise.to_csv(result_dir / "architecture_noise.csv", index=False, float_format="%.12g")
    _write_json(result_dir / "mismatch_summary.json", mismatch_summary)
    make_figures(result_dir, figure_dir)

    summary = {
        "project": "P1.1",
        "package_version": "V1.1-ARCH-REPAIRED",
        "random_seed_policy": "fixed per fixture and replicate; see code/p11/benchmark.py",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "noise_replicates_per_cell": replicates,
        "synthetic_path_chain_lengths": list(range(2, 10)),
        "architecture_repair_cases": arch_summary.to_dict(orient="records"),
        "data_motion_pkl_present_at_phase_0": False,
        "public_data_status": "PUBLIC_DATA_GROUNDING_MODULE_PENDING",
        "sarcomere_model_status": mismatch_summary["recovery_status"],
        "exact_recovery_status_counts": exact["status"].value_counts().to_dict(),
        "finite_noise_trials": int(len(raw_noise)),
        "finite_noise_fit_success_fraction": float(raw_noise["constrained_optimizer_success"].mean()),
        "nonlinear_model_mismatch_psd_rows": int(len(mismatch_rows)),
        "theorem_boundary_controls": [
            "noncyclic root-pole cancellation in K8 hidden-twin star",
            "unlinked K9 component typed refusal",
            "K9 clamped external-edge stiffness retention control",
            "K8 duplicate-frequency contract error",
        ],
        "evidence_types": [
            "exact theorem-class synthetic forward data",
            "finite-noise synthetic numerical trials",
            "public nonlinear SarcomereModel mismatch simulation",
            "public cardiac-branching evidence from literature; graph fixtures remain synthetic",
            "public human dataset described from local article and README only",
        ],
    }
    _write_json(result_dir / "result_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run_v1()
    print(json.dumps(result, indent=2, sort_keys=True))
