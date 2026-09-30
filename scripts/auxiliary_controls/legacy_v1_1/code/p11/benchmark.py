from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .errors import ReconstructionError
from .fixtures import exact_k8_hidden_twin_fixture
from .approximate import fit_k8_path, fit_k9_path_records, record_noisy_k9_schedule
from .k8 import K8Observations, default_profiles, generate_observations, reconstruct
from .k9 import CountingThresholdOracle, recover_joint_LC, threshold_from_descriptor
from .models import make_path_system, normalized_J, normalized_condition
from .scaling import J_to_L


def rel_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1e-15))


def run_exact_recoveries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact_rows: list[dict] = []
    family_rows: list[dict] = []
    scaling_rows: list[dict] = []
    for n in range(2, 10):
        system = make_path_system(n, 5100 + n)
        condition = normalized_condition(system)
        family_rows.append({
            "family": system.label,
            "n": n,
            "local_tangent_min": float(np.min(system.local_tangent)),
            "local_tangent_max": float(np.max(system.local_tangent)),
            "coupling_min": float(np.min(system.coupling)),
            "coupling_max": float(np.max(system.coupling)),
            "storage_min": float(np.min(np.diag(system.C))),
            "storage_max": float(np.max(np.diag(system.C))),
            "normalized_condition_number": condition,
            "parameter_scale": "dimensionless synthetic values; not data estimates",
        })
        edge_list = [(i, i + 1) for i in range(n - 1)]
        J = normalized_J(system)
        omega = np.linspace(0.35, 1.75, n)
        t0 = time.perf_counter()
        try:
            obs = generate_observations(J, edge_list, omega, default_profiles(n))
            rec = reconstruct(obs, edge_list)
            Lhat = J_to_L(rec.J, system.C)
            k8_status = "exact_contract_recovered"
            k8_error = rel_frobenius(Lhat, system.L)
            k8_fit = rec.coherent_fit_error
            k8_profile_fit = rec.profile_fit_error
            k8_runtime = time.perf_counter() - t0
        except Exception as exc:
            k8_status = f"{type(exc).__name__}: {exc}"
            k8_error = k8_fit = k8_profile_fit = np.nan
            k8_runtime = time.perf_counter() - t0
        d = n * (n + 1) // 2
        p = n * (n + 3) // 2
        scaling_rows.append({
            "n": n,
            "condition_number": condition,
            "K8_noise_profiles": n - 2,
            "K8_real_scalar_values_accounted": n * n,
            "K8_profile_budget_theorem": n - 2,
            "K8_runtime_seconds": k8_runtime,
            "K8_numeric_status": k8_status,
            "K9_fixed_growth_thresholds_d": d,
            "K9_joint_thresholds_d_plus_n": p,
        })
        exact_rows.append({
            "method": "K8",
            "case": "path",
            "n": n,
            "condition_number": condition,
            "relative_error_target": "L given calibrated C",
            "relative_frobenius_error": k8_error,
            "coherent_fit_error": k8_fit,
            "profile_fit_error": k8_profile_fit,
            "query_count": np.nan,
            "runtime_seconds": k8_runtime,
            "status": k8_status,
        })
        t0 = time.perf_counter()
        try:
            oracle = None
            oracle = CountingThresholdOracle(
                lambda lam, q, retained, s=system: threshold_from_descriptor(s.L, s.C, lam, q, retained)
            )
            rec9 = recover_joint_LC(n, oracle, growth_rate_storage=0.8)
            k9_status = "exact_contract_recovered"
            k9_error = float(np.sqrt(
                np.linalg.norm(rec9.L - system.L) ** 2 + np.linalg.norm(rec9.C - system.C) ** 2
            ) / np.sqrt(np.linalg.norm(system.L) ** 2 + np.linalg.norm(system.C) ** 2))
            k9_queries = rec9.query_count
        except Exception as exc:
            k9_status = f"{type(exc).__name__}: {exc}"
            k9_error, k9_queries = np.nan, getattr(oracle, "query_count", np.nan)
        runtime = time.perf_counter() - t0
        exact_rows.append({
            "method": "K9",
            "case": "path",
            "n": n,
            "condition_number": condition,
            "relative_error_target": "joint (L,C)",
            "relative_frobenius_error": k9_error,
            "coherent_fit_error": np.nan,
            "profile_fit_error": np.nan,
            "query_count": k9_queries,
            "runtime_seconds": runtime,
            "status": k9_status,
        })
    J, edges, obs = exact_k8_hidden_twin_fixture()
    t0 = time.perf_counter()
    twin = reconstruct(obs, edges)
    exact_rows.append({
        "method": "K8",
        "case": "hidden_twin_star_noncyclic_root",
        "n": 4,
        "condition_number": float(np.linalg.cond(-J)),
        "relative_error_target": "J with a cancelled root pole",
        "relative_frobenius_error": rel_frobenius(twin.J, J),
        "coherent_fit_error": twin.coherent_fit_error,
        "profile_fit_error": twin.profile_fit_error,
        "query_count": np.nan,
        "runtime_seconds": time.perf_counter() - t0,
        "status": f"exact_contract_recovered; reduced_denominator_degree={len(twin.reduced_denominator)-1}",
    })
    near = make_path_system(7, 7717, near_ill_conditioned=True)
    Jnear = normalized_J(near)
    omega = np.linspace(0.35, 1.75, near.n)
    t0 = time.perf_counter()
    try:
        obs = generate_observations(Jnear, [(i, i + 1) for i in range(near.n - 1)], omega)
        rec = reconstruct(obs, [(i, i + 1) for i in range(near.n - 1)])
        near_status, near_error = "exact_contract_recovered", rel_frobenius(rec.J, Jnear)
    except Exception as exc:
        near_status, near_error = f"{type(exc).__name__}: {exc}", np.nan
    exact_rows.append({
        "method": "K8",
        "case": "near_ill_conditioned_path",
        "n": near.n,
        "condition_number": normalized_condition(near),
        "relative_error_target": "J given calibrated C",
        "relative_frobenius_error": near_error,
        "coherent_fit_error": np.nan,
        "profile_fit_error": np.nan,
        "query_count": np.nan,
        "runtime_seconds": time.perf_counter() - t0,
        "status": near_status,
    })
    return pd.DataFrame(exact_rows), pd.DataFrame(family_rows), pd.DataFrame(scaling_rows)


def summarize_noise_records(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["exact_inverse_accepted"] = raw["exact_relative_error"].notna() & raw["exact_inverse_status"].eq("exact_inverse_recovered")
    raw["fit_success"] = raw["constrained_fit_relative_error"].notna() & raw["constrained_optimizer_success"]
    raw["successful_fit_error"] = raw["constrained_fit_relative_error"].where(raw["fit_success"])
    return (
        raw.groupby(["method", "family", "n", "condition_number", "relative_noise_scale"], as_index=False)
        .agg(
            trials=("replicate", "count"),
            exact_inverse_acceptance_fraction=("exact_inverse_accepted", "mean"),
            finite_fit_success_fraction=("fit_success", "mean"),
            median_exact_inverse_error=("exact_relative_error", "median"),
            median_finite_fit_error=("successful_fit_error", "median"),
            p90_finite_fit_error=("successful_fit_error", lambda x: x.quantile(0.9)),
        )
    )


def run_noise_trials(*, replicates: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    levels = [0.0, 1e-10, 1e-8, 1e-6]
    families = [
        make_path_system(6, 6206),
        make_path_system(6, 7717, near_ill_conditioned=True),
    ]
    for family_index, system in enumerate(families):
        family_name = "near_ill_conditioned" if family_index else "moderate_condition"
        n = system.n
        edges = [(i, i + 1) for i in range(n - 1)]
        omega = np.linspace(0.35, 1.75, n)
        J = normalized_J(system)
        base = generate_observations(J, edges, omega, default_profiles(n))
        max_psd = max(float(np.max(np.abs(base.psd_differences))), 1e-12)
        for method in ("K8", "K9"):
            for level in levels:
                for replicate in range(replicates):
                    rng = np.random.default_rng(990000 + 1000 * family_index + int(level * 1e9) + replicate)
                    exact_error = np.nan
                    exact_status = "not_run"
                    approximate_error = np.nan
                    approximate_status = "not_run"
                    approximate_fit_ok = False
                    diagnostics = {}
                    queries = np.nan
                    try:
                        if method == "K8":
                            noisy = K8Observations(
                                base.frequencies,
                                base.coherent_root + level * max(float(np.max(np.abs(base.coherent_root))), 1e-12)
                                * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2.0),
                                base.profiles,
                                base.psd_differences + rng.normal(0.0, level * max_psd, base.psd_differences.shape),
                            )
                            try:
                                rec = reconstruct(noisy, edges)
                                L_exact = J_to_L(rec.J, system.C)
                                exact_error = rel_frobenius(L_exact, system.L)
                                exact_status = "exact_inverse_recovered"
                            except ReconstructionError as exc:
                                exact_status = type(exc).__name__
                            Lhat, diagnostics = fit_k8_path(noisy, system.C)
                            approximate_error = rel_frobenius(Lhat, system.L)
                            approximate_status = "constrained_path_fit"
                            approximate_fit_ok = bool(diagnostics["optimizer_success"])
                            queries = n * n
                        else:
                            sigma = level * max(1.0, float(np.linalg.norm(system.L, ord=2)))
                            schedule = record_noisy_k9_schedule(
                                system.L,
                                system.C,
                                growth_rate_storage=0.8,
                                rng=rng,
                                noise_sd=sigma,
                            )
                            approximate_status = schedule.design_status
                            if schedule.design_status == "complete":
                                Lhat, Chat, diagnostics = fit_k9_path_records(n, schedule.records)
                                approximate_error = float(np.sqrt(
                                    np.linalg.norm(Lhat - system.L) ** 2 + np.linalg.norm(Chat - system.C) ** 2
                                ) / np.sqrt(np.linalg.norm(system.L) ** 2 + np.linalg.norm(system.C) ** 2))
                                approximate_fit_ok = bool(diagnostics["optimizer_success"])
                                approximate_status = "constrained_path_fit"
                            noisy_rng = np.random.default_rng(1990000 + 1000 * family_index + int(level * 1e9) + replicate)
                            oracle = CountingThresholdOracle(
                                lambda lam, q, retained, s=system, r=rng, sd=sigma:
                                    threshold_from_descriptor(s.L, s.C, lam, q, retained)
                                    + float(noisy_rng.normal(0.0, sd))
                            )
                            try:
                                rec = recover_joint_LC(n, oracle, growth_rate_storage=0.8)
                                exact_status = "exact_inverse_recovered"
                                exact_error = float(np.sqrt(
                                    np.linalg.norm(rec.L - system.L) ** 2 + np.linalg.norm(rec.C - system.C) ** 2
                                ) / np.sqrt(np.linalg.norm(system.L) ** 2 + np.linalg.norm(system.C) ** 2))
                            except ReconstructionError as exc:
                                exact_status = type(exc).__name__
                            queries = oracle.query_count
                        status = "trial_complete"
                        error_text = ""
                    except ReconstructionError as exc:
                        status = type(exc).__name__
                        error_text = str(exc)
                    except Exception as exc:
                        status = f"unexpected_{type(exc).__name__}"
                        error_text = str(exc)
                    rows.append({
                        "method": method,
                        "family": family_name,
                        "n": n,
                        "condition_number": normalized_condition(system),
                        "relative_noise_scale": level,
                        "replicate": replicate,
                        "exact_relative_error": exact_error,
                        "exact_inverse_status": exact_status,
                        "constrained_fit_relative_error": approximate_error,
                        "constrained_fit_status": approximate_status,
                        "constrained_optimizer_success": approximate_fit_ok,
                        "fit_normalized_residual": diagnostics.get("normalized_residual_norm", np.nan),
                        "fit_function_evaluations": diagnostics.get("nfev", np.nan),
                        "query_count": queries,
                        "status": status,
                        "failure_detail": error_text,
                        "replicate_interpretation": "numerical noise draw; not a biological sample",
                    })
    raw = pd.DataFrame(rows)
    return raw, summarize_noise_records(raw)


def clamp_boundary_control() -> pd.DataFrame:
    """Quantify the error from wrongly deleting clamped-edge stiffness terms."""
    system = make_path_system(4, 8304)
    retained = (0, 1)
    q = np.zeros(system.n)
    correct = threshold_from_descriptor(system.L, system.C, 0.0, q, retained)
    # Deliberately incorrect reduction: keep only local tangent stiffness and
    # edges internal to the retained set, dropping diagonal terms from edges
    # to clamped states.
    wrong = np.diag(system.local_tangent[:2]).copy()
    k = system.coupling[0]
    wrong += np.array([[k, -k], [-k, k]])
    wrong_threshold = float(wrong[0, 0] - wrong[0, 1] ** 2 / wrong[1, 1])
    return pd.DataFrame([{
        "retained_nodes": "0,1",
        "correct_principal_matrix_threshold": correct,
        "incorrect_deleted_edge_threshold": wrong_threshold,
        "absolute_threshold_shift": abs(correct - wrong_threshold),
        "semantics": "clamped external-edge stiffness remains on retained diagonal",
    }])
