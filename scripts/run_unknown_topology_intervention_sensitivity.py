"""Illustrative finite-intervention and measurement-error unknown-topology sweeps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_practical_unknown_topology_validation import (  # noqa: E402
    load_contexts, make_reference_case, make_schedule, make_reader, score,
)
from practical_estimator.estimator import fit_all_candidate_edges  # noqa: E402


def run() -> None:
    ctx = next(c for c in load_contexts() if c["movie"] == "E3" and c["n"] == 5)
    n, root = int(ctx["n"]), int(ctx["root_index"])
    L, C, edges, lengths, *_ = make_reference_case(ctx, np.random.default_rng(2026092901))
    base_schedule = make_schedule(n, root, "high_redundancy")
    scenarios = [
        ("ideal_control", {}, 1),
        ("shunt_bias_minus_15pct", {"q_error": -0.15}, 1),
        ("shunt_bias_minus_5pct", {"q_error": -0.05}, 1),
        ("shunt_bias_plus_5pct", {"q_error": 0.05}, 1),
        ("shunt_bias_plus_15pct", {"q_error": 0.15}, 1),
        ("finite_clamp_50x_median_diagonal", {"finite_clamp_penalty": 50 * float(np.median(np.diag(L)))}, 1),
        ("quantization_0p1pct_baseline", {"quantization_fraction": 1e-3}, 1),
        ("quantization_0p5pct_baseline", {"quantization_fraction": 5e-3}, 1),
        ("root_gain_plus_5pct", {"root_gain": 1.05}, 1),
        ("repeated_all_queries_x3", {}, 3),
    ]
    rows = []
    for scenario_index, (name, imperfections, repeat_factor) in enumerate(scenarios):
        for replicate in range(5):
            seed = 2026092902 + scenario_index * 100 + replicate
            rng = np.random.default_rng(seed)
            reader, records = make_reader(L, C, root, 1e-3, rng, **imperfections)
            schedule = [row for row in base_schedule for _ in range(repeat_factor)]
            for row in schedule:
                reader(row["growth_rate"], row["q"], row["retained"])
            try:
                fit = fit_all_candidate_edges(n, root, records)
                metrics = score(fit.L, fit.C, fit.edge_weights, fit.candidate_edges, L, C, edges)
                status, success, residual, condition = fit.message, fit.success, fit.relative_residual, fit.jacobian_condition
            except Exception as exc:
                metrics = {key: np.nan for key in ("L_relative_error", "C_relative_error", "topology_precision", "topology_recall", "topology_F1")}
                status, success, residual, condition = f"{type(exc).__name__}: {exc}", False, np.nan, np.nan
            rows.append({"movie": ctx["movie"], "n": n, "scenario": name, "replicate": replicate,
                         "noise_fraction": 1e-3, "query_count": len(records), "repeat_factor": repeat_factor,
                         "q_error": imperfections.get("q_error", 0.0),
                         "root_gain": imperfections.get("root_gain", 1.0),
                         "finite_clamp_penalty": imperfections.get("finite_clamp_penalty", np.nan),
                         "quantization_fraction": imperfections.get("quantization_fraction", 0.0),
                         "fit_success": bool(success), "fit_residual": residual,
                         "jacobian_condition": condition, "status": status, **metrics})
        print(f"completed {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "k9_intervention_imperfections.csv", index=False, float_format="%.10g")
    summary = df.groupby("scenario").agg(
        cases=("replicate", "count"), failure_rate=("fit_success", lambda x: 1 - x.mean()),
        queries_median=("query_count", "median"), L_error_median=("L_relative_error", "median"),
        C_error_median=("C_relative_error", "median"), topology_F1_median=("topology_F1", "median"),
        residual_median=("fit_residual", "median"), condition_median=("jacobian_condition", "median")
    ).reset_index()
    summary.to_csv(ROOT / "results" / "k9_intervention_imperfections_summary.csv", index=False, float_format="%.10g")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
