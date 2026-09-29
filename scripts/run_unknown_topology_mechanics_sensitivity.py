"""High-redundancy unknown-topology checks across morphology and effective-mechanics ensembles."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_practical_unknown_topology_validation import (  # noqa: E402
    incidence_matrix, load_contexts, make_schedule, make_reader, score,
)
from practical_estimator.estimator import fit_all_candidate_edges  # noqa: E402


ENSEMBLES = {
    "reference": {"ratio": 5.0, "edge_scale": 1.0, "local_scale": 1.0, "storage_scale": 1.0, "sigma": 0.08, "mapping": "inverse_length"},
    "low_stiffness": {"ratio": 5.0, "edge_scale": 0.5, "local_scale": 0.5, "storage_scale": 1.0, "sigma": 0.08, "mapping": "inverse_length"},
    "high_stiffness": {"ratio": 5.0, "edge_scale": 2.0, "local_scale": 2.0, "storage_scale": 1.0, "sigma": 0.08, "mapping": "inverse_length"},
    "low_storage": {"ratio": 5.0, "edge_scale": 1.0, "local_scale": 1.0, "storage_scale": 0.5, "sigma": 0.08, "mapping": "inverse_length"},
    "high_storage": {"ratio": 5.0, "edge_scale": 1.0, "local_scale": 1.0, "storage_scale": 1.5, "sigma": 0.08, "mapping": "inverse_length"},
    "low_edge_ratio": {"ratio": 1.25, "edge_scale": 1.0, "local_scale": 1.0, "storage_scale": 1.0, "sigma": 0.08, "mapping": "inverse_length"},
    "high_edge_ratio": {"ratio": 10.0, "edge_scale": 1.0, "local_scale": 1.0, "storage_scale": 1.0, "sigma": 0.08, "mapping": "inverse_length"},
    "heterogeneous_edges": {"ratio": 5.0, "edge_scale": 1.0, "local_scale": 1.0, "storage_scale": 1.0, "sigma": 0.35, "mapping": "inverse_length"},
    "constant_edge_mapping": {"ratio": 5.0, "edge_scale": 1.0, "local_scale": 1.0, "storage_scale": 1.0, "sigma": 0.08, "mapping": "constant"},
}


def make_case(ctx, cfg, seed):
    n = int(ctx["n"])
    edges = [tuple(map(int, e)) for e in ctx["edges"]]
    xy = np.asarray(ctx["node_positions_px"], dtype=float)
    lengths = np.array([max(float(np.linalg.norm(xy[u] - xy[v])), 1e-6) for u, v in edges])
    median_length = float(np.median(lengths))
    rng = np.random.default_rng(seed)
    local = np.clip(rng.normal(16.0, 1.3, size=n) / 16.0, 0.65, 1.35) * cfg["local_scale"]
    if cfg["mapping"] == "inverse_length":
        coupling = cfg["ratio"] * median_length / lengths
    else:
        coupling = np.full(len(edges), cfg["ratio"])
    coupling *= cfg["edge_scale"] * rng.lognormal(mean=0.0, sigma=cfg["sigma"], size=len(edges))
    storage = rng.uniform(0.8, 1.2, size=n) * cfg["storage_scale"]
    B = incidence_matrix(n, edges)
    L = np.diag(local) + B @ np.diag(coupling) @ B.T
    C = np.diag(storage)
    return L, C, edges, lengths, local, coupling, storage


def main():
    contexts = [c for c in load_contexts() if c["n"] == 5]
    rows = []
    schedule_cache = {}
    for context_index, ctx in enumerate(contexts):
        n, root = int(ctx["n"]), int(ctx["root_index"])
        schedule = schedule_cache.setdefault((n, root), make_schedule(n, root, "high_redundancy"))
        for ensemble_index, (name, cfg) in enumerate(ENSEMBLES.items()):
            for draw in range(4):
                parameter_seed = 20261000 + context_index * 10000 + ensemble_index * 100 + draw
                L, C, edges, lengths, local, coupling, storage = make_case(ctx, cfg, parameter_seed)
                for replicate in range(3):
                    seed = parameter_seed + 1000 + replicate
                    reader, records = make_reader(L, C, root, 1e-3, np.random.default_rng(seed))
                    for query in schedule:
                        reader(query["growth_rate"], query["q"], query["retained"])
                    try:
                        fit = fit_all_candidate_edges(n, root, records)
                        metrics = score(fit.L, fit.C, fit.edge_weights, fit.candidate_edges, L, C, edges)
                        status, success = fit.message, fit.success
                        residual, condition, nfev = fit.relative_residual, fit.jacobian_condition, fit.nfev
                    except Exception as exc:
                        metrics = {key: np.nan for key in ("L_relative_error", "C_relative_error", "topology_precision", "topology_recall", "topology_F1")}
                        status, success, residual, condition, nfev = f"{type(exc).__name__}: {exc}", False, np.nan, np.nan, np.nan
                    rows.append({"movie": ctx["movie"], "n": n, "cycle_rank": int(ctx["cycle_rank"]),
                                 "ensemble": name, "draw": draw, "replicate": replicate,
                                 "ratio": cfg["ratio"], "local_scale": cfg["local_scale"],
                                 "edge_scale": cfg["edge_scale"], "storage_scale": cfg["storage_scale"],
                                 "heterogeneity_sigma": cfg["sigma"], "edge_mapping": cfg["mapping"],
                                 "noise_fraction": 1e-3, "query_count": len(records), "status": status,
                                 "fit_success": success, "fit_residual": residual,
                                 "jacobian_condition": condition, "nfev": nfev, **metrics})
            print(f"completed {ctx['movie']} {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "k9_mechanics_sensitivity.csv", index=False, float_format="%.10g")
    summary = df.groupby("ensemble").agg(
        cases=("replicate", "count"), failure_rate=("fit_success", lambda x: 1 - x.mean()),
        query_count=("query_count", "median"), L_error_median=("L_relative_error", "median"),
        L_error_p90=("L_relative_error", lambda x: x.quantile(.9)), C_error_median=("C_relative_error", "median"),
        C_error_p90=("C_relative_error", lambda x: x.quantile(.9)), topology_F1_median=("topology_F1", "median"),
        residual_median=("fit_residual", "median"), condition_median=("jacobian_condition", "median")
    ).reset_index()
    summary.to_csv(ROOT / "results" / "k9_mechanics_sensitivity_summary.csv", index=False, float_format="%.10g")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
