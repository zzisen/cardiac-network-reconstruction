"""Run the frozen no-oracle unknown-topology estimator on real-morphology subgraphs."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from p11.unknown_topology import recover_unknown_topology
from practical_estimator.estimator import (
    SUPPORT_THRESHOLD,
    descriptor_from_parameters,
    fit_all_candidate_edges,
    inferred_support,
    threshold_finite_clamp,
    threshold_schur,
)


NOISE_LEVELS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3)
DESIGNS = ("minimal_exact_budget", "moderate_redundancy", "high_redundancy")


def incidence_matrix(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    B = np.zeros((n, len(edges)), dtype=float)
    for k, (u, v) in enumerate(edges):
        B[u, k] = -1.0
        B[v, k] = 1.0
    return B


def load_contexts() -> list[dict]:
    selected = json.loads((ROOT / "results" / "public_subgraph_selection.json").read_text(encoding="utf-8"))
    contexts = []
    for movie in ("E3", "E4", "E5"):
        for n in (4, 5, 6):
            item = selected["graphs"][movie]["k9"].get(str(n))
            if item is not None:
                contexts.append({"movie": movie, "n": n, **item})
    return contexts


def make_reference_case(ctx: dict, rng: np.random.Generator):
    n = int(ctx["n"])
    edges = [tuple(map(int, e)) for e in ctx["edges"]]
    xy = np.asarray(ctx["node_positions_px"], dtype=float)
    lengths = np.array([max(float(np.linalg.norm(xy[u] - xy[v])), 1e-6) for u, v in edges])
    median_length = float(np.median(lengths)) if len(lengths) else 1.0
    # Z-disc transverse values are normalized by the adult-rat reported 16.0 pN/nm mean.
    # The ±1.3 term is sampled only as an illustrative source uncertainty, not as a population SD.
    grounding = np.clip(rng.normal(16.0, 1.3, size=n) / 16.0, 0.65, 1.35)
    # The reference edge ratio (k_MZ/k_TT=0.5/0.1=5) is an explicit dimensionless
    # coarse-graining assumption; edge-length inverse scaling is tested separately.
    coupling = 5.0 * median_length / lengths * rng.lognormal(mean=0.0, sigma=0.08, size=len(edges))
    storage = rng.uniform(0.8, 1.2, size=n)  # Dimensionless sensitivity range; not experimentally calibrated.
    B = incidence_matrix(n, edges)
    L = np.diag(grounding) + B @ np.diag(coupling) @ B.T
    C = np.diag(storage)
    return L, C, edges, lengths, grounding, coupling, storage


def make_schedule(n: int, root: int, design: str) -> list[dict]:
    if design == "minimal_exact_budget":
        return []  # Adaptive theorem schedule is executed by recover_unknown_topology below.
    if design == "moderate_redundancy":
        singles, pairs = (0.5, 2.0), (1.0,)
    elif design == "high_redundancy":
        singles, pairs = (0.5, 2.0, 10.0, 50.0), (0.5, 2.0, 10.0)
    else:
        raise ValueError(design)
    hidden = [v for v in range(n) if v != root]
    zero = np.zeros(n, dtype=float)
    rows: list[dict] = []
    for lam in (0.0, 1.0):
        for _ in range(2):  # repeated baselines expose read noise and drift
            rows.append({"growth_rate": lam, "q": zero.copy(), "retained": tuple(range(n))})
        for node in hidden:
            for load in singles:
                q = zero.copy(); q[node] = load
                rows.append({"growth_rate": lam, "q": q, "retained": tuple(range(n))})
        for i in range(len(hidden)):
            for j in range(i + 1, len(hidden)):
                for load in pairs:
                    q = zero.copy(); q[hidden[i]] = q[hidden[j]] = load
                    rows.append({"growth_rate": lam, "q": q, "retained": tuple(range(n))})
    # A fixed morphology-label clamp design: root-only and each root/node pair.
    # It does not depend on the hidden parameter values or graph support.
    rows.append({"growth_rate": 1.0, "q": zero.copy(), "retained": (root,)})
    for node in hidden:
        rows.append({"growth_rate": 1.0, "q": zero.copy(), "retained": (root, node)})
    return rows


def score(Lhat, Chat, weights, candidate_edges, Ltrue, Ctrue, true_edges):
    predicted = {edge for edge, weight in zip(candidate_edges, weights) if weight >= SUPPORT_THRESHOLD}
    true = {tuple(sorted(edge)) for edge in true_edges}
    tp = len(predicted & true); fp = len(predicted - true); fn = len(true - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "L_relative_error": float(np.linalg.norm(Lhat - Ltrue) / np.linalg.norm(Ltrue)),
        "C_relative_error": float(np.linalg.norm(Chat - Ctrue) / np.linalg.norm(Ctrue)),
        "topology_precision": precision,
        "topology_recall": recall,
        "topology_F1": f1,
    }


def make_reader(L, C, root, noise_level, rng, *, q_error=0.0, root_gain=1.0,
                finite_clamp_penalty=None, quantization_fraction=0.0):
    n = len(L)
    baseline_scale = max(abs(threshold_schur(L, C, 0.0, np.zeros(n), None, root)), 1e-10)
    reads = []

    def read(lam, q, retained=None):
        q_actual = np.asarray(q, dtype=float).copy() * (1.0 + q_error)
        if retained is not None and finite_clamp_penalty is not None:
            y = threshold_finite_clamp(L, C, lam, q_actual, retained, root, finite_clamp_penalty)
        else:
            y = threshold_schur(L, C, lam, q_actual, retained, root)
        y *= root_gain
        y *= 1.0 + float(rng.normal(0.0, noise_level))
        if quantization_fraction > 0:
            quantum = quantization_fraction * baseline_scale
            y = float(np.round(y / quantum) * quantum)
        keep = tuple(range(n)) if retained is None else tuple(retained)
        row = {"growth_rate": float(lam), "q": np.asarray(q, dtype=float).copy(),
               "retained": keep, "observed_threshold": float(y)}
        reads.append(row)
        return float(y)

    return read, reads


def run_exact(L, C, root, noise_level, seed):
    rng = np.random.default_rng(seed)
    reader, reads = make_reader(L, C, root, noise_level, rng)
    try:
        result = recover_unknown_topology(len(L), reader, root=root, growth_rate_storage=1.0, load=1.0)
        return result.L, result.C, len(reads), "recovered", True, None, None
    except Exception as exc:  # refusal/failure is an observed result
        return None, None, len(reads), f"{type(exc).__name__}: {exc}", False, None, None


def run_redundant(L, C, root, ctx, design, noise_level, seed, *, q_error=0.0,
                  root_gain=1.0, finite_clamp_penalty=None, quantization_fraction=0.0):
    rng = np.random.default_rng(seed)
    reader, records = make_reader(L, C, root, noise_level, rng, q_error=q_error, root_gain=root_gain,
                                  finite_clamp_penalty=finite_clamp_penalty,
                                  quantization_fraction=quantization_fraction)
    schedule = make_schedule(len(L), root, design)
    for row in schedule:
        reader(row["growth_rate"], row["q"], row["retained"])
    try:
        fit = fit_all_candidate_edges(len(L), root, records)
        return (fit.L, fit.C, len(records), "fit_complete" if fit.success else "optimizer_not_converged",
                bool(fit.success), fit.relative_residual, fit.jacobian_condition, fit.edge_weights,
                fit.candidate_edges, fit.normalized_residual, fit.nfev)
    except Exception as exc:
        return None, None, len(records), f"{type(exc).__name__}: {exc}", False, None, None, None, None, None, None


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    numeric = ["L_relative_error", "C_relative_error", "topology_precision", "topology_recall",
               "topology_F1", "fit_residual", "jacobian_condition", "query_count"]
    rows = []
    for (design, noise), group in df.groupby(["design", "noise_fraction"], dropna=False):
        row = {"design": design, "noise_fraction": noise, "cases": len(group),
               "failure_rate": float(1.0 - group.fit_success.mean()),
               "median_queries": float(group.query_count.median())}
        for field in numeric:
            row[field + "_median"] = float(group[field].median()) if field in group else float("nan")
            row[field + "_p90"] = float(group[field].quantile(0.9)) if field in group else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()
    contexts = load_contexts()
    out_rows = []
    param_rows = []
    context_idx = 0
    for ctx in contexts:
        n = int(ctx["n"]); root = int(ctx["root_index"])
        # Full five-level, multi-draw validation is concentrated on one n=5
        # local graph from each movie. The n=4/n=6 windows are dimensional
        # stress controls at zero and 0.1% noise, keeping the suite bounded.
        is_primary_size = n == 5
        draw_ids = (0, 1) if is_primary_size else (0,)
        noise_levels = NOISE_LEVELS if is_primary_size else (0.0, 1e-3)
        replicates = 4 if is_primary_size else 1
        for draw_id in draw_ids:
            param_seed = 20260929 + context_idx * 1000 + draw_id * 17
            L, C, edges, lengths, grounding, coupling, storage = make_reference_case(
                ctx, np.random.default_rng(param_seed))
            for node, val in enumerate(grounding):
                param_rows.append({"movie": ctx["movie"], "n": n, "draw_id": draw_id,
                                   "parameter": "local_grounding_normalized_to_16_pN_per_nm",
                                   "index": node, "value": float(val)})
            for (u, v), ell, val in zip(edges, lengths, coupling):
                param_rows.append({"movie": ctx["movie"], "n": n, "draw_id": draw_id,
                                   "parameter": "edge_weight_reference_kMZ_over_kTT_length_scaled",
                                   "index": f"{u}-{v}", "value": float(val), "edge_length_px": float(ell)})
            for node, val in enumerate(storage):
                param_rows.append({"movie": ctx["movie"], "n": n, "draw_id": draw_id,
                                   "parameter": "dimensionless_storage_C_not_calibrated",
                                   "index": node, "value": float(val)})
            for noise_idx, noise_level in enumerate(noise_levels):
                for rep in range(replicates):
                    case_seed = param_seed + 10000 * noise_idx + 101 * rep
                    for design in DESIGNS:
                        if design == "minimal_exact_budget":
                            Lhat, Chat, nq, status, ok, fitres, cond = run_exact(
                                L, C, root, noise_level, case_seed)
                            weights = candidate = residual = nfev = None
                        else:
                            (Lhat, Chat, nq, status, ok, fitres, cond, weights, candidate,
                             residual, nfev) = run_redundant(L, C, root, ctx, design, noise_level, case_seed)
                        metrics = {key: np.nan for key in ("L_relative_error", "C_relative_error",
                                                            "topology_precision", "topology_recall", "topology_F1")}
                        if Lhat is not None:
                            candidate = candidate if candidate is not None else [(i, j) for i in range(n) for j in range(i + 1, n)]
                            if weights is None:
                                weights = np.array([-Lhat[u, v] for u, v in candidate])
                            metrics = score(Lhat, Chat, weights, candidate, L, C, edges)
                        out_rows.append({"movie": ctx["movie"], "n": n, "cycle_rank": int(ctx["cycle_rank"]),
                                         "draw_id": draw_id, "replicate": rep, "noise_fraction": noise_level,
                                         "noise_percent": 100 * noise_level, "design": design,
                                         "query_count": int(nq), "status": status, "fit_success": bool(ok),
                                         "fit_residual": fitres, "jacobian_condition": cond,
                                         "optimizer_residual_norm": residual, "nfev": nfev, **metrics})
            print(f"completed {ctx['movie']} n={n} draw={draw_id}", flush=True)
        context_idx += 1

    df = pd.DataFrame(out_rows)
    df.to_csv(ROOT / "results" / "k9_practical_replicates.csv", index=False, float_format="%.10g")
    summarize(df).to_csv(ROOT / "results" / "k9_practical_summary.csv", index=False, float_format="%.10g")
    pd.DataFrame(param_rows).to_csv(ROOT / "results" / "mechanics_reference_draws.csv", index=False, float_format="%.10g")
    receipt = {
        "contexts": len(contexts), "primary_validation": "all three n=5 movie contexts; 2 mechanics draws; all five noise levels; 4 independent measurement replicates",
        "size_controls": "all n=4 and n=6 contexts; 1 mechanics draw; noise 0 and 0.1%; 1 measurement replicate",
        "noise_levels_primary": list(NOISE_LEVELS), "noise_replicates_primary": 4,
        "noise_levels_size_controls": [0.0, 1e-3], "noise_replicates_size_controls": 1,
        "designs": list(DESIGNS), "support_threshold_normalized": SUPPORT_THRESHOLD,
        "runtime_seconds": time.perf_counter() - start,
        "candidate_support_rule": "all undirected pairs fitted; edge present when nonnegative fitted weight >= 0.25 normalized stiffness units, fixed before final suite",
        "root_policy": "morphology-only proximal major-axis node, reindexed to 0",
    }
    (ROOT / "results" / "k9_practical_run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    main()
