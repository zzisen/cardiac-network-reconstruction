"""Re-extract fitted edge weights for the frozen n=5 unknown-topology cases and rescore a fixed cutoff grid.

The V2.1 replicate CSV stores topology metrics at 0.25 but not fitted edge weights.
This script deterministically replays only the primary n=5 cases with the original
parameter/noise seeds, runs the same frozen estimator, and evaluates all five
predeclared support cutoffs. It does not tune the estimator or generate new draws.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_practical_unknown_topology_validation import (  # noqa: E402
    NOISE_LEVELS,
    make_reference_case,
    load_contexts,
    run_exact,
    run_redundant,
)

CUTOFFS = (0.15, 0.20, 0.25, 0.30, 0.35)


def support_scores(candidate_edges, weights, true_edges, cutoff):
    predicted = {tuple(sorted(edge)) for edge, weight in zip(candidate_edges, weights)
                 if float(weight) >= cutoff}
    truth = {tuple(sorted(edge)) for edge in true_edges}
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    started = time.perf_counter()
    all_contexts = load_contexts()
    contexts = [(i, ctx) for i, ctx in enumerate(all_contexts) if int(ctx["n"]) == 5]
    baseline = pd.read_csv(ROOT / "results" / "k9_practical_replicates.csv")
    baseline = baseline[baseline.n == 5].copy()
    expected_cases = 3 * 2 * len(NOISE_LEVELS) * 4 * 3
    if len(contexts) != 3 or len(baseline) != expected_cases:
        raise RuntimeError(f"Expected three primary n=5 graphs and {expected_cases} stored design rows.")
    original = {}
    for row in baseline.itertuples(index=False):
        key = (row.movie, int(row.n), int(row.draw_id), float(row.noise_fraction),
               int(row.replicate), row.design)
        original[key] = row

    replicate_rows = []
    prediction_rows = []
    replayed = 0
    cutoff_025_diffs = []
    for context_index, ctx in contexts:
        n = int(ctx["n"])
        root = int(ctx["root_index"])
        true_edges = [tuple(map(int, edge)) for edge in ctx["edges"]]
        candidate_all = list(itertools.combinations(range(n), 2))
        draw_ids = (0, 1)
        for draw_id in draw_ids:
            parameter_seed = 20260929 + context_index * 1000 + draw_id * 17
            L, C, _, _, _, _, _ = make_reference_case(ctx, np.random.default_rng(parameter_seed))
            for noise_index, noise in enumerate(NOISE_LEVELS):
                for rep in range(4):
                    case_seed = parameter_seed + 10000 * noise_index + 101 * rep
                    for design in ("minimal_exact_budget", "moderate_redundancy", "high_redundancy"):
                        if design == "minimal_exact_budget":
                            Lhat, Chat, query_count, status, ok, _, _ = run_exact(
                                L, C, root, noise, case_seed)
                            if Lhat is None:
                                weights = None
                                candidates = candidate_all
                            else:
                                candidates = candidate_all
                                weights = np.asarray([-Lhat[u, v] for u, v in candidates], dtype=float)
                        else:
                            (Lhat, Chat, query_count, status, ok, _, _, weights, candidates,
                             _, _) = run_redundant(L, C, root, ctx, design, noise, case_seed)
                            if weights is not None:
                                weights = np.asarray(weights, dtype=float)
                        key = (ctx["movie"], n, draw_id, float(noise), rep, design)
                        prior = original.get(key)
                        if prior is None:
                            raise RuntimeError(f"No baseline case for {key}")
                        replayed += 1
                        has_prediction = Lhat is not None and weights is not None
                        if bool(prior.fit_success) != bool(ok):
                            raise RuntimeError(f"Fit status mismatch on deterministic replay: {key}")
                        if int(prior.query_count) != int(query_count):
                            raise RuntimeError(f"Query-count mismatch on deterministic replay: {key}")
                        for cutoff in CUTOFFS:
                            if has_prediction:
                                precision, recall, f1 = support_scores(candidates, weights, true_edges, cutoff)
                            else:
                                precision = recall = f1 = np.nan
                            replicate_rows.append({
                                "movie": ctx["movie"], "n": n, "cycle_rank": int(ctx["cycle_rank"]),
                                "draw_id": draw_id, "replicate": rep, "noise_fraction": noise,
                                "noise_percent": 100 * noise, "design": design,
                                "query_count": int(query_count), "cutoff": cutoff,
                                "fit_success": bool(ok), "status": status,
                                "prediction_available": bool(has_prediction),
                                "topology_precision": precision, "topology_recall": recall,
                                "topology_F1": f1,
                            })
                        if has_prediction:
                            for edge, weight in zip(candidates, weights):
                                prediction_rows.append({
                                    "movie": ctx["movie"], "n": n, "draw_id": draw_id,
                                    "replicate": rep, "noise_fraction": noise,
                                    "design": design, "query_count": int(query_count),
                                    "edge_u": int(edge[0]), "edge_v": int(edge[1]),
                                    "true_edge": tuple(sorted(edge)) in {tuple(sorted(x)) for x in true_edges},
                                    "fitted_weight": float(weight),
                                })
                        base_f1 = float(prior.topology_F1) if pd.notna(prior.topology_F1) else np.nan
                        replay_f1 = (support_scores(candidates, weights, true_edges, 0.25)[2]
                                     if has_prediction else np.nan)
                        if np.isfinite(base_f1) and np.isfinite(replay_f1):
                            cutoff_025_diffs.append(abs(base_f1 - replay_f1))
                        elif np.isfinite(base_f1) != np.isfinite(replay_f1):
                            raise RuntimeError(f"0.25 score availability mismatch on {key}")
                        if design == "high_redundancy":
                            print(f"replayed {ctx['movie']} n=5 draw={draw_id} noise={noise:g} replicate={rep}", flush=True)
            print(f"replayed {ctx['movie']} n=5 draw={draw_id}", flush=True)

    reps = pd.DataFrame(replicate_rows)
    predictions = pd.DataFrame(prediction_rows)
    expected_replay = 3 * 2 * len(NOISE_LEVELS) * 4 * 3
    if replayed != expected_replay:
        raise RuntimeError(f"Expected {expected_replay} replayed rows, got {replayed}.")
    max_diff = max(cutoff_025_diffs, default=0.0)
    if max_diff > 1e-9:
        raise RuntimeError(f"Replay at frozen 0.25 cutoff does not reproduce stored F1 (max diff {max_diff}).")

    summary_rows = []
    for (design, noise, cutoff), group in reps.groupby(["design", "noise_fraction", "cutoff"], sort=True):
        scored = group[group.prediction_available]
        row = {
            "design": design, "noise_fraction": float(noise), "noise_percent": 100 * float(noise),
            "cutoff": float(cutoff), "total_cases": int(len(group)),
            "scored_cases": int(len(scored)), "fit_success_cases": int(group.fit_success.sum()),
            "fit_failure_cases": int((~group.fit_success).sum()),
        }
        for metric in ("topology_precision", "topology_recall", "topology_F1"):
            vals = scored[metric].dropna()
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{metric}_q25"] = float(vals.quantile(.25)) if len(vals) else np.nan
            row[f"{metric}_q75"] = float(vals.quantile(.75)) if len(vals) else np.nan
            row[f"{metric}_IQR"] = float(vals.quantile(.75) - vals.quantile(.25)) if len(vals) else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    reps.to_csv(ROOT / "results" / "k9_support_cutoff_replicates.csv", index=False, float_format="%.10g")
    predictions.to_csv(ROOT / "results" / "k9_support_cutoff_edge_predictions.csv", index=False, float_format="%.10g")
    summary.to_csv(ROOT / "results" / "k9_support_cutoff_summary.csv", index=False, float_format="%.10g")
    receipt = {
        "scope": "primary n=5 practical unknown-topology validation cases across E3/E4/E5",
        "cases_per_design_noise_cutoff": 24,
        "designs": ["minimal_exact_budget", "moderate_redundancy", "high_redundancy"],
        "noise_levels": list(NOISE_LEVELS), "cutoffs": list(CUTOFFS),
        "primary_cutoff_unchanged": 0.25,
        "estimator_changed": False, "new_random_draws": False,
        "stored_edge_weights_were_available": False,
        "method": "deterministically replayed the existing n=5 cases with the original fixed parameter and measurement-noise seeds, extracted fitted edge predictions, then rescored all five cutoffs",
        "replayed_case_design_rows": replayed,
        "weight_prediction_rows": int(len(predictions)),
        "maximum_absolute_F1_difference_at_primary_cutoff": max_diff,
        "runtime_seconds": time.perf_counter() - started,
        "replicate_table": "results/k9_support_cutoff_replicates.csv",
        "edge_predictions": "results/k9_support_cutoff_edge_predictions.csv",
        "summary_table": "results/k9_support_cutoff_summary.csv",
    }
    (ROOT / "results" / "k9_support_cutoff_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
