from __future__ import annotations
import csv, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
reference = json.loads((ROOT / "results" / "publication_reference.json").read_text(encoding="utf-8"))
def table(path):
    with path.open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

summary = {row["movie"]: row for row in table(ROOT / "results" / "public_graph_summary.csv")}
for movie, expected in reference["morphology_counts"].items():
    for field, value in expected.items():
        if int(summary[movie][field]) != value: raise SystemExit(f"Graph reference mismatch: {movie}/{field}")
practical = table(ROOT / "results" / "k9_practical_summary.csv")
exact = next(r for r in practical if r["design"] == "minimal_exact_budget" and float(r["noise_fraction"]) == 0.0)
if int(float(exact["median_queries"])) != reference["exact_minimum_queries_n5"]: raise SystemExit("Exact query minimum mismatch")
for design, expected in reference["practical_unknown_topology_noise_0p1_percent"].items():
    row = next(r for r in practical if r["design"] == design and float(r["noise_fraction"]) == 0.001)
    if int(float(row["median_queries"])) != expected["median_queries"]: raise SystemExit(f"Query budget mismatch: {design}")
    if expected["median_f1"] is not None and not math.isclose(float(row["topology_F1_median"]), expected["median_f1"], rel_tol=0, abs_tol=reference["f1_absolute_tolerance"]): raise SystemExit(f"F1 mismatch: {design}")
cutoff = table(ROOT / "results" / "k9_support_cutoff_summary.csv")
for design, expected in reference["support_cutoff_0p1_percent_noise_at_0p25"].items():
    row = next(r for r in cutoff if r["design"] == design and float(r["noise_fraction"]) == 0.001 and float(r["cutoff"]) == 0.25)
    if expected is not None and not math.isclose(float(row["topology_F1_median"]), expected, rel_tol=0, abs_tol=reference["f1_absolute_tolerance"]): raise SystemExit(f"Cutoff F1 mismatch: {design}")
checks = json.loads((ROOT / "results" / "k8_exact_checks.json").read_text(encoding="utf-8"))["checks"]
for row in checks:
    if not row["known_tree_fit_success"] or row["known_tree_fit_relative_L_error"] > reference["known_tree_relative_error_max"]: raise SystemExit("Known-tree recovery regression")
    if row["coherent_daughter_swap_max_abs"] > reference["coherent_alias_max_abs"]: raise SystemExit("Coherent alias regression")
    if row["labelled_profile_swap_max_abs"] < reference["profile_separation_min"]: raise SystemExit("Known-tree profile separation regression")
print("headline outputs match publication_reference.json")
