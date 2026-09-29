from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Rebuild, validate, and compare publication figures and source tables.")
parser.add_argument("--font-family", choices=["auto", "public"], default="auto")
args = parser.parse_args()

def run(*cmd):
    subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, check=True)

def compare_tables(expected: Path, actual: Path, label: str):
    if not actual.exists():
        raise SystemExit(f"regenerated {label} is missing")
    try:
        assert_frame_equal(pd.read_csv(actual), pd.read_csv(expected), check_exact=False,
                           rtol=1e-12, atol=1e-14, check_dtype=False)
    except AssertionError as exc:
        raise SystemExit(f"regenerated {label} differs from the frozen release table: {exc}") from exc

with tempfile.TemporaryDirectory(prefix="p11_figure_rebuild_") as temp:
    output = Path(temp)
    run(ROOT / "figure_code" / "build_all_figures.py", "--input-root", ROOT / "inputs",
        "--output-root", output, "--font-family", args.font_family)
    run(ROOT / "figure_code" / "audit_frozen_aggregates.py", "--input-root", ROOT / "inputs",
        "--output-root", output)
    run(ROOT / "figure_code" / "verify_delivery.py", "--output-root", output)
    for figure in range(1, 5):
        expected = ROOT / "figure_source_data" / f"Figure{figure}_source_data.csv"
        actual = output / "source_data" / expected.name
        compare_tables(expected, actual, f"Figure {figure} source data")
    for figure in range(2, 5):
        expected = ROOT / "figure_source_data" / "raw_and_summaries" / f"Figure{figure}_summaries.csv"
        actual = output / "source_data" / "raw_and_summaries" / expected.name
        compare_tables(expected, actual, f"Figure {figure} summary data")
    environment = json.loads((output / "qa" / "BUILD_ENVIRONMENT.json").read_text(encoding="utf-8"))
    print(f"Figure source data and summaries match; delivery QA passed with {environment['font']}.")
