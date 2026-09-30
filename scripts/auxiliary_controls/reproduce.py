from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEGACY_CODE = ROOT / "scripts" / "auxiliary_controls" / "legacy_v1_1" / "code"
sys.path.insert(0, str(LEGACY_CODE))

from p11.benchmark import clamp_boundary_control, run_exact_recoveries, run_noise_trials
from p11.mismatch import run_sarcomere_model_mismatch
from p11.pipeline import _architecture_repair_results, _write_json


def write_csv(frame: pd.DataFrame, output_dir: Path, relative_path: str) -> None:
    path = output_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the retained synthetic and nonlinear auxiliary-control tables."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for regenerated tables; use an empty temporary directory.",
    )
    parser.add_argument("--noise-replicates", type=int, default=10)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    exact, path_family, scaling = run_exact_recoveries()
    write_csv(exact, out, "synthetic/path/exact_recovery.csv")
    write_csv(path_family, out, "synthetic/path/synthetic_path_family.csv")
    write_csv(scaling, out, "synthetic/path/scaling.csv")

    noise, noise_summary = run_noise_trials(replicates=args.noise_replicates)
    write_csv(noise, out, "synthetic/path/noise_replicates.csv")
    write_csv(noise_summary, out, "synthetic/path/noise_summary.csv")

    clamp = clamp_boundary_control()
    write_csv(clamp, out, "clamp_boundary/clamp_boundary_control.csv")

    architecture, branch_alias, cyclic, architecture_noise = _architecture_repair_results()
    write_csv(architecture, out, "synthetic/known_tree_y/architecture_repair_exact.csv")
    write_csv(branch_alias, out, "synthetic/known_tree_y/branch_alias_frequency.csv")
    write_csv(cyclic, out, "cyclic_exact/cyclic_k9_matrices.csv")
    write_csv(architecture_noise, out, "synthetic/architecture_noise.csv")

    model_source = ROOT / "support" / "SarcomereModel-v0.2.0" / "src"
    mismatch_rows, mismatch_summary = run_sarcomere_model_mismatch(model_source)
    write_csv(pd.DataFrame(mismatch_rows), out, "nonlinear_mismatch/mismatch_observations.csv")
    _write_json(out / "nonlinear_mismatch" / "mismatch_summary.json", mismatch_summary)


if __name__ == "__main__":
    main()

