from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUNNERS = [
    "run_known_tree_morphology_validation.py",
    "run_unknown_topology_intervention_sensitivity.py",
    "run_unknown_topology_mechanics_sensitivity.py",
    "run_practical_unknown_topology_validation.py",
    "run_support_cutoff_sensitivity.py",
]
for name in RUNNERS:
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(ROOT / "scripts" / "check_reference.py")], cwd=ROOT, check=True)
