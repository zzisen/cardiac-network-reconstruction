from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def run(*args): subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)
run(ROOT / "scripts" / "verify_inputs.py")
run(ROOT / "scripts" / "reproduce_main_results.py")
run(ROOT / "scripts" / "reproduce_figures.py")
run("-m", "unittest", "discover", "-s", "tests", "-v")
run(ROOT / "scripts" / "check_reference.py")
