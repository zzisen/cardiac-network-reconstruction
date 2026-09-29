from __future__ import annotations
import csv, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "data" / "derived_input_manifest.csv"
with manifest.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    path = ROOT / row["file"]
    data = path.read_bytes()
    actual = (len(data), hashlib.sha256(data).hexdigest())
    expected = (int(row["bytes"]), row["sha256"])
    if actual != expected:
        raise SystemExit(f"Input verification failed: {row['file']}")
    print(f"verified {row['file']}")
