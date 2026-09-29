"""Fetch the pinned E3/E4/E5 example movies used for morphology-derived graphs.

The data live outside the delivery folder. Expected byte counts and both Git
blob and SHA-256 checksums are taken from results/public_topology_input_manifest.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "public_source_manifest.csv"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw_sarcgraph")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    for row in manifest.itertuples(index=False):
        target = args.output_dir / row.Filename
        data = target.read_bytes() if target.exists() else None
        if data is None:
            with urlopen(row.SourceURL, timeout=90) as response:
                data = response.read()
            target.write_bytes(data)
        checks = {
            "bytes": len(data) == int(row.Bytes),
            "git_blob_sha1": git_blob_sha1(data) == row.GitBlobSHA1,
            "sha256": hashlib.sha256(data).hexdigest().upper() == row.SHA256,
        }
        if not all(checks.values()):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum failure for {row.Filename}: {checks}")
        print(f"verified {row.Filename} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
