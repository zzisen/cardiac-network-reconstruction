from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw_sarcgraph")
    args = parser.parse_args()
    manifest = pd.read_csv(ROOT / "data" / "public_source_manifest.csv")
    failures = []
    for row in manifest.itertuples(index=False):
        path = args.data_dir / row.Filename
        if not path.is_file():
            failures.append(f"missing {row.Filename}")
            continue
        data = path.read_bytes()
        checks = (len(data) == int(row.Bytes), git_blob_sha1(data) == row.GitBlobSHA1,
                  hashlib.sha256(data).hexdigest().upper() == row.SHA256)
        if not all(checks):
            failures.append(f"checksum mismatch {row.Filename}")
        else:
            print(f"verified {row.Filename} ({len(data)} bytes)")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {len(manifest)} public source files")

if __name__ == "__main__":
    main()
