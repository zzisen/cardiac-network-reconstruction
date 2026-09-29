# Data

`public_source_manifest.csv` identifies the 15 public E3/E4/E5 TIFF chunks by pinned source URL, byte size, Git blob SHA-1, and SHA-256. The raw movies are not included. Use `python scripts/fetch_public_data.py` to retrieve them and `python scripts/verify_public_data.py` to verify them.

Compact derived node/edge tables and deterministic subgraph selections are stored under `results/` because the analysis scripts consume those stable paths. `derived_input_manifest.csv` verifies these shipped inputs.
