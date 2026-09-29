# Reproducibility

## Environment

The tested runtime is Python 3.10.18 with direct dependencies in `requirements.txt` and `figure_code/requirements.txt`. Create a fresh virtual environment before installation and install both requirement files.

## Commands

```bash
python scripts/verify_inputs.py
python scripts/fetch_public_data.py
python scripts/verify_public_data.py
python scripts/reproduce_main_results.py
python scripts/reproduce_figures.py
python -m unittest discover -s tests -v
```

`reproduce_main_results.py` runs the known-tree morphology validation, unknown-topology intervention and mechanics sensitivities, practical estimator validation, and fixed support-cutoff replay. It compares generated headline values with `results/publication_reference.json`. Each runner fails with a nonzero exit if a required input is missing or an analysis fails. The unit suite covers the exact branch and cycle controls.

`reproduce_all.py` validates the shipped compact graph inputs, runs the full numeric workflow, regenerates all four figures and their source tables, executes the test suite, and checks the frozen headline reference. Raw movie retrieval is a separate step because the files are not redistributed.

The approved final figure artwork is included as PDF, SVG, and PNG. Figure 1 is preserved from that artwork during the integrated figure rebuild. Figures 2–4 can be rendered with Arial when available or Matplotlib's public DejaVu Sans fallback; typography and rendered bytes can vary. Numerical checkpoints and source tables remain checked, with a tight floating-point tolerance for regenerated numeric fields. No font files are distributed.

The compact morphology inputs have a separate SHA-256 list at `data/derived_input_manifest.csv`. If the source movies are retrieved, `verify_public_data.py` validates byte size, Git blob SHA-1, and SHA-256 for all 15 inputs. `scripts/build_public_topology.py --data-dir data/raw_sarcgraph` and `scripts/select_public_subgraphs.py` regenerate the graph tables and deterministic selections.

## Comparison rules

Integer graph counts and query budgets compare exactly. Deterministic numerical checks use tolerances recorded in `results/publication_reference.json`; fitted optimizer coefficients are not compared bit for bit. Topology support metrics compare at the frozen classification threshold.
