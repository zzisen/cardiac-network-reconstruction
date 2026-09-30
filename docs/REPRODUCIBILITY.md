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

## Supplement auxiliary controls

The retained controls claimed in Supplement S5.1–S5.4 and S6 are available in this release:

| Supplement control | Public record | Code or test |
|---|---|---|
| S5.1 generated path family and exact recovery | results/auxiliary_controls/synthetic/path/synthetic_path_family.csv, exact_recovery.csv, and scaling.csv | Archived generator in scripts/auxiliary_controls/legacy_v1_1/code/p11/benchmark.py; current path tests in tests/test_core.py |
| S5.1 known labelled Y tree and daughter-swap ambiguity | results/auxiliary_controls/synthetic/known_tree_y/architecture_repair_exact.csv and branch_alias_frequency.csv | code/p11/architecture_repair.py and tests/test_core.py::test_y_tree_profiles_resolve_labelled_daughter_swap |
| S5.1 endpoint-path coherent-only control | Six-node path reconstruction test in tests/test_core.py::test_endpoint_path_coherent_spectral_measure_control | code/p11/architecture_repair.py |
| S5.1 five-node cyclic exact control, true/recovered L and C matrices | results/auxiliary_controls/cyclic_exact/cyclic_k9_matrices.csv | code/p11/architecture_repair.py::make_cyclic_network_fixture and tests/test_core.py::test_cyclic_general_graph_joint_recovery; the test checks the 20-query budget |
| S5.2 inherited path noise and Y/cycle perturbations | results/auxiliary_controls/synthetic/path/noise_replicates.csv, noise_summary.csv, and results/auxiliary_controls/synthetic/architecture_noise.csv | Fixed-seed historical generator modules in scripts/auxiliary_controls/legacy_v1_1/code/p11/benchmark.py |
| S5.3 clamp-boundary semantic control | results/auxiliary_controls/clamp_boundary/clamp_boundary_control.csv | Archived clamp comparison in scripts/auxiliary_controls/legacy_v1_1/code/p11/benchmark.py; current principal-block behavior is in code/p11/unknown_topology.py |
| S5.4 nonlinear SarcomereModel mismatch probe | results/auxiliary_controls/nonlinear_mismatch/mismatch_observations.csv and mismatch_summary.json | Runner in scripts/auxiliary_controls/legacy_v1_1/code/p11/mismatch.py and reproduction command in scripts/auxiliary_controls/reproduce.py |

The auxiliary-control tables can be regenerated without overwriting the retained release copies:

    python scripts/auxiliary_controls/reproduce.py --output-dir aux-control-reproduction

The historical V1.1 source snapshot is kept in its own namespace to preserve provenance. The current public unknown-topology API is p11.unknown_topology.recover_unknown_topology.


The public control mapping is in docs/SUPPLEMENT_CONTROL_MAP.csv. SHA-256 values for packaged auxiliary-control payload files are in docs/AUXILIARY_CONTROL_SHA256.csv; the manifest excludes itself.
