# Structural priors determine the measurements needed to reconstruct mechanics in cardiac myofibrillar networks

This release accompanies a study of how prior knowledge of network structure changes the information needed to recover effective mechanical parameters. It combines a serial-path control, recovery on a known labelled tree with spatially coded profiles, and practical recovery when topology and storage are unknown. The cardiac input is public morphology; mechanical coefficients are effective model quantities, not direct nodewise measurements.

## Contents

- `code/`: recovery models, estimators, and JSON utilities.
- `scripts/`: public-data retrieval, analysis runs, and figure generation.
- `data/`: public source manifest and hashes for compact derived morphology inputs.
- `results/`: frozen numerical tables and generated analysis outputs.
- `figures/`: approved final PDF, SVG, and PNG files for each main figure.
- `figure_code/`: the complete figure generator, style, and delivery checks.
- `figure_source_data/`: panel-indexed source tables, summaries, and dictionary.
- `figure1_reproduction/`: Figure 1 source, inputs, and deterministic crop data.
- `inputs/`: frozen figure-build inputs and numerical checkpoints.
- `tests/`: recovery, estimator, input-contract, and serialization tests.
- `docs/`: reproducibility, provenance, and terminology notes.
- `support/Sarc-Graph-legacy-ff3d665/`: MIT-licensed upstream source used for optional morphology regeneration.

## Install

Use Python 3.10.18 and the versions in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r figure_code/requirements.txt
```

## Public data

The 15 raw movie chunks are approximately 225 MB and are not redistributed. Fetch and verify them with:

```bash
python scripts/fetch_public_data.py
python scripts/verify_public_data.py
```

The pinned source revision and per-file checksums are in `data/public_source_manifest.csv`. The release already includes compact derived graph tables for the analyses. To regenerate them from the movies, see `docs/DATA_PROVENANCE.md`.

## Quick reproduction

```bash
python scripts/verify_inputs.py
python scripts/reproduce_figures.py
python -m unittest discover -s tests -v
```

## Full reproduction

```bash
python scripts/reproduce_all.py
```

This reruns the numerical validations and sensitivities, regenerates figure source tables and figures, runs the regression tests, and checks headline outputs against `results/publication_reference.json`. The full practical estimator and cutoff replay take several minutes each. Commands and outputs are described in `docs/REPRODUCIBILITY.md`.

## Citation

Please cite the study using `CITATION.cff`. Software release version: `1.0.1`. The repository is [zzisen/cardiac-network-reconstruction](https://github.com/zzisen/cardiac-network-reconstruction). The v1.0.1 reproducibility archive is available on Zenodo at https://doi.org/10.5281/zenodo.23051131. The preserved v1.0.0 record remains at https://doi.org/10.5281/zenodo.23049835.

## Auxiliary controls

Supplementary path, labelled-tree, cyclic, clamp-boundary, fixed-seed noise, and nonlinear mismatch controls are retained under results/auxiliary_controls/. Their file-to-test mapping and regeneration instructions are in docs/REPRODUCIBILITY.md. This release documents the current unknown-topology API as p11.unknown_topology.recover_unknown_topology.

## License

Original P1.1 code is released under the MIT License; see `LICENSE`. Upstream Sarc-Graph source retains its own MIT license and attribution. Do not redistribute the raw movies from this archive.

## Data provenance

The morphology inputs are E3, E4, and E5 example movies from the commit-pinned Sarc-Graph repository. Details, processing choices, and limits are in `docs/DATA_PROVENANCE.md`.

## Support

Corresponding author: Zisen Zhou.


## Expected outputs

The full run refreshes the known-tree checks and morphology summaries, unknown-topology intervention and mechanics summaries, practical-estimator and support-cutoff tables, four figure-source CSV files, and four PNG/SVG figures. Historical result filenames are mapped to publication terminology in `docs/LEGACY_FILENAME_MAP.md`. The run ends by checking the stable headline metrics in `results/publication_reference.json`.









## Figure fonts

The figure builder uses Arial when installed. Otherwise it automatically uses DejaVu Sans from Matplotlib; Matplotlib's bundled Computer Modern symbol face may supply isolated math glyphs. The `--font-family public` option forces this fallback for testing. No font files are included. Typography and PDF/SVG bytes may differ from the approved Arial artwork, while generated numerical tables are checked with a tight floating-point tolerance.
