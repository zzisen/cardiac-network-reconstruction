# Figure source and reproduction

The main figure artwork, source tables, and complete generator in this release are the final figure set dated 2026-09-30.

## Rebuild

Install both dependency lists in a Python 3.10 environment:

```bash
python -m pip install -r requirements.txt -r figure_code/requirements.txt
python scripts/reproduce_figures.py
```

The script builds into a temporary directory, verifies all supplied numerical checkpoints and retained-data aggregates, audits the PDF/SVG outputs, and compares regenerated panel-indexed source tables with the frozen `figure_source_data/` tables (exact for text/categorical fields and within 1e-12 relative / 1e-14 absolute tolerance for numbers). Figure 1 is copied from its approved native artwork; its separate generator and inputs are preserved under `figure1_reproduction/`.

The generator prefers Arial if available. If it is absent, it uses DejaVu Sans from Matplotlib and continues. Use `--font-family public` to force that open-font path:

```bash
python scripts/reproduce_figures.py --font-family public
```

The public fallback may change text metrics and visual bytes. It does not change the scientific results; regenerated floating-point tables are checked with the tolerance stated above. PDF checks accept DejaVu Sans and Matplotlib's bundled Computer Modern math-symbol face in the public-font run, and confirm all used fonts are embedded. No font files are distributed.

The `inputs/` tree contains frozen numerical figure inputs and provenance code snapshots needed by the generator.
