# Reproduce Figure 1

Use Python 3.10 or newer. Install the listed dependencies and run the generator from any working directory:

```bash
python -m pip install -r source/requirements.txt
python source/build_figure1.py
```

The code resolves inputs relative to itself and writes the three Figure 1 files and crop records within this folder. It uses Arial where available and has public sans-serif fallbacks; no font files are bundled. The supplied approved PDF, SVG, and PNG are included at this directory's root.

Inputs include the frozen graph tables, topology selections, schematic positions, provenance, and deterministic spatial crop records. The optional `source/verify_figure1.py` checks output dimensions, vector content, label colors, frozen topology identities, crop membership, and repeat-build checksums.
