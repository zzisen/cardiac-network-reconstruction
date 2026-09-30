# Auxiliary controls

The retained control tables are organized by the Supplement sections that describe them. Their sources, current API names, tests, and reproduction command are documented in ../docs/REPRODUCIBILITY.md.

The historical V1.1 generator modules are preserved under legacy_v1_1/code/p11 as isolated source provenance. They retain the earlier internal function names. The current public API remains p11.unknown_topology.recover_unknown_topology; no compatibility alias is provided.

Regenerate the retained auxiliary-control tables into a separate directory with:

    python scripts/auxiliary_controls/reproduce.py --output-dir aux-control-reproduction

The runner writes only to the directory passed with --output-dir. It requires the versions from the repository requirements files and the included SarcomereModel v0.2.0 source snapshot.

