# Legacy filename map

Frozen input/output filenames preserve their provenance and may retain historical labels. They are not method names.

| Historical label in a retained filename | Publication-facing terminology |
|---|---|
| `k8` | known-tree coded-profile recovery |
| `k9` | unknown-topology recovery / practical unknown-topology estimator |

The source modules and command-line runners in this release use publication-facing terminology. The map applies only to preserved result and figure-source filenames whose renaming would break provenance.
## Auxiliary-control and API names

| Historical item | Current release name or location |
|---|---|
| cyclic_k9_matrices.csv | results/auxiliary_controls/cyclic_exact/cyclic_k9_matrices.csv |
| clamp_boundary_control.csv | results/auxiliary_controls/clamp_boundary/clamp_boundary_control.csv |
| mismatch_observations.csv and mismatch_summary.json | results/auxiliary_controls/nonlinear_mismatch/ |
| recover_joint_LC in the isolated historical V1.1 source | p11.unknown_topology.recover_unknown_topology in the current public code |

Historical V1.1 generator code is isolated under scripts/auxiliary_controls/legacy_v1_1/code/p11 for provenance. It does not provide a compatibility alias in the current package.
