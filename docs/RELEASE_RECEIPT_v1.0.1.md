# P1.1 reproducibility release v1.0.1 receipt

Release version: 1.0.1
Source base: public repository main commit 3b41ab81e1fbdbf9b326e64bd2d496fa7ad89b35 (the DOI metadata patch after v1.0.0).
Scientific content: unchanged. This release adds retained auxiliary-control records and provenance, and updates reproducibility/version documentation.

## Added controls

- Cyclic exact true/recovered L and C matrix record and the retained clamp-boundary comparison.
- Retained path, Y-tree, branch-ambiguity, and fixed-seed perturbation control tables.
- SarcomereModel v0.2.0 mismatch observations/configuration, with its upstream source snapshot, README, and MIT license.
- Isolated historical V1.1 generator modules and a runner that writes regenerated tables only to an explicitly selected output directory.
- Auxiliary-control mapping and SHA-256 manifest.

## Validation

- Python 3.10.18; the existing scientific regression suite passed all 21 tests.
- Targeted retained-control checks passed: 25 cyclic matrix entries with recovered L and diagonal C matching the stored true values within tolerance; retained and intentionally edge-deleted clamp thresholds reproduced the stored 0.134604870185 shift; 24 mismatch observations cover four profiles and six frequencies, with the stored n=6, dt=0.002 s, T=6 s, 1 Hz, seed 78031 configuration and typed PrecisionFailure.
- The included 41-entry auxiliary SHA-256 manifest verified against file sizes and hashes. The SarcomereModel v0.2.0 snapshot includes its upstream README, version metadata, and MIT license.
- Privacy, credential, workstation-path, prompt, audit, tracker, and internal-review scan passed for 70 package files.
- Diff against validated base 3b41ab81e1fbdbf9b326e64bd2d496fa7ad89b35 shows no change to estimator logic, figure artwork, or existing numerical results. The only code-file change is the package version string in `code/p11/__init__.py`.
- The already-passed v1.0.0 full clean-room scientific reproduction is reused because its scientific inputs, algorithms, existing results, and figure artwork are unchanged; no full practical-estimator, mechanics, intervention, or figure campaign was rerun for this completeness patch.
- Targeted figure/source-data delivery QA passed: 279/279 numerical checks, source-data and summary comparisons, and vector/font/page checks for Figures 1–4.
- Package ZIP CRC and integrity are checked after archive creation; the final SHA-256 values are returned with the release details.
- Zenodo DOI reserved before release: 10.5281/zenodo.23051131 (new-version draft 23051131, linked to the v1.0.0 concept record).
