# Data provenance

## Raw morphology inputs

The source is the public Sarc-Graph repository at commit `ff3d6657654747b1d48bd28c1bdd677ac6c38a27`: <https://github.com/elejeune11/Sarc-Graph/tree/ff3d6657654747b1d48bd28c1bdd677ac6c38a27>. The input manifest lists the 15 E3/E4/E5 TIFF chunks, exact pinned URLs, byte sizes, Git blob SHA-1 values, and SHA-256 values. Raw movie files are not redistributed.

The copied upstream source is covered by the MIT license included at `support/Sarc-Graph-legacy-ff3d665/LICENSE`. Its upstream file hashes and commit record are preserved beside that license.

## Processing and derived graph data

The source represents z-discs as nodes and detected sarcomeres as edges. Five consecutive 30-frame chunks form each 150-frame example movie. Segmentation and tracking use the pinned upstream implementation. A graph edge is retained when it is detected in at least 15 of 150 frames; isolated nodes are removed. Coordinates are mean tracked positions in pixels. Edge detection persistence is metadata and is not interpreted as stiffness or as a calibrated probability.

E3, E4, and E5 are three example movies, not independent biological replicates. The included nodes, edges, summary, and selected local subgraphs are compact derived inputs. Their hashes are in `data/derived_input_manifest.csv`. The graph-derived structures ground algorithm examples; they are not direct mechanical measurements.

## Mechanical parameters

Mechanical coefficients are effective, literature-informed model quantities. The release does not claim nodewise empirical calibration or intervention validation.
