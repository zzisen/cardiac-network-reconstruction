# Public topology provenance

## Source and scope

The morphology inputs are the public experimental example movies E3, E4, and E5 in the legacy [Sarc-Graph repository](https://github.com/elejeune11/Sarc-Graph), pinned at commit `ff3d6657654747b1d48bd28c1bdd677ac6c38a27`. This is a cardiac hiPSC-cardiomyocyte sarcomere-imaging resource. The source defines z-discs as nodes and detected sarcomeres as edges. Fifteen TIFF chunks were downloaded from the commit-pinned `ALL_MOVIES_RAW/experimental_data` path: five contiguous 30-frame chunks per example movie, each 512 × 512, unsigned 16-bit. Their URLs, byte counts, Git blob IDs, and SHA-256 values are in `results/public_topology_input_manifest.csv`. Raw TIFFs are intentionally not copied into this package; `code/fetch_sarcgraph_public_data.py` verifies them if re-downloaded.

E3, E4, and E5 are three example movies, not three independent biological replicates. No cell-level population inference is made.

## Processing

Segmentation and tracking use the original legacy source files copied under `support/Sarc-Graph-legacy-ff3d665/`, checked against the pinned repository commit. The input frames are ordered by the encoded frame interval. The upstream calls are `segmentation.segmentation_all(movie, gaussian_filter_size=1)` and `tracking.run_all_tracking(movie, tp_depth=4)`. The build script writes tracked z-disc positions and detected sarcomere links using the published spatial-graph convention (image x; y flipped upward). A candidate edge is retained when it is detected in at least `floor(0.10 × 150) = 15` frames. Isolated nodes are removed. Node coordinates are mean tracked positions in pixels. Edge length is Euclidean distance between these mean coordinates.

The edge detection count and fraction are stored separately as detection-persistence metadata. Neither is interpreted as stiffness, confidence-calibrated probability, or a mechanical weight. Tracking, segmentation, overlap, and the fixed 10% persistence cutoff can all alter apparent connectivity. The graphs are therefore morphology-derived inputs for algorithmic examples, not validated mechanical networks.

## Derived graph sizes

| Movie | Frames | Nodes | Persistent edges | Components | Largest component | Cycle rank | Nodes with degree ≥ 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E3 | 150 | 201 | 238 | 10 | 96 | 47 | 78 |
| E4 | 150 | 1,356 | 1,550 | 76 | 748 | 270 | 485 |
| E5 | 150 | 1,267 | 1,487 | 14 | 1,189 | 234 | 478 |

These counts describe the selected legacy segmentation/tracking pipeline and must not be treated as biological prevalence estimates. Detailed compact coordinates, links, source hashes, and deterministic subgraph selections are in `results/public_graph_nodes.csv`, `results/public_graph_edges.csv`, `results/public_graph_summary.csv`, and `results/public_subgraph_selection.json`.

## Selection rules fixed without inverse-performance feedback

For known-tree recovery, the V2.1 central motif was selected and frozen using morphology/detection data only, before inspecting reconstruction performance. The candidate pool includes all Y motifs in E3, E4, and E5 whose three center-to-leaf edges each have detection persistence at least 0.40 and whose induced four-node subgraph contains exactly the three center-leaf edges and no leaf-leaf edge. The deterministic ranking is descending minimum edge persistence, descending mean edge persistence, dataset/file ID in ascending lexical order, center-node ID ascending, then sorted leaf-node tuple ascending. The full movie graph's major-axis projection selects the proximal leaf as root (ties by node ID); the two daughters are ordered by polar angle around the center (ties by node ID). The scan produced 47 eligible candidates; the selected first-ranked motif is E5 nodes `[448, 446, 408, 463]`, rooted at node 448 with center 446.

### Frozen known-tree E5 motif provenance

The source is the public E5 Sarc-Graph example movie, repository commit `ff3d6657654747b1d48bd28c1bdd677ac6c38a27`. The graph is derived from `results/public_graph_nodes.csv` (SHA-256 `fc755a10656a82f68f8e7a13a666da4210ede1834072df29eddfff2ff8871d66`) and `results/public_graph_edges.csv` (SHA-256 `b9abd910bb9c495814cfd4239896373b45a5f7d0ab4211eff2bcdf28367f4735`). The full machine-readable decision is in `results/k8_motif_selection_freeze.json`.

| Ordered state | Original E5 node ID | x (px) | y (px) | Role |
|---|---:|---:|---:|---|
| 0 | 448 | 260.187171 | −259.524050 | proximal root leaf |
| 1 | 446 | 260.991159 | −252.767327 | center/junction |
| 2 | 408 | 265.389212 | −247.325660 | daughter 1 |
| 3 | 463 | 254.469868 | −257.635924 | daughter 2 |

| Original-node edge | Detection persistence |
|---|---:|
| 448–446 | 0.50 |
| 446–408 | 1.00 |
| 446–463 | 0.50 |

The selected four-node induced subgraph has exactly these three edges and is an acyclic Y tree. Persistence is a detection fraction from the Sarc-Graph pipeline, not stiffness, a calibrated probability, or a mechanical edge weight. These are real morphology-derived cardiac edges, not directly measured mechanical links. The motif is a local graph excerpt, not a claim that four states form a mechanically isolated cell.

For unknown-topology recovery, the largest connected component is selected (ties by smallest node ID). The root is the most proximal major-axis node. Starting there, breadth-first expansion uses neighbors sorted by major-axis projection and node ID until reaching n = 4, 5, or 6; all edges induced by the selected nodes are retained. The resulting nine morphology contexts include paths, branches, and cycles; no hidden true support is supplied to the practical estimator. Some selected local windows are trees and others contain cycles. This is a bounded local subgraph test, not whole-cell topology recovery.

## Reproduction

1. Run `python code/fetch_sarcgraph_public_data.py --output-dir <external-data-folder>` to fetch and checksum the pinned inputs.
2. Install the dependencies in `support/Sarc-Graph-legacy-ff3d665/requirements.txt` in a separate environment, including the tested `trackpy==0.6.4` and `opencv-python-headless==4.10.0.84` versions.
3. Run `python code/build_public_topology.py --data-dir <external-data-folder>`.
4. Run `python code/select_public_subgraphs.py`.

Only compact derived node/edge tables are included. The 15 raw movies are about 225 MB and remain outside the delivery directory.

## Sources

- Sarc-Graph paper: [PLOS Computational Biology, DOI 10.1371/journal.pcbi.1009443](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009443).
- Pinned source/data repository: [elejeune11/Sarc-Graph](https://github.com/elejeune11/Sarc-Graph/tree/ff3d6657654747b1d48bd28c1bdd677ac6c38a27).
- Updated SarcGraph paper notes z-disc overlap as a source of fragmentation: [PLOS Computational Biology, DOI 10.1371/journal.pcbi.1013436](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013436).
