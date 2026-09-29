"""Build compact z-disc/sarcomere graphs from pinned public SarcGraph movies.

Segmentation and tracking call the original legacy Sarc-Graph functions at the
pinned commit. Only the final graph table writer is local, following the same
per-frame link and 10%-of-movie edge-persistence rule while also preserving
the detection counts as a separate field.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "support" / "Sarc-Graph-legacy-ff3d665"
DEFAULT_RESULTS = ROOT / "results"
MOVIES = ("E3", "E4", "E5")
PERSISTENCE_FRACTION = 0.10


def _slice_order(path: Path) -> int:
    match = re.search(r"frame(\d+)_to", path.name)
    if not match:
        raise ValueError(f"Cannot read frame start from {path.name}")
    return int(match.group(1))


def _as_2d(path: Path, ncols: int) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=float)
    return np.atleast_2d(np.loadtxt(path, dtype=float))


def _graph_for_movie(movie: str, processed: Path, out_dir: Path) -> dict:
    num_frames = len(glob.glob(str(processed / "ALL_MOVIES_MATRICES" / f"{movie}_matrices" / "*.npy")))
    track_dir = processed / "ALL_MOVIES_PROCESSED" / movie / "tracking_results"
    disc = _as_2d(track_dir / "tracking_results_zdisks.txt", 9)
    sarc = _as_2d(track_dir / "tracking_results_sarcomeres.txt", 10)

    # Legacy spatial_graph.py defines x as image column and flips y upward.
    observations_by_frame: dict[tuple[int, int], int] = {}
    positions: dict[int, list[tuple[float, float]]] = {}
    for row in disc:
        frame, local_id, particle = int(row[0]), int(row[1]), int(row[2])
        observations_by_frame[(frame, local_id)] = particle
        positions.setdefault(particle, []).append((float(row[4]), -float(row[3])))

    detections: Counter[tuple[int, int]] = Counter()
    for row in sarc:
        frame = int(row[0])
        local_u, local_v = int(row[5]), int(row[6])
        u = observations_by_frame.get((frame, local_u))
        v = observations_by_frame.get((frame, local_v))
        if u is None or v is None or u == v:
            continue
        detections[tuple(sorted((u, v)))] += 1

    cutoff = int(np.floor(PERSISTENCE_FRACTION * num_frames))
    edges = [(u, v, count) for (u, v), count in sorted(detections.items()) if count >= cutoff]
    graph = nx.Graph()
    for node in sorted(positions):
        xy = np.mean(np.asarray(positions[node], dtype=float), axis=0)
        graph.add_node(node, x_px=float(xy[0]), y_px=float(xy[1]), tracked_frames=len(positions[node]))
    for u, v, count in edges:
        graph.add_edge(u, v, detections=int(count), detection_fraction=float(count / num_frames))
    graph.remove_nodes_from(list(nx.isolates(graph)))

    node_rows = []
    for node, attrs in sorted(graph.nodes(data=True)):
        node_rows.append({"movie": movie, "node": int(node), **attrs, "degree": int(graph.degree[node])})
    edge_rows = []
    for u, v, attrs in sorted(graph.edges(data=True)):
        edge_rows.append({"movie": movie, "source": int(u), "target": int(v), **attrs,
                          "length_px": float(np.linalg.norm(np.array([graph.nodes[u]["x_px"], graph.nodes[u]["y_px"]]) -
                                                            np.array([graph.nodes[v]["x_px"], graph.nodes[v]["y_px"]])))} )
    pd.DataFrame(node_rows, columns=["movie", "node", "x_px", "y_px", "tracked_frames", "degree"]).to_csv(
        out_dir / "public_graph_nodes.csv", mode="a", header=not (out_dir / "public_graph_nodes.csv").exists(), index=False)
    pd.DataFrame(edge_rows, columns=["movie", "source", "target", "detections", "detection_fraction", "length_px"]).to_csv(
        out_dir / "public_graph_edges.csv", mode="a", header=not (out_dir / "public_graph_edges.csv").exists(), index=False)

    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c) if c else -1))
    return {
        "movie": movie,
        "frames": num_frames,
        "persistence_count_cutoff": cutoff,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "largest_component_nodes": len(components[0]) if components else 0,
        "cycle_rank": graph.number_of_edges() - graph.number_of_nodes() + len(components),
        "branchpoints_degree_ge_3": sum(degree >= 3 for _, degree in graph.degree()),
        "graph": graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing the 15 pinned E3/E4/E5 TIFF slices.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for old in (args.results_dir / "public_graph_nodes.csv", args.results_dir / "public_graph_edges.csv"):
        old.unlink(missing_ok=True)

    # Runtime imports are pinned to the verbatim upstream source copy.
    sys.path.insert(0, str(REPO))
    import segmentation
    import tracking

    summaries = []
    with tempfile.TemporaryDirectory(prefix="p11_sarcgraph_") as temp_name:
        work = Path(temp_name)
        matrices = work / "ALL_MOVIES_MATRICES"
        matrices.mkdir()
        old_cwd = Path.cwd()
        try:
            os.chdir(work)
            for movie in MOVIES:
                slices = sorted(args.data_dir.glob(f"real_data_{movie}_frame*_to*.tif"), key=_slice_order)
                if len(slices) != 5:
                    raise FileNotFoundError(f"Expected five TIFF slices for {movie}; found {len(slices)}")
                out_frames = matrices / f"{movie}_matrices"
                out_frames.mkdir()
                frame_idx = 0
                for source in slices:
                    stack = tifffile.imread(source)
                    if stack.shape != (30, 512, 512) or stack.dtype != np.uint16:
                        raise ValueError(f"Unexpected TIFF shape/dtype in {source.name}: {stack.shape}/{stack.dtype}")
                    for frame in stack:
                        np.save(out_frames / f"frame-{frame_idx:04d}.npy", frame)
                        frame_idx += 1
                if frame_idx != 150:
                    raise ValueError(f"{movie} did not yield 150 ordered frames")
                segmentation.segmentation_all(movie, gaussian_filter_size=1)
                tracking.run_all_tracking(movie, tp_depth=4)
                summaries.append(_graph_for_movie(movie, work, args.results_dir))
                graph = summaries[-1].pop("graph")
                print(f"{movie}: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
                      f"components={summaries[-1]['components']} branches={summaries[-1]['branchpoints_degree_ge_3']}",
                      flush=True)
        finally:
            os.chdir(old_cwd)

    pd.DataFrame(summaries).to_csv(args.results_dir / "public_graph_summary.csv", index=False)
    print(f"Wrote compact graph tables to {args.results_dir}")


if __name__ == "__main__":
    main()
