"""Apply the predeclared geometry/topology-only known-tree/unknown-topology selections."""
from __future__ import annotations

import json
import hashlib
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MOVIES = ("E3", "E4", "E5")
UnknownTopology_SIZES = (4, 5, 6)
KnownTree_MIN_PERSISTENCE = 0.40


def load_graphs() -> dict[str, nx.Graph]:
    nodes = pd.read_csv(RESULTS / "public_graph_nodes.csv")
    edges = pd.read_csv(RESULTS / "public_graph_edges.csv")
    graphs = {}
    for movie in MOVIES:
        G = nx.Graph()
        subnodes = nodes[nodes.movie == movie].sort_values("node")
        for row in subnodes.itertuples(index=False):
            G.add_node(int(row.node), x_px=float(row.x_px), y_px=float(row.y_px),
                       tracked_frames=int(row.tracked_frames), degree=int(row.degree))
        subedges = edges[edges.movie == movie].sort_values(["source", "target"])
        for row in subedges.itertuples(index=False):
            G.add_edge(int(row.source), int(row.target), detections=int(row.detections),
                       detection_fraction=float(row.detection_fraction), length_px=float(row.length_px))
        graphs[movie] = G
    return graphs


def axis_projection(G: nx.Graph) -> dict[int, float]:
    labels = sorted(G.nodes)
    xy = np.array([[G.nodes[u]["x_px"], G.nodes[u]["y_px"]] for u in labels], dtype=float)
    centered = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    # Fix sign in image Cartesian coordinates for deterministic proximal/distal ordering.
    if axis[0] < 0 or (abs(axis[0]) <= 1e-12 and axis[1] < 0):
        axis = -axis
    return {u: float(np.dot(G.nodes[u]["x_px"] * np.array([1.0, 0.0]) +
                            G.nodes[u]["y_px"] * np.array([0.0, 1.0]) - xy.mean(axis=0), axis))
            for u in labels}


def eligible_k8_candidates(G: nx.Graph, movie: str) -> list[dict]:
    projection = axis_projection(G)
    candidates = []
    for junction in sorted(G.nodes):
        neighbors = sorted(v for v in G.neighbors(junction)
                           if float(G.edges[junction, v]["detection_fraction"]) >= KnownTree_MIN_PERSISTENCE)
        for leaves in combinations(neighbors, 3):
            if any(G.has_edge(u, v) for u, v in combinations(leaves, 2)):
                continue
            root_leaf = min(leaves, key=lambda u: (projection[u], u))
            daughters = [u for u in leaves if u != root_leaf]
            jxy = np.array([G.nodes[junction]["x_px"], G.nodes[junction]["y_px"]])
            daughters.sort(key=lambda u: (float(np.arctan2(G.nodes[u]["y_px"] - jxy[1],
                                                            G.nodes[u]["x_px"] - jxy[0]) % (2 * np.pi)), u))
            ordered = [root_leaf, junction, *daughters]
            persistence_by_leaf = {int(u): float(G.edges[junction, u]["detection_fraction"]) for u in leaves}
            persistence_ordered = [persistence_by_leaf[int(u)] for u in ordered if u != junction]
            candidates.append({
                "movie": movie,
                "selection": "all center/3-leaf candidates with each center-leaf detection persistence >= 0.40 and no induced leaf-leaf edge; ranked morphology-only by descending minimum persistence, descending mean persistence, movie ID, center ID, then sorted leaf IDs",
                "node_order": [int(u) for u in ordered],
                "root_node": int(root_leaf),
                "junction_node": int(junction),
                "sorted_leaf_nodes": [int(u) for u in leaves],
                "edges": [[0, 1], [1, 2], [1, 3]],
                "node_positions_px": [[float(G.nodes[u]["x_px"]), float(G.nodes[u]["y_px"])] for u in ordered],
                "edge_original_nodes": [[int(root_leaf), int(junction)], [int(junction), int(daughters[0])],
                                         [int(junction), int(daughters[1])]],
                "detection_persistence": persistence_ordered,
                "minimum_detection_persistence": float(min(persistence_ordered)),
                "mean_detection_persistence": float(np.mean(persistence_ordered)),
                "induced_subgraph_edge_count": 3,
                "minimum_edge_length_px": float(min(G.edges[junction, u]["length_px"] for u in leaves)),
            })
    return candidates


def k8_rank_key(motif: dict) -> tuple:
    return (-motif["minimum_detection_persistence"], -motif["mean_detection_persistence"],
            motif["movie"], motif["junction_node"], tuple(motif["sorted_leaf_nodes"]))


def select_k8(G: nx.Graph, movie: str) -> dict | None:
    candidates = eligible_k8_candidates(G, movie)
    return min(candidates, key=k8_rank_key) if candidates else None


def ordered_connected_set(G: nx.Graph, size: int) -> tuple[list[int], int] | None:
    if G.number_of_nodes() < size:
        return None
    components = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    comp = G.subgraph(components[0]).copy()
    projection = axis_projection(comp)
    root = min(comp.nodes, key=lambda u: (projection[u], u))
    ordered = [root]
    queued = {root}
    cursor = 0
    while cursor < len(ordered) and len(ordered) < size:
        u = ordered[cursor]
        cursor += 1
        for v in sorted(comp.neighbors(u), key=lambda v: (projection[v], v)):
            if v not in queued:
                queued.add(v)
                ordered.append(v)
                if len(ordered) == size:
                    break
    if len(ordered) < size:
        return None
    return ordered, root


def main() -> None:
    graphs = load_graphs()
    all_subgraphs = {}
    summary = []
    all_k8_candidates = []
    for movie in MOVIES:
        G = graphs[movie]
        movie_candidates = eligible_k8_candidates(G, movie)
        all_k8_candidates.extend(movie_candidates)
        motif = min(movie_candidates, key=k8_rank_key) if movie_candidates else None
        all_subgraphs[movie] = {"k8_motif": motif, "k9": {}}
        for n in UnknownTopology_SIZES:
            selected = ordered_connected_set(G, n)
            if selected is None:
                all_subgraphs[movie]["k9"][str(n)] = None
                continue
            ordered, root = selected
            induced = G.subgraph(ordered)
            index = {node: i for i, node in enumerate(ordered)}
            edges = [[index[u], index[v]] for u, v in sorted(induced.edges())]
            xy = [[float(G.nodes[u]["x_px"]), float(G.nodes[u]["y_px"])] for u in ordered]
            all_subgraphs[movie]["k9"][str(n)] = {
                "node_order": [int(u) for u in ordered],
                "root_index": int(index[root]),
                "edges": edges,
                "node_positions_px": xy,
                "cycle_rank": len(edges) - n + 1,
                "selection": "largest connected component (size, then lowest id); seed proximal major-axis node; deterministic breadth-first expansion with neighbors ordered by major-axis projection then node id; induce all edges",
            }
            summary.append({"movie": movie, "n": n, "nodes": len(ordered), "edges": len(edges),
                            "cycle_rank": len(edges) - n + 1, "root_original_node": int(root)})
        summary.append({"movie": movie, "n": "full", "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                        "cycle_rank": nx.number_of_edges(G) - nx.number_of_nodes(G) + nx.number_connected_components(G),
                        "root_original_node": None,
                        "components": nx.number_connected_components(G),
                        "branches_deg3plus": sum(d >= 3 for _, d in G.degree())})

    all_k8_candidates.sort(key=k8_rank_key)
    selected_k8 = all_k8_candidates[0] if all_k8_candidates else None
    selection_rule = {
        "minimum_detection_persistence": KnownTree_MIN_PERSISTENCE,
        "candidate": "one center and three distinct neighboring leaves; each center-leaf detection persistence >= 0.40; induced four-node subgraph has exactly the three Y edges and no leaf-leaf edge",
        "ranking": ["descending minimum edge persistence", "descending mean edge persistence",
                    "dataset/file ID ascending lexical order", "center-node ID ascending", "sorted leaf-node tuple ascending"],
        "root_orientation": "root is the proximal leaf by major-axis projection over the full movie graph (ties by node ID); daughter leaves are ordered by polar angle around the center (ties by node ID)",
        "performance_blind": True,
        "eligible_candidate_count": len(all_k8_candidates),
    }
    source_metadata = {
        "upstream_repository": "https://github.com/elejeune11/Sarc-Graph",
        "upstream_commit": (ROOT / "support" / "Sarc-Graph-legacy-ff3d665" / "UPSTREAM_COMMIT.txt").read_text(
            encoding="utf-8").strip(),
        "movie_file_ids": list(MOVIES),
        "derived_node_table": "results/public_graph_nodes.csv",
        "derived_node_table_sha256": hashlib.sha256((RESULTS / "public_graph_nodes.csv").read_bytes()).hexdigest(),
        "derived_edge_table": "results/public_graph_edges.csv",
        "derived_edge_table_sha256": hashlib.sha256((RESULTS / "public_graph_edges.csv").read_bytes()).hexdigest(),
    }
    (RESULTS / "public_subgraph_selection.json").write_text(
        json.dumps({"k8_selected_frozen": selected_k8, "k8_selection_rule": selection_rule,
                    "k8_selection_source": source_metadata,
                    "graphs": all_subgraphs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    pd.DataFrame(summary).to_csv(RESULTS / "public_selected_subgraphs_summary.csv", index=False)
    (RESULTS / "k8_motif_selection_freeze.json").write_text(
        json.dumps({"selection_rule": selection_rule, "source": source_metadata, "selected": selected_k8},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("known-tree selected:", None if selected_k8 is None else
          f"{selected_k8['movie']} nodes={selected_k8['node_order']} root={selected_k8['root_node']} "
          f"persistence={selected_k8['detection_persistence']}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
