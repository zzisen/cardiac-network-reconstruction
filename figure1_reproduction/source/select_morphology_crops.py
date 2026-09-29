"""Deterministic local ROIs, reusing the project's morphology-only BFS ordering.

Same rule for all three movies, fixed before visual rendering: take the first
24 nodes by the existing proximal-root BFS, then include ALL source nodes and
ALL induced source edges in their closed axis-aligned coordinate bounding box.
The crop is spatial, not a pruned subgraph. No mechanics/recovery inputs exist.
"""
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np

RULE = (
    "Largest connected component (descending size, then lowest node ID); "
    "major-axis SVD with positive x sign (positive y if x approximately zero); "
    "root at lowest projection (node ID breaks ties); breadth-first expansion "
    "with neighbors ordered by major-axis projection, then ID; stop at 24 nodes. "
    "Their closed axis-aligned x/y bounding rectangle is the ROI. Retain all "
    "source nodes inside the rectangle, including other components, and every "
    "source edge with both endpoints inside. No performance or aesthetic ranking."
)


def select_crops(nodes, edges):
    crops = {}
    for movie in ("E3", "E4", "E5"):
        rows = [r for r in nodes if r["movie"] == movie]
        xy = {int(r["node"]): np.array([float(r["x_px"]), float(r["y_px"])]) for r in rows}
        adj = {i: set() for i in xy}
        pairs = sorted((int(r["source"]), int(r["target"])) for r in edges if r["movie"] == movie)
        for u,v in pairs:
            adj[u].add(v); adj[v].add(u)
        remaining, components = set(xy), []
        while remaining:
            queue, comp = deque([min(remaining)]), set()
            while queue:
                u = queue.popleft()
                if u in comp: continue
                comp.add(u); queue.extend(sorted(adj[u]-comp))
            remaining -= comp; components.append(comp)
        comp = min(components, key=lambda c: (-len(c), min(c)))
        ids = sorted(comp)
        points = np.array([xy[u] for u in ids])
        mean = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points-mean, full_matrices=False)
        axis = vt[0]
        if axis[0] < 0 or (abs(axis[0]) <= 1e-12 and axis[1] < 0): axis = -axis
        projection = {u: float(np.dot(xy[u]-mean, axis)) for u in ids}
        root = min(ids, key=lambda u: (projection[u], u))
        ordered, seen, cursor = [root], {root}, 0
        while cursor < len(ordered) and len(ordered) < 24:
            u = ordered[cursor]; cursor += 1
            for v in sorted(adj[u], key=lambda v: (projection[v],v)):
                if v in seen: continue
                seen.add(v); ordered.append(v)
                if len(ordered) == 24: break
        assert len(ordered) == 24
        bounds = np.array([xy[u] for u in ordered])
        lo, hi = bounds.min(0), bounds.max(0)
        chosen = sorted(u for u,p in xy.items() if np.all(p >= lo) and np.all(p <= hi))
        keep = set(chosen)
        es = [[u,v] for u,v in pairs if u in keep and v in keep]
        crops[movie] = {
            "crop_id": f"{movie}_proximal_BFS24_closed_bbox",
            "rule": RULE, "root_source_node": root,
            "largest_component_node_count": len(comp),
            "major_axis_xy": axis.tolist(), "bfs24_source_node_order": ordered,
            "bounds_source_px": {"xmin": float(lo[0]), "xmax": float(hi[0]),
                                  "ymin": float(lo[1]), "ymax": float(hi[1])},
            "selected_source_node_ids": chosen, "selected_source_edges": es,
            "nodes": len(chosen), "edges": len(es),
            "source_xy": [[u, *xy[u].tolist()] for u in chosen],
            "boundary_policy": "Induced source edges only; no boundary-crossing stubs; no deleted internal edges.",
            "presentation": "Representative local crop of an example movie; not a population summary."
        }
    return crops


def write_crops(crops, nodes, edges, root):
    root = Path(root)
    (root/"CROP_SELECTION.json").write_text(json.dumps(crops, indent=2), encoding="utf-8")
    for filename, rows, flag in [("CROP_NODES.csv",nodes,"node"),("CROP_EDGES.csv",edges,"edge")]:
        with (root/filename).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader()
            for r in rows:
                keep=set(crops[r["movie"]]["selected_source_node_ids"])
                inside = int(r["node"]) in keep if flag=="node" else int(r["source"]) in keep and int(r["target"]) in keep
                if inside: w.writerow(r)
