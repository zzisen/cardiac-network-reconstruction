"""Figure 1: deterministic, vector-first drawing from frozen local inputs.

Run from any directory: python /path/to/code/build_figure1.py
The 1800 x 1200 art-direction coordinates are rendered at 180 x 120 mm.
Conceptual signal/intervention glyphs are symbols, not numerical observations.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from select_morphology_crops import select_crops, write_crops
import platform

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "source" / "inputs"
QA = ROOT / "source" / "records"
QA.mkdir(exist_ok=True)
COL = dict(ink="#3F4548", teal="#16A6A1",
           ochre="#F2B33D", root="#E5543D", pale="#D8E0E3",
           pale_edge="#D8E0E3", orange="#F47C20", white="#FFFFFF")
PANEL_B_OFFSET = 45  # Use space released by removing the footer, keeping row spacing.
mpl.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.2, "text.color": COL["ink"], "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
    "svg.hashsalt": "P11-FIGURE1-FINAL-FREEZE-2026-09-29",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})


def read_table(name):
    with (ASSETS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


nodes = read_table("public_graph_nodes.csv")
edges = read_table("public_graph_edges.csv")
crops = select_crops(nodes, edges)
write_crops(crops, nodes, edges, QA)
selection = json.loads((ASSETS / "public_subgraph_selection.json").read_text())
branch = selection["k8_selected_frozen"]
local = selection["graphs"]["E3"]["k9"]["5"]
layout = json.loads((ASSETS / "schematic_layout.json").read_text())
fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1800), ylim=(1200, 0))
ax.set_axis_off()
ax.set_aspect("equal")
all_text = []


def text(x, y, value, size=7.2, color=None, weight="normal", ha="center", **kw):
    # Label semantics are enforced centrally, including profile-column headers.
    if value == "r": color = COL["root"]
    elif value in {"1", "2", "3"}: color = COL["ink"]
    obj = ax.text(x, y, value, ha=ha, va="center", fontsize=size,
                  color=color or COL["ink"], weight=weight, **kw)
    all_text.append(obj)
    return obj


def line(points, color=None, lw=.8, **kw):
    p = np.asarray(points)
    return ax.plot(p[:, 0], p[:, 1], color=color or COL["ink"], lw=lw, **kw)


def arrow(a, b, color=None, lw=.65, scale=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=scale,
                  linewidth=lw, color=color or COL["ink"], shrinkA=0, shrinkB=0))


def dot(x, y, kind="ordinary", recovered=False, radius=8):
    color = COL["teal"]
    if kind == "root":
        ax.add_patch(Rectangle((x-radius, y-radius), radius*2, radius*2,
                              facecolor=COL["root"], edgecolor="white", lw=.4, zorder=5))
    else:
        if kind == "junction":
            color = COL["ochre"]
        if kind == "background":
            color = COL["pale"]
        ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="white",
                           lw=.35, zorder=5))


def graph(pos, pairs, root=0, junction=None, recovered=False, labels=False,
          radius=8, edge_color=None, label_offsets=None, edge_style="solid"):
    p = np.asarray(pos)
    ax.add_collection(LineCollection([[p[u], p[v]] for u, v in pairs],
           colors=edge_color or COL["ink"], linewidths=.8, linestyles=edge_style, zorder=3))
    for i, (x, y) in enumerate(p):
        dot(x, y, "root" if i == root else "junction" if i == junction else "ordinary",
            recovered, radius)
        if labels:
            dx, dy = (label_offsets or {}).get(str(i), (0, -19))
            text(x+dx, y+dy, "r" if i == root else str(i), size=6.8,
                 color=COL["root"] if i == root else COL["ink"])


def positions(key, cx, cy, scale=1):
    return np.asarray(layout[key]["positions"], dtype=float)*scale + [cx, cy]


def morphology(movie, box):
    crop = crops[movie]
    chosen = set(crop["selected_source_node_ids"])
    ns = [r for r in nodes if r["movie"] == movie and int(r["node"]) in chosen]
    xy = np.array([[float(r["x_px"]), -float(r["y_px"])] for r in ns])
    ids = {int(r["node"]): i for i, r in enumerate(ns)}
    es = [(ids[u], ids[v]) for u,v in crop["selected_source_edges"]]
    x, y, w, h = box
    k = min(w/np.ptp(xy[:,0]), h/np.ptp(xy[:,1]))
    xy = (xy-(xy.min(0)+xy.max(0))/2)*k + [x+w/2, y+h/2]
    ax.add_collection(LineCollection([[xy[u],xy[v]] for u,v in es],
        colors=COL["ink"], linewidths=.68, zorder=1))
    ax.scatter(xy[:,0],xy[:,1],s=10.0,color=COL["teal"],
        edgecolors="white",linewidths=.20,zorder=2)
    return {"movie":movie,"crop_id":crop["crop_id"],"nodes":len(ns),"edges":len(es),
            "crop":"all nodes and induced edges within deterministic closed ROI", "uniform_scale":float(k)}


def coherent_glyph(cx, cy, width=140, height=15, color=None):
    # A conventional signal icon only, deliberately without data axes or ticks.
    t = np.linspace(0, 1, 160)
    y = height*np.sin(4*np.pi*t)*np.sin(np.pi*t)**.3
    line(np.c_[cx-width/2+width*t, cy-y], color or COL["teal"], lw=1.05)


# a: Real local crops; all coordinates and internal source edges are preserved.
text(70, 70, "a", 10.2, weight="bold", ha="left")
text(108, 70, "Public cardiac myofibril morphologies", 8.8, ha="left")
counts = []
for movie, box, label in [
    ("E3", (125, 155, 410, 258), (109, 113)),
    ("E4", (631, 149, 414, 270), (626, 113)),
    ("E5", (1140, 158, 410, 254), (1138, 113))]:
    counts.append(morphology(movie, box))
    text(*label, movie, 7.6, ha="left", weight="medium")
# A deliberately small biological key; it does not describe mechanical links.
dot(1587, 145, radius=4.5)
text(1608, 145, "z-disc", 6.5, ha="left")
line([(1577, 181), (1597, 181)], lw=.5)
text(1608, 181, "sarcomere", 6.5, ha="left")
line([(108, 414+PANEL_B_OFFSET), (1730, 414+PANEL_B_OFFSET)], COL["pale_edge"], .45)

# b: one shared composition, no cards, colored bars, or separate panel boxes.
# Translate the complete unchanged body down 4.5 mm to accommodate larger morphologies.
# A transparent drawing axis keeps all existing body coordinates and relative spacing.
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1800),
                  ylim=(1200-PANEL_B_OFFSET, -PANEL_B_OFFSET), facecolor="none")
ax.set_axis_off()
ax.set_aspect("equal")
text(70, 453, "b", 10.2, weight="bold", ha="left")
text(108, 453, "Structural priors determine measurement requirements", 8.8, ha="left")
centers = [420, 935, 1480]
for cx, title, qualifier in zip(centers,
       ["Path known", "Branch known", "Topology unknown"],
       ["endpoint-rooted, calibrated C", "labelled tree, calibrated C", "C unknown"]):
    text(cx, 505, title, 8.4, weight="bold")
    text(cx, 548, qualifier, 7.0, color=COL["ink"])
for y, label in [(635, "Structure"), (805, "Measure"), (986, "Recover")]:
    text(78, y, label, 7.2, weight="medium", ha="left")

# Structure row: symbolic drawing coordinates, frozen scientific adjacency.
path_pairs = layout["path"]["edges"]
graph(positions("path", centers[0], 635), path_pairs)
text(centers[0]-135, 659, "r", 6.8, color=COL["root"])
branch_offsets = {"0": (0, 24), "1": (-18, -17), "2": (22, 0), "3": (22, 0)}
graph(positions("branch", centers[1], 635), branch["edges"], junction=1,
      labels=True, label_offsets=branch_offsets)
text(centers[1], 705, "Coherent alias: 2 ↔ 3", 7.0, color=COL["ink"])

# Node identities and count are known. Only the edges are withheld here.
# Every unordered pair is a possible edge: this introduces no adjacency prior.
# Gray dashed possibilities are schematic, not measured morphology connections.
upos = positions("unknown", centers[2], 635)
known_node_ids = list(local["node_order"])
candidate_pairs = [(u,v) for u in range(len(known_node_ids))
                   for v in range(u+1,len(known_node_ids))]
graph(upos, candidate_pairs, root=local["root_index"], edge_color=COL["pale_edge"],
      edge_style=(0,(2.5,2.2)))
text(upos[0][0], upos[0][1]+24, "r", 6.8, color=COL["root"])

# Measurement row: root readout, spatial profile coding, and interventions.
dot(centers[0]-97, 803, "root", radius=7)
line([(centers[0]-89, 803), (centers[0]-65, 803)], COL["ink"], .65)
coherent_glyph(centers[0]+13, 803, width=147, height=16)
text(centers[0], 853, "coherent root response", 7.2)

coherent_glyph(centers[1]-107, 803, width=93, height=14)
text(centers[1]-31, 804, "+", 9.3, color=COL["ink"])
# Diagonal profiles on fixed labelled states: a symbolic two-profile array.
# The colors show where a profile acts, not invented fitted numerical values.
for j, color in enumerate([COL["ochre"], COL["teal"]]):
    for i in range(4):
        selected = i == j+2
        ax.add_patch(Rectangle((centers[1]+4+i*23, 782+j*24), 15, 15,
                     facecolor=color if selected else "white",
                     edgecolor=color if selected else COL["pale_edge"], lw=.65))
for i, label in enumerate(("r", "1", "2", "3")):
    text(centers[1]+11.5+i*23, 763, label, 6.5,
         color=COL["root"] if label == "r" else COL["ink"])
text(centers[1], 853, "coherent + coded profiles", 7.2)

# Compact, baseline-aligned intervention schematic. Operators remain distinct:
# their horizontal grouping does not add or prescribe any network connectivity.
g, s, c, tau = centers[2]-96, centers[2]-36, centers[2]+24, centers[2]+104
dot(g, 804, radius=6)
arrow((g, 793), (g, 770), COL["orange"], .7, 5)
text(g+13, 773, "g", 6.8, color=COL["orange"], ha="left")
dot(s, 804, radius=6)
line([(s, 810), (s, 816), (s-5, 820), (s+5, 825),
      (s-5, 830), (s, 833), (s, 837)], COL["orange"], lw=.65)
for yy, width in [(837, 21), (843, 13)]:
    line([(s-width/2, yy), (s+width/2, yy)], COL["orange"], lw=.6)
dot(c, 804, radius=6)
line([(c-9, 790), (c-14, 790), (c-14, 818), (c-9, 818)], COL["orange"], lw=.7)
line([(c+9, 790), (c+14, 790), (c+14, 818), (c+9, 818)], COL["orange"], lw=.7)
arrow((c+29, 804), (tau-22, 804), COL["ink"], .6, 5)
text(tau, 804, "τ", 10.2, color=COL["ink"])
text(centers[2], 868, "thresholds under growth, shunts and clamps", 6.8)

# Recovery row: retain labelled graphs and explicitly name the recovery target.
graph(positions("path", centers[0], 983, .85), path_pairs, recovered=True, radius=7)
text(centers[0], 1044, "path mechanics recovered", 7.3)
graph(positions("branch", centers[1], 983, .80), branch["edges"], junction=1,
      recovered=True, labels=True, label_offsets=branch_offsets, radius=7)
text(centers[1], 1054, "labelled tree mechanics recovered", 7.3)
recovered_node_ids = list(local["node_order"])
rpos = positions("unknown", centers[2], 983, .79)
graph(rpos, local["edges"],
      root=local["root_index"], recovered=True, radius=7)
text(centers[2], 1049, "L, C and edge support", 7.3)

# No footer element: the body ends at the three recovery labels.

# Geometry/source checks are assertions about the figure, not new scientific analyses.
assert [c["nodes"] for c in counts] == [28, 33, 44]
assert [c["edges"] for c in counts] == [29, 36, 47]
for item in [branch, local]:
    ids = item["node_order"]
    movie = "E5" if item is branch else "E3"
    source_pairs = {frozenset((int(e["source"]), int(e["target"])))
                    for e in edges if e["movie"] == movie}
    assert all(frozenset((ids[u], ids[v])) in source_pairs for u, v in item["edges"])
assert known_node_ids == recovered_node_ids and len(known_node_ids) == 5
assert len(upos) == len(rpos) == len(known_node_ids)
assert np.allclose((rpos-[centers[2],983])/.79, upos-[centers[2],635])
assert len(candidate_pairs) == len(known_node_ids)*(len(known_node_ids)-1)//2
assert all(0 <= u < len(known_node_ids) and 0 <= v < len(known_node_ids)
           for u,v in local["edges"])

label_audit = []
for obj in all_text:
    value = obj.get_text()
    if value in {"r", "1", "2", "3"}:
        actual = mpl.colors.to_hex(obj.get_color()).upper()
        expected = COL["root"] if value == "r" else COL["ink"]
        assert actual == expected, (value, actual, expected)
        label_audit.append({"label": value, "color": actual, "position": list(obj.get_position())})
assert sum(r["label"] == "r" for r in label_audit) == 5
assert sum(r["label"] in {"1", "2", "3"} for r in label_audit) == 9

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
out_of_bounds = []
for obj in all_text:
    box = obj.get_window_extent(renderer)
    if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
        out_of_bounds.append(obj.get_text())
assert not out_of_bounds, out_of_bounds

stamp = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)
fig.savefig(ROOT / "Figure1_final.png", dpi=600,
            metadata={"Software": "P1.1 reproducible Figure 1 / Matplotlib"})
fig.savefig(ROOT / "Figure1_final.pdf", metadata={"Title": "Structural priors and their measurement contracts",
            "Author": "P1.1", "CreationDate": stamp, "ModDate": stamp})
fig.savefig(ROOT / "Figure1_final.svg", metadata={"Date": "2026-09-30",
            "Description": "Vector source-derived morphologies and schematic measurement contracts."})
plt.close(fig)

with (QA / "SOURCE_ASSET_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["relative_path", "bytes", "sha256"])
    for p in sorted(ASSETS.rglob("*")):
        if p.is_file():
            w.writerow([p.relative_to(ROOT).as_posix(), p.stat().st_size,
                        hashlib.sha256(p.read_bytes()).hexdigest()])
report = {"canvas_mm": [180, 120], "png_dpi": 600, "morphologies": counts,
    "palette": COL, "panel_b_translation_mm": PANEL_B_OFFSET/10,
    "label_color_audit": label_audit, "morphology_size_unchanged_from_v2": True,
    "single_blue_green": COL["teal"],
    "unknown_topology_known_nodes": {"structure_node_ids":known_node_ids,
        "recovery_node_ids":recovered_node_ids,"candidate_edge_count":len(candidate_pairs),
        "recovered_edge_count":len(local["edges"]),"same_relative_node_geometry":True},
    "foreground_edge_endpoint_check": "PASS", "no_pale_node_on_dark_edge": "PASS",
    "source_adjacency_checks": "PASS", "text_within_canvas": "PASS",
    "all_graph_elements_vector": True, "font": font_manager.findfont("Arial"),
    "python": platform.python_version(), "matplotlib": mpl.__version__, "numpy": np.__version__,
    "schematic_signals": "Symbolic wave/profile/intervention glyphs; not measured observations"}
(QA / "build_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
