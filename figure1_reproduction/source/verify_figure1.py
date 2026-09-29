"""Verify the final freeze without writing preview exports into the active package.

Run: python source/verify_figure1.py
Optional: --proof-dir /external/directory (requires Poppler).
Validation dependencies: pypdf, PyMuPDF, Pillow. Building needs only requirements.txt.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import pymupdf as fitz
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"source"/"inputs"
RECORDS = ROOT/"source"/"records"
args = argparse.ArgumentParser()
args.add_argument("--proof-dir", type=Path)
opts = args.parse_args()
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names = [f"Figure1_final.{ext}" for ext in ("png","pdf","svg")]
repeat_files = [ROOT/n for n in names] + [RECORDS/n for n in
    ("CROP_SELECTION.json","CROP_NODES.csv","CROP_EDGES.csv")]
before = {str(p.relative_to(ROOT)):digest(p) for p in repeat_files}
subprocess.run([sys.executable,str(ROOT/"source"/"build_figure1.py")],
               check=True,capture_output=True)
after = {str(p.relative_to(ROOT)):digest(p) for p in repeat_files}
assert before == after, "Repeat build changed artwork or crop records"
assert digest(DATA/"public_graph_nodes.csv") == "fc755a10656a82f68f8e7a13a666da4210ede1834072df29eddfff2ff8871d66"
assert digest(DATA/"public_graph_edges.csv") == "b9abd910bb9c495814cfd4239896373b45a5f7d0ab4211eff2bcdf28367f4735"

pdf = PdfReader(ROOT/"Figure1_final.pdf")
assert len(pdf.pages)==1 and len(list(pdf.pages[0].images))==0
size = [float(pdf.pages[0].mediabox.width)*25.4/72,
        float(pdf.pages[0].mediabox.height)*25.4/72]
assert all(abs(a-b)<.001 for a,b in zip(size,[180,120]))
txt = pdf.pages[0].extract_text()
required = ["Public cardiac myofibril morphologies", "Path known", "endpoint-rooted, calibrated C",
    "Branch known", "labelled tree, calibrated C", "Topology unknown", "C unknown",
    "coherent root response", "coherent + coded profiles", "thresholds under growth, shunts and clamps",
    "path mechanics recovered", "labelled tree mechanics recovered", "L, C and edge support",
    "Structural priors determine measurement requirements"]
assert all(s in txt for s in required)
assert all(s not in txt for s in ["Representative local crops", "structural prior decreases",
    "measurement contract changes", "· J", "endpoint rooted", "K8", "K9",
    "path identified", "branch labels resolved", "Structure changes the measurement contract"])

# Verify exported PDF text colors independently of the drawing script.
doc = fitz.open(ROOT/"Figure1_final.pdf")
labels = []
for block in doc[0].get_text("dict")["blocks"]:
    for row in block.get("lines",[]):
        for span in row["spans"]:
            value = span["text"].strip()
            color = f'#{span["color"]:06X}'
            if value == "r":
                assert color == "#E5543D", (value,color)
                labels.append({"label":value,"color":color})
            elif re.fullmatch(r"[123\s]+",value):
                assert color == "#3F4548", (value,color)
                labels.extend({"label":ch,"color":color} for ch in value if ch in "123")
            elif "Coherent alias:" in value:
                assert color == "#3F4548"
assert sum(x["label"]=="r" for x in labels)==5
assert sum(x["label"] in {"1","2","3"} for x in labels)==9

svg_path = ROOT/"Figure1_final.svg"
svg = ET.parse(svg_path).getroot()
assert not any(e.tag.endswith("}image") or e.tag.endswith("}text") for e in svg.iter())
used = {c.upper() for c in re.findall(r"(?:fill|stroke):\s*(#[a-fA-F0-9]{6})",svg_path.read_text(encoding="utf-8"))}
allowed = {"#16A6A1","#F2B33D","#F47C20","#E5543D",
           "#3F4548","#D8E0E3","#FFFFFF"}
assert used <= allowed, used-allowed
assert {"#16A6A1","#F2B33D","#F47C20","#E5543D","#3F4548","#D8E0E3"} <= used
assert "#0B7C90" not in used and "#5CBF8A" not in used
build=json.loads((RECORDS/"build_checks.json").read_text())
contract=build["unknown_topology_known_nodes"]
assert contract["structure_node_ids"]==contract["recovery_node_ids"]==[159,49,62,240,127]
assert contract["candidate_edge_count"]==10 and contract["recovered_edge_count"]==5
assert contract["same_relative_node_geometry"]
assert "background_positions" not in json.loads((DATA/"schematic_layout.json").read_text())["unknown"]
with Image.open(ROOT/"Figure1_final.png") as im:
    pixels,dpi=list(im.size),list(im.info["dpi"])
assert all(abs(v-600)<.01 for v in dpi)

def table(p):
    with p.open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
crops=json.loads((RECORDS/"CROP_SELECTION.json").read_text())
nodes,edges=table(DATA/"public_graph_nodes.csv"),table(DATA/"public_graph_edges.csv")
cn,ce=table(RECORDS/"CROP_NODES.csv"),table(RECORDS/"CROP_EDGES.csv")
frozen=json.loads((DATA/"public_subgraph_selection.json").read_text())
for movie,crop in crops.items():
    b=crop["bounds_source_px"]
    ns=[r for r in nodes if r["movie"]==movie and b["xmin"]<=float(r["x_px"])<=b["xmax"]
        and b["ymin"]<=float(r["y_px"])<=b["ymax"]]
    keep={int(r["node"]) for r in ns}
    es=[r for r in edges if r["movie"]==movie and int(r["source"]) in keep and int(r["target"]) in keep]
    assert ns==[r for r in cn if r["movie"]==movie]
    assert es==[r for r in ce if r["movie"]==movie]
    assert crop["selected_source_node_ids"]==sorted(keep)
    assert crop["bfs24_source_node_order"][:5]==frozen["graphs"][movie]["k9"]["5"]["node_order"]

if opts.proof_dir:
    external=opts.proof_dir.resolve()
    assert not external.is_relative_to(ROOT), "Keep proof images outside the active package"
    poppler=shutil.which("pdftoppm")
    if not poppler: raise RuntimeError("Poppler pdftoppm is required for --proof-dir")
    external.mkdir(parents=True,exist_ok=True)
    for options,name in [(["-scale-to","1800"],"Figure1_pdfproof"),
                          (["-scale-to","900"],"Figure1_pdfproof_50pct"),
                          (["-r","150"],"Figure1_pdfproof_150dpi")]:
        subprocess.run([poppler,*options,"-singlefile","-png",str(ROOT/"Figure1_final.pdf"),
                        str(external/name)],check=True,capture_output=True)
report={"repeat_build_artwork_and_crops_byte_identical":True,
        "one_shared_blue_green":"#16A6A1",
        "unknown_topology_same_five_nodes_before_after":"PASS",
        "palette_only_requested_colors_plus_white":True,"exported_svg_colors":sorted(used),
        "pdf_root_labels_all_coral":5,"pdf_numeric_node_labels_all_graphite":9,
        "pdf_label_color_details":labels,"source_hashes_and_crop_membership":"PASS",
        "accepted_wording_and_removed_artifact_check":"PASS","pdf_page_mm":size,
        "png_pixels":pixels,"png_dpi":dpi,"pdf_svg_raster_images":0,
        "output_sha256":{n:digest(ROOT/n) for n in names}}
(RECORDS/"FINAL_QA.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
with (RECORDS/"OUTPUT_SHA256.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["file","sha256"]);w.writerows(report["output_sha256"].items())
print(json.dumps(report,indent=2))
