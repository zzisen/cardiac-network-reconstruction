"""Shared vector-first figure style with an open-font fallback."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

TEAL, GOLD, CORAL, ORANGE = "#16A6A1", "#F2B33D", "#E5543D", "#F47C20"
INK, PALE = "#3F4548", "#D8E0E3"
COLORS = {"moderate_redundancy": GOLD, "high_redundancy": TEAL}
DESIGNS = list(COLORS)
MARKERS = {"moderate_redundancy": "D", "high_redundancy": "o"}
ACTIVE_FONT = None


def _font_family(preference):
    if preference == "public":
        font_manager.findfont("DejaVu Sans", fallback_to_default=False)
        return "DejaVu Sans"
    if preference != "auto":
        raise ValueError("font family must be 'auto' or 'public'")
    for name in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        candidate = Path("C:/Windows/Fonts") / name
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial"
    except (ValueError, OSError):
        font_manager.findfont("DejaVu Sans", fallback_to_default=False)
        return "DejaVu Sans"


def setup(font_family="auto"):
    global ACTIVE_FONT
    ACTIVE_FONT = _font_family(font_family)
    plt.rcParams.update({
        "font.family": ACTIVE_FONT, "font.size": 7.2, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 1, "lines.linewidth": 1.4,
        "patch.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1,
        "xtick.minor.width": 1, "ytick.minor.width": 1, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.labelsize": 7.4, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "P11-final-20260930",
        "axes.unicode_minus": True, "savefig.facecolor": "white",
    })
    if ACTIVE_FONT == "Arial":
        plt.rcParams.update({
            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial", "mathtext.tt": "Arial",
            "mathtext.cal": "Arial", "mathtext.fallback": None,
        })
    else:
        plt.rcParams.update({"mathtext.fontset": "dejavusans"})
    return ACTIVE_FONT


def page(height):
    f = plt.figure(figsize=(180/25.4, height/25.4), facecolor="white")
    f.mm_height = height
    return f


def axes(f, x, y, w, h):
    a = f.add_axes([x/180, 1-(y+h)/f.mm_height, w/180, h/f.mm_height])
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(length=3, pad=3)
    return a


def text(f, x, y, s, size=7.2, **kw):
    return f.text(x/180, 1-y/f.mm_height, s, fontsize=size, va="top", **kw)


def title(f, x, y, letter, s):
    text(f, x, y, letter, 10, fontweight="bold")
    text(f, x+5, y+0.2, s, 8.2, fontweight="bold")


def line(f, x0, y0, x1, y1, color=PALE, ls="-"):
    f.add_artist(Line2D([x0/180, x1/180], [1-y0/f.mm_height, 1-y1/f.mm_height],
                        transform=f.transFigure, color=color, lw=1, ls=ls))


def design_handles():
    return [Line2D([], [], color=COLORS[d], marker=MARKERS[d],
                   mfc="white" if i == 0 else COLORS[d], ls="--" if i == 0 else "-",
                   ms=4.2, label=f"{[37, 77][i]} reads") for i, d in enumerate(DESIGNS)]


def metric_handles():
    return [Line2D([], [], color=TEAL, marker=m, mfc=TEAL if k == 0 else "white",
                   ls=["-", "--"][k], ms=4, label=["L", "C"][k])
            for k, m in enumerate(["o", "s"])]


def save(f, out, num):
    dest = Path(out) / "figures"
    dest.mkdir(exist_ok=True, parents=True)
    for ext in ["pdf", "svg", "png"]:
        meta = {"Creator": "P1.1 reproducible figure build"} if ext == "pdf" else None
        if ext == "pdf":
            meta.update(CreationDate=None, ModDate=None)
        if ext == "svg":
            meta = {"Date": None, "Creator": "P1.1 reproducible figure build"}
        f.savefig(dest / f"Figure{num}_final.{ext}", dpi=600, metadata=meta)
    plt.close(f)
