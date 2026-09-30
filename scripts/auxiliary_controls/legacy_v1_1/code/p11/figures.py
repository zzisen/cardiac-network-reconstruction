from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#626262"
LIGHT = "#E8EEF2"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _finish(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panel(ax, label: str):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _draw_graph(ax, positions, edges, *, root=0, node_labels=None, title=None):
    for u, v in edges:
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        ax.plot([x0, x1], [y0, y1], color=GRAY, lw=1.5, zorder=1)
    for node, (x, y) in positions.items():
        color = ORANGE if node == root else LIGHT
        ax.scatter([x], [y], s=420, color=color, edgecolor=BLUE, linewidth=1.1, zorder=2)
        label = node_labels[node] if node_labels else str(node)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="black", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, pad=4)


def make_figures(results_dir: Path, figure_dir: Path) -> None:
    _style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir = figure_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    exact = pd.read_csv(results_dir / "exact_recovery.csv")
    scaling = pd.read_csv(results_dir / "scaling.csv")
    noise = pd.read_csv(results_dir / "noise_summary.csv")
    mismatch = pd.read_csv(results_dir / "mismatch_observations.csv")
    arch = pd.read_csv(results_dir / "architecture_repair_exact.csv")
    alias = pd.read_csv(results_dir / "branch_alias_frequency.csv")
    cycle = pd.read_csv(results_dir / "cyclic_k9_matrices.csv")
    arch_noise = pd.read_csv(results_dir / "architecture_noise.csv")

    # Figure 1: coarse-grained branching mechanics, path negative control, regimes.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    ax.set_xlim(-1.1, 5.3)
    ax.set_ylim(-1.45, 1.65)
    ax.axis("off")
    positions = {0: (0.0, 0.0), 1: (1.5, 0.0), 2: (3.0, 0.85), 3: (3.0, -0.85)}
    _draw_graph(ax, positions, [(0, 1), (1, 2), (1, 3)], root=0,
                node_labels={0: "root", 1: "junction", 2: "daughter", 3: "daughter"},
                title="A  Effective coarse-grained branching graph")
    ax.text(1.45, -1.28, r"$C\dot z + [\mathrm{diag}(a_i)+B\,\mathrm{diag}(\kappa_e)B^T]z=u$",
            ha="center", va="center", fontsize=8.3)
    ax = axes[1]
    ax.axis("off")
    ax.text(0.02, 0.98, "B  Two inverse regimes", transform=ax.transAxes, fontweight="bold", va="top")
    rows = [
        ("K8", "known labelled tree; known C", "coherent root response +\nfixed diagonal-noise profiles", "recover normalized J"),
        ("K9", "unknown connected graph; unknown C", "ideal shunts, feedback,\nclamps, and thresholds", "recover effective L, C, graph"),
    ]
    y = 0.76
    for method, assumptions, design, target in rows:
        ax.add_patch(plt.Rectangle((0.02, y - 0.24), 0.94, 0.33, transform=ax.transAxes,
                                   facecolor="#F7F9FA", edgecolor="#AAB7C0", lw=0.8))
        ax.text(0.06, y + 0.035, method, transform=ax.transAxes,
                color=BLUE if method == "K8" else ORANGE, fontweight="bold", fontsize=10, va="center")
        ax.text(0.21, y + 0.07, assumptions, transform=ax.transAxes, fontsize=7.2, va="center")
        ax.text(0.21, y - 0.035, design, transform=ax.transAxes, fontsize=7.0, va="center", color=GRAY)
        ax.text(0.21, y - 0.145, target, transform=ax.transAxes, fontsize=7.0, va="center", color=GREEN)
        y -= 0.39
    ax.text(0.03, 0.02,
            "Endpoint path control: its root Weyl function already identifies the Jacobi path (no K8 profiles).\n"
            "Branching motivates the graph class; fixtures are synthetic, not extracted adjacency.\n"
            "K9 operations are idealized in-silico queries, not validated laboratory procedures.",
            transform=ax.transAxes, fontsize=6.5, color=GRAY, va="bottom")
    fig.tight_layout(w_pad=1.3)
    _finish(fig, figure_dir / "Figure1_cardiac_reduction")
    pd.DataFrame([
        {"node": 0, "label": "root", "edge_u": 0, "edge_v": 1, "model_role": "synthetic Y-tree fixture"},
        {"node": 1, "label": "junction", "edge_u": 1, "edge_v": 2, "model_role": "synthetic Y-tree fixture"},
        {"node": 2, "label": "daughter", "edge_u": 1, "edge_v": 3, "model_role": "synthetic Y-tree fixture"},
        {"node": 3, "label": "daughter", "edge_u": np.nan, "edge_v": np.nan, "model_role": "synthetic Y-tree fixture"},
    ]).to_csv(source_dir / "figure1_source.csv", index=False)

    # Figure 2: coherent daughter-swap alias and K8 branch-profile recovery.
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [0.85, 1.2, 1.0]})
    ax = axes[0]
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    ax.plot([0.15, 0.65], [0, 0], color=GRAY, lw=1.4)
    ax.plot([0.65, 1.1], [0, 0.65], color=GRAY, lw=1.4)
    ax.plot([0.65, 1.1], [0, -0.65], color=GRAY, lw=1.4)
    for x, y, label, color in [(0.15, 0, "0", ORANGE), (0.65, 0, "1", LIGHT), (1.1, 0.65, "2", LIGHT), (1.1, -0.65, "3", LIGHT)]:
        ax.scatter([x], [y], s=230, color=color, edgecolor=BLUE, linewidth=1)
        ax.text(x, y, label, ha="center", va="center", fontsize=7)
    ax.set_title("Labelled Y tree")
    ax.text(0.38, -0.95, "swap labels 2 ↔ 3", ha="center", fontsize=7, color=PURPLE)

    ax = axes[1]
    coherent_diff = np.hypot(alias["coherent_original_real"] - alias["coherent_daughter_swapped_real"],
                             alias["coherent_original_imag"] - alias["coherent_daughter_swapped_imag"])
    ax.plot(alias["frequency"], alias["profile_power_difference"], "s--", color=ORANGE,
            label=r"$|G_{0,2}|^2-|G^{swap}_{0,2}|^2$")
    ax.set_xlabel("normalized frequency")
    ax.set_ylabel("fixed node-2 profile difference")
    ax.set_title("Coherent transfer aliases")
    ax.set_ylim(0, 0.007)
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.04, 0.96, f"max coherent difference = {coherent_diff.max():.0f}",
            transform=ax.transAxes, va="top", color=BLUE, fontsize=7.2)

    ax = axes[2]
    k8 = arch[arch["case"] == "K8_Y_tree_labelled"].iloc[0]
    ax.axis("off")
    ax.set_title("K8 recovers labelled Y")
    ax.text(0.03, 0.76, f"relative J error: {k8['J_or_L_relative_error']:.2e}", transform=ax.transAxes, fontsize=8)
    ax.text(0.03, 0.56, f"profiles used: 2 = n−2", transform=ax.transAxes, fontsize=8)
    ax.text(0.03, 0.36, f"max fixed-profile contrast: {k8['profile_node3_max_abs_difference']:.2e}", transform=ax.transAxes, fontsize=7.2)
    ax.text(0.03, 0.10, "Y fixture is synthetic; the daughter swap\nkeeps root transfer fixed but changes\nlabel-specific row power.",
            transform=ax.transAxes, fontsize=7, color=GRAY)
    for ax, letter in zip(axes, "ABC"):
        _panel(ax, letter)
    fig.tight_layout(w_pad=1.2)
    _finish(fig, figure_dir / "Figure2_K8_recovery")
    alias.assign(source="daughter-swap alias fixture").to_csv(source_dir / "figure2_source.csv", index=False)

    # Figure 3: exact K9 reconstruction on an unknown-topology cyclic graph.
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [0.85, 1.2, 1.0]})
    ax = axes[0]
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.1, 0.0), 3: (1.0, 1.0), 4: (1.0, -1.0)}
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 1)]
    _draw_graph(ax, pos, edges, root=0, title="True five-node network")
    ax.set_xlim(-0.8, 2.9)
    ax.set_ylim(-2.1, 1.6)
    ax.text(1.0, -1.75, "cycle 1–3–4–1; root = 0", ha="center", fontsize=6.7, color=GRAY)

    matrix = cycle[cycle["i"].between(0, 4) & cycle["j"].between(0, 4)]
    Ltrue = np.zeros((5, 5))
    Lhat = np.zeros((5, 5))
    Ctrue = np.zeros(5)
    Chat = np.zeros(5)
    for row in matrix.itertuples():
        i, j = int(row.i), int(row.j)
        Ltrue[i, j] = row.L_true
        Lhat[i, j] = row.L_recovered
        if i == j:
            Ctrue[i] = row.C_true_diagonal
            Chat[i] = row.C_recovered_diagonal
    vmax = max(np.max(np.abs(Ltrue)), np.max(np.abs(Lhat)))
    axes[1].imshow(np.concatenate((Ltrue, Lhat), axis=1), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_title("L true | recovered")
    axes[1].set_xticks([0, 2, 4, 5, 7, 9], ["0", "2", "4", "0", "2", "4"])
    axes[1].set_yticks(range(5), range(5))
    axes[1].set_ylabel("node")
    axes[1].set_xlabel("node index in each matrix")

    ax = axes[2]
    x = np.arange(5)
    ax.plot(x, Ctrue, "o", color=BLUE, label="true C")
    ax.plot(x, Chat, "x", color=ORANGE, label="recovered C")
    ax.set_xticks(x)
    ax.set_xlabel("node")
    ax.set_ylabel("diagonal storage")
    ax.set_title("Joint recovery")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    case = arch[arch["case"] == "K9_cyclic_general_graph"].iloc[0]
    ax.text(0.03, 0.95,
            f"L rel. error {case['J_or_L_relative_error']:.1e}\nC rel. error {case['C_relative_error']:.1e}\n"
            f"queries {int(case['query_or_profile_count'])} = n(n+3)/2",
            transform=ax.transAxes, va="top", fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    for ax, letter in zip(axes, "ABC"):
        _panel(ax, letter)
    fig.tight_layout(w_pad=1.2)
    _finish(fig, figure_dir / "Figure3_K9_joint_recovery")
    cycle.to_csv(source_dir / "figure3_source.csv", index=False)

    # Figure 4: conditioning/noise limits, with the path retained as a control.
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.25))
    positions = np.arange(4)
    labels = ["0", "1e-10", "1e-8", "1e-6"]
    branch_noise = arch_noise[(arch_noise["method"] == "K8") & (arch_noise["fit_success"] == True)]
    med = branch_noise.groupby("relative_noise_scale")["fit_relative_parameter_error"].median().reindex([0.0, 1e-10, 1e-8, 1e-6])
    axes[0, 0].plot(positions, np.maximum(med.to_numpy(dtype=float), 1e-14), "o-", color=BLUE)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set_xlabel("relative observation-noise scale")
    axes[0, 0].set_ylabel("median relative J error")
    axes[0, 0].set_title("K8 known-Y constrained fit")
    axes[0, 0].grid(axis="y", alpha=0.2)

    # Exact-inverse acceptance: path benchmark versus the repaired branch/cycle cases.
    ax = axes[0, 1]
    series = [
        ("K8 path", noise[noise["method"] == "K8"], BLUE, "o"),
        ("K8 Y tree", arch_noise[arch_noise["method"] == "K8"], GREEN, "s"),
        ("K9 path", noise[noise["method"] == "K9"], PURPLE, "^"),
        ("K9 cycle", arch_noise[arch_noise["method"] == "K9"], ORANGE, "D"),
    ]
    for label, frame, color, marker in series:
        if "exact_inverse_status" in frame:
            group = frame.groupby("relative_noise_scale")["exact_inverse_status"].apply(
                lambda s: float((s == "accepted").mean())
            )
        else:
            group = frame.groupby("relative_noise_scale")["exact_inverse_acceptance_fraction"].mean()
        group = group.reindex([0.0, 1e-10, 1e-8, 1e-6])
        ax.plot(positions, group.to_numpy(dtype=float), marker=marker, color=color, lw=1.1, label=label)
    ax.set_xticks(positions, labels)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("relative observation-noise scale")
    ax.set_ylabel("exact-inverse acceptance fraction")
    ax.set_title("Exact algorithms reject inconsistent inputs")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=6.3)

    ax = axes[1, 0]
    accepted_cycle = arch_noise[(arch_noise["method"] == "K9") & (arch_noise["exact_inverse_status"] == "accepted")]
    cycle_median = accepted_cycle.groupby("relative_noise_scale")["exact_relative_parameter_error"].median().reindex([0.0, 1e-10, 1e-8, 1e-6])
    ax.plot(positions, np.maximum(cycle_median.to_numpy(dtype=float), 1e-14), "D--", color=ORANGE,
            label="K9 cycle, accepted trials")
    path_noise = noise[noise["method"] == "K9"].groupby("relative_noise_scale")["median_finite_fit_error"].median().reindex([0.0, 1e-10, 1e-8, 1e-6])
    ax.plot(positions, np.maximum(path_noise.to_numpy(dtype=float), 1e-14), "o-", color=PURPLE,
            label="K9 path, constrained fit")
    ax.set_yscale("log")
    ax.set_xticks(positions, labels)
    ax.set_xlabel("relative observation-noise scale")
    ax.set_ylabel("median relative L error")
    ax.set_title("Conditioning depends on contract and topology")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.text(0.03, 0.97, "K9 cycle: 0/8 accepted for noise ≥ 1e−10", transform=ax.transAxes,
            va="top", fontsize=6.4, color=GRAY)

    ax = axes[1, 1]
    for site, sub in mismatch.groupby("injected_site"):
        ax.plot(sub["frequency_hz"], sub["psd_difference"], marker="o", ms=3, lw=0.9, label=f"site {int(site)+1}")
    ax.axhline(0.0, color=GRAY, lw=0.8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("paired root PSD difference")
    ax.set_title("Nonlinear-model mismatch probe")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    for ax, letter in zip(axes.flat, "ABCD"):
        _panel(ax, letter)
    fig.suptitle("Synthetic perturbation trials; no biological replicates", y=1.02, fontsize=8.5)
    fig.tight_layout()
    _finish(fig, figure_dir / "Figure4_noise_and_mismatch")
    arch_noise.to_csv(source_dir / "figure4_architecture_noise_source.csv", index=False)
    noise.to_csv(source_dir / "figure4_path_noise_control_source.csv", index=False)
    mismatch.to_csv(source_dir / "figure4_mismatch_source.csv", index=False)
