#!/usr/bin/env python3
"""
generate_ensemble_selection_figures.py
======================================
Generates thesis figures for the validation-selected ensemble analysis (Section 7.7).

Outputs:
    results/figures/ch7_fig9_val_selection_strategies.pdf
    results/figures/ch7_fig10_ensemble_comparison.pdf

Run:
    python scripts/visualization/generate_ensemble_selection_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

C_CNN  = "#1f77b4"
C_RE3D = "#ff7f0e"
C_25D  = "#2ca02c"
C_ENS  = "#d62728"
C_GRAY = "#888888"


def savefig(fig, name):
    for fmt in ("pdf", "png"):
        path = OUT / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight")
    print(f"  Saved -> {name}.pdf / .png")
    plt.close(fig)


# =============================================================================
# Figure 9: Validation selection strategies — grouped bar chart
# =============================================================================
def fig9_val_selection_strategies():
    """Side-by-side bars: Strategy 1 (Top-K) vs Strategy 2 (Best-N/arch)."""

    # Strategy 1: Top-K global
    s1_labels = ["Top-3", "Top-5", "Top-6", "Top-10", "Top-15"]
    s1_ds001  = [0.6927, 0.6984, 0.7026, 0.7078, 0.7148]
    s1_ds002  = [0.7909, 0.7953, 0.7982, 0.8017, 0.8026]
    s1_counts = [3, 5, 6, 10, 15]
    # Arch mix for annotation
    s1_arch   = ["3×2.5D", "4×2.5D\n1×RE3D", "4×2.5D\n1×CNN\n1×RE3D",
                 "5×2.5D\n2×CNN\n3×RE3D", "All 15"]

    # Strategy 2: Best-N/arch
    s2_labels = ["Best-1\n/arch", "Best-2\n/arch", "Best-3\n/arch", "All-15"]
    s2_ds001  = [0.7145, 0.7179, 0.7159, 0.7148]
    s2_ds002  = [0.8008, 0.8032, 0.8030, 0.8026]
    s2_counts = [3, 6, 9, 15]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    # ── Left panel: DS001 ──
    ax = axes[0]
    x1 = np.arange(len(s1_labels))
    x2 = np.arange(len(s2_labels)) + len(s1_labels) + 1
    w = 0.38

    bars1 = ax.bar(x1, s1_ds001, w, color=C_GRAY, alpha=0.6, label="Strategy 1 (Top-K global)")
    bars2 = ax.bar(x2, s2_ds001, w, color=C_ENS, alpha=0.85, label="Strategy 2 (Best-N/arch)")

    # Highlight best
    best_idx = np.argmax(s2_ds001)
    bars2[best_idx].set_edgecolor("black")
    bars2[best_idx].set_linewidth(2)

    ax.set_xticks(np.concatenate([x1, x2]))
    ax.set_xticklabels(s1_labels + s2_labels, fontsize=8)
    ax.set_ylabel("Test DSC")
    ax.set_title("DS001 — MSLesSeg (n=22)")
    ax.set_ylim(0.68, 0.73)
    ax.axhline(0.714, color="purple", ls="--", lw=1, alpha=0.7)
    ax.text(len(s1_labels) + len(s2_labels) - 0.5, 0.7145, "MadSeg (0.714)",
            ha="right", va="bottom", fontsize=8, color="purple")
    ax.legend(loc="lower right", fontsize=8)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # Separator
    sep_x = len(s1_labels) + 0.3
    ax.axvline(sep_x, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(sep_x, ax.get_ylim()[1] - 0.001, "│", ha="center", fontsize=6, color="gray")

    # ── Right panel: DS002 ──
    ax = axes[1]
    bars1 = ax.bar(x1, s1_ds002, w, color=C_GRAY, alpha=0.6, label="Strategy 1 (Top-K global)")
    bars2 = ax.bar(x2, s2_ds002, w, color=C_ENS, alpha=0.85, label="Strategy 2 (Best-N/arch)")

    best_idx = np.argmax(s2_ds002)
    bars2[best_idx].set_edgecolor("black")
    bars2[best_idx].set_linewidth(2)

    ax.set_xticks(np.concatenate([x1, x2]))
    ax.set_xticklabels(s1_labels + s2_labels, fontsize=8)
    ax.set_ylabel("Test DSC")
    ax.set_title("DS002 — WMH (n=110)")
    ax.set_ylim(0.785, 0.81)
    ax.axhline(0.80, color="purple", ls="--", lw=1, alpha=0.7)
    ax.text(len(s1_labels) + len(s2_labels) - 0.5, 0.8005, "sysu_media (0.80)",
            ha="right", va="bottom", fontsize=8, color="purple")
    ax.legend(loc="lower right", fontsize=8)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
                f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    sep_x = len(s1_labels) + 0.3
    ax.axvline(sep_x, color="gray", ls=":", lw=0.8, alpha=0.5)

    fig.suptitle("Validation-Selected Ensemble: Strategy Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "ch7_fig9_val_selection_strategies")


# =============================================================================
# Figure 10: Master ensemble comparison — horizontal bar chart
# =============================================================================
def fig10_ensemble_comparison():
    """Horizontal bar chart comparing all ensemble approaches + benchmarks."""

    labels = [
        "ResEncL-2.5D 5-fold",
        "ResEncL-3D 5-fold",
        "CNN-3D 5-fold",
        "MadSeg (ICPR 2024)",
        "All-15 (no selection)",
        "Best-2/arch (val EMA)",
        "Oracle (test-selected)",
    ]
    ds001 = [0.6930, 0.7073, 0.7088, 0.714, 0.7148, 0.7179, 0.7213]
    ds002 = [0.7922, 0.7986, 0.7856, None,  0.8026, 0.8032, 0.8055]

    colors_ds001 = [C_25D, C_RE3D, C_CNN, "purple", C_GRAY, C_ENS, "#aaa"]
    hatches      = ["", "", "", "", "", "", "//"]
    edge_colors  = ["none"]*5 + ["black", "none"]
    edge_widths  = [0]*5 + [2, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    # DS001
    ax = axes[0]
    y = np.arange(len(labels))
    for i, (val, c, h, ec, ew) in enumerate(zip(ds001, colors_ds001, hatches, edge_colors, edge_widths)):
        alpha = 0.4 if h == "//" else 0.85
        ax.barh(i, val, color=c, alpha=alpha, hatch=h, edgecolor=ec, linewidth=ew)
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.68, 0.73)
    ax.set_xlabel("Test DSC")
    ax.set_title("DS001 — MSLesSeg (n=22)")
    ax.invert_yaxis()

    # DS002
    ax = axes[1]
    for i, (val, c, h, ec, ew) in enumerate(zip(ds002, colors_ds001, hatches, edge_colors, edge_widths)):
        if val is None:
            ax.text(0.795, i, "N/A", va="center", fontsize=8, color="gray")
            continue
        alpha = 0.4 if h == "//" else 0.85
        ax.barh(i, val, color=c, alpha=alpha, hatch=h, edgecolor=ec, linewidth=ew)
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlim(0.78, 0.815)
    ax.set_xlabel("Test DSC")
    ax.set_title("DS002 — WMH (n=110)")
    ax.invert_yaxis()

    # Legend
    patches = [
        mpatches.Patch(color=C_CNN, label="CNN-3D"),
        mpatches.Patch(color=C_RE3D, label="ResEncL-3D"),
        mpatches.Patch(color=C_25D, label="ResEncL-2.5D"),
        mpatches.Patch(color=C_ENS, label="Best-2/arch (recommended)"),
        mpatches.Patch(color=C_GRAY, label="All-15 / benchmark"),
        mpatches.Patch(facecolor="#aaa", hatch="//", alpha=0.4, label="Oracle (test-selected)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Ensemble Comparison: All Approaches", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "ch7_fig10_ensemble_comparison")


# =============================================================================
# Figure 11: Val EMA vs Test DSC scatter — shows validation-test correlation
# =============================================================================
def fig11_val_vs_test_scatter():
    """Scatter: Val EMA (x) vs Test DSC (y) for all 15 models, both datasets."""

    val_ema = {
        "CNN_3D_fold0": 0.8291, "CNN_3D_fold1": 0.8418, "CNN_3D_fold2": 0.8242,
        "CNN_3D_fold3": 0.8356, "CNN_3D_fold4": 0.8259,
        "ResEncL_3D_fold0": 0.8135, "ResEncL_3D_fold1": 0.8430, "ResEncL_3D_fold2": 0.8334,
        "ResEncL_3D_fold3": 0.8344, "ResEncL_3D_fold4": 0.8317,
        "25D_fold0": 0.8385, "25D_fold1": 0.8582, "25D_fold2": 0.8507,
        "25D_fold3": 0.8522, "25D_fold4": 0.8476,
    }
    test_ds001 = {
        "CNN_3D_fold0": 0.7103, "CNN_3D_fold1": 0.7143, "CNN_3D_fold2": 0.6985,
        "CNN_3D_fold3": 0.7011, "CNN_3D_fold4": 0.6980,
        "ResEncL_3D_fold0": 0.7026, "ResEncL_3D_fold1": 0.6931, "ResEncL_3D_fold2": 0.6792,
        "ResEncL_3D_fold3": 0.7020, "ResEncL_3D_fold4": 0.6921,
        "25D_fold0": 0.6829, "25D_fold1": 0.6811, "25D_fold2": 0.6923,
        "25D_fold3": 0.6857, "25D_fold4": 0.6921,
    }
    test_ds002 = {
        "CNN_3D_fold0": 0.7848, "CNN_3D_fold1": 0.7913, "CNN_3D_fold2": 0.7534,
        "CNN_3D_fold3": 0.7792, "CNN_3D_fold4": 0.7810,
        "ResEncL_3D_fold0": 0.7923, "ResEncL_3D_fold1": 0.7818, "ResEncL_3D_fold2": 0.7843,
        "ResEncL_3D_fold3": 0.7965, "ResEncL_3D_fold4": 0.7824,
        "25D_fold0": 0.7889, "25D_fold1": 0.7894, "25D_fold2": 0.7873,
        "25D_fold3": 0.7836, "25D_fold4": 0.7872,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, test_data, title in [(axes[0], test_ds001, "DS001 — MSLesSeg"),
                                  (axes[1], test_ds002, "DS002 — WMH")]:
        for name in val_ema:
            x = val_ema[name]
            y = test_data[name]
            if name.startswith("CNN"):
                c, m, label = C_CNN, "o", "CNN-3D"
            elif name.startswith("ResEncL"):
                c, m, label = C_RE3D, "s", "ResEncL-3D"
            else:
                c, m, label = C_25D, "^", "ResEncL-2.5D"
            ax.scatter(x, y, c=c, marker=m, s=60, alpha=0.8, zorder=3)

        # Add legend (deduplicated)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CNN, markersize=8, label="CNN-3D"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=C_RE3D, markersize=8, label="ResEncL-3D"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_25D, markersize=8, label="ResEncL-2.5D"),
        ]
        ax.legend(handles=handles, fontsize=8)

        # Trend line
        x_all = np.array([val_ema[k] for k in val_ema])
        y_all = np.array([test_data[k] for k in val_ema])
        z = np.polyfit(x_all, y_all, 1)
        p = np.poly1d(z)
        xs = np.linspace(x_all.min() - 0.005, x_all.max() + 0.005, 50)
        ax.plot(xs, p(xs), "--", color="gray", alpha=0.5, lw=1)

        # Correlation
        r = np.corrcoef(x_all, y_all)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Validation EMA fg_dice")
        ax.set_ylabel("Test DSC")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Validation EMA vs. Test DSC — Per-Model Correlation", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "ch7_fig11_val_vs_test_scatter")


# =============================================================================
if __name__ == "__main__":
    print("Generating ensemble selection figures...")
    print("\n[Fig 9] Validation selection strategies")
    fig9_val_selection_strategies()
    print("\n[Fig 10] Master ensemble comparison")
    fig10_ensemble_comparison()
    print("\n[Fig 11] Val EMA vs Test DSC scatter")
    fig11_val_vs_test_scatter()
    print("\nDone!")
