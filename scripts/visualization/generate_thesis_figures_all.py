#!/usr/bin/env python3
"""
generate_thesis_figures_all.py
=====================================
Comprehensive thesis visualization — Chapters 4, 5, 7.

Run from the project root:
    python scripts/visualization/generate_thesis_figures_all.py

Optional flags (positional):
    --skip-nifti    Skip figures that require nibabel / NIfTI I/O
    --skip-wandb    Skip figures that require W&B API
    --only <tag>    Generate one figure only  e.g. --only ch4_fig1

Outputs go to:  results/figures/  (PDF by default).
"""

import os, sys, re, warnings
import argparse
from pathlib import Path

# ensure UTF-8 output even when wandb/Windows hijacks stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")

# ── optional heavy deps ───────────────────────────────────────────────────────
try:
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False
    print("[warn] nibabel not installed — NIfTI figures will be skipped")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[warn] wandb not installed — training-curve figures will be skipped")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy not installed — lesion-detection figure will be skipped")

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE  = REPO_ROOT
DATA  = BASE / "data" / "nnUNet_raw"
PREDS = BASE / "results" / "predictions"
EVAL  = BASE / "results" / "evaluation"
ANAL  = BASE / "results" / "analysis"
OUT   = REPO_ROOT / "results" / "figures"
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
    "pdf.fonttype":    42,  # embed fonts
    "ps.fonttype":     42,
})
SAVE_FMT = "pdf"

COLORS = {
    "CNN-3D":        "#1f77b4",
    "ResEncL-3D":    "#ff7f0e",
    "ResEncL-2.5D":  "#2ca02c",
    "TopK":          "#9467bd",
    "Ensemble":      "#d62728",
    "LST-AI":        "#8c564b",
    "5-fold Bin":    "#e377c2",
    "5-fold Multi":  "#7f7f7f",
    "Binary labels":    "#e377c2",
    "Multi-size labels": "#7f7f7f",
    "DS001":         "#1f77b4",
    "DS002":         "#ff7f0e",
}

# ── W&B (only used for TopK which was trained remotely) ──────────────────────
WANDB_ENTITY  = "broozey-vsb-technical-university-of-ostrava"
WANDB_PROJECT = "ms-lesion-seg"
WANDB_RUNS = {
    "TopK":         "fh1kqtb9",   # DS003 TopK fold0, 297 ep (remote, no local log)
}

# ── local nnUNet training logs (reliable, full, cover split runs) ─────────────
RESULTS = BASE / "data" / "nnUNet_results" / "Dataset003_Combined"
TRAINING_LOG_DIRS = {
    "CNN-3D":      RESULTS / "nnUNetTrainer_WandB__nnUNetPlans__3d_fullres" / "fold_0",
    "ResEncL-3D":  RESULTS / "nnUNetTrainer_WandB__nnUNetResEncUNetLPlans__3d_fullres" / "fold_0",
    "ResEncL-2.5D": RESULTS / "nnUNetTrainer_25D__nnUNetPlans__3d_fullres" / "fold_0",
}

# ── lesion size bins ───────────────────────────────────────────────────────────
SIZE_EDGES = [0, 10, 100, 1000, np.inf]
SIZE_LABELS = ["Tiny\n(<10)", "Small\n(10–100)", "Medium\n(100–1k)", "Large\n(>1k)"]
SIZE_LABELS_PLAIN = ["Tiny\n<10 mm³", "Small\n10–100 mm³",
                     "Medium\n100–1000 mm³", "Large\n>1000 mm³"]

# ─────────────────── UTILITIES ────────────────────────────────────────────────

def savefig(fig, name):
    path = OUT / f"{name}.{SAVE_FMT}"
    fig.savefig(path, format=SAVE_FMT, bbox_inches="tight")
    print(f"  Saved -> {path.name}")
    plt.close(fig)


def load_nii(p, canonical=False):
    p = Path(p)
    if not p.exists():
        return None, None
    img = nib.load(str(p))
    if canonical:
        img = nib.as_closest_canonical(img)
    return img.get_fdata(), img.affine


def clip_norm(vol, lo=1, hi=99):
    """Robust min-max normalisation for display."""
    pos = vol[vol > 0]
    if pos.size == 0:
        return np.zeros_like(vol, dtype=float)
    a, b = np.percentile(pos, [lo, hi])
    b = max(b, a + 1e-5)
    return np.clip((vol - a) / (b - a), 0, 1)


def best_axial_slice(label_vol):
    """Index of the axial (axis=2) slice with the most lesion voxels."""
    sums = (label_vol > 0).sum(axis=(0, 1))
    return int(np.argmax(sums))


def dice_coef(pred, gt):
    p = (pred > 0).astype(np.float32)
    g = (gt   > 0).astype(np.float32)
    num = 2 * (p * g).sum()
    den = p.sum() + g.sum()
    return float(num / den) if den > 0 else 0.0


def load_wandb_history(run_id, keys=("train_loss", "val_loss", "ema_fg_dice")):
    if not HAS_WANDB:
        return None
    try:
        api = wandb.Api(timeout=45)
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        df  = run.history(keys=list(keys), pandas=True)
        return df
    except Exception as exc:
        print(f"    [W&B] could not load {run_id}: {exc}")
        return None


def parse_training_logs(log_dir):
    """Parse and stitch nnUNet training_log_*.txt files from a fold directory.

    Returns dict with keys: epoch, train_loss, val_loss, pseudo_dice.
    For split runs, later files overwrite earlier data at the same epoch.
    """
    import glob as _glob
    log_files = sorted(_glob.glob(str(Path(log_dir) / "training_log_*.txt")))
    data = {}  # epoch -> {train_loss, val_loss, pseudo_dice}
    for lf in log_files:
        with open(lf, "r") as f:
            lines = f.readlines()
        cur_ep = None
        for line in lines:
            m = re.match(r".*Epoch (\d+)\s*$", line.strip())
            if m:
                cur_ep = int(m.group(1))
                continue
            m = re.search(r"train_loss ([\-\d.]+)", line)
            if m and cur_ep is not None:
                data.setdefault(cur_ep, {})["train_loss"] = float(m.group(1))
            m = re.search(r"val_loss ([\-\d.]+)", line)
            if m and cur_ep is not None:
                data.setdefault(cur_ep, {})["val_loss"] = float(m.group(1))
            # Match both: Pseudo dice [np.float32(0.79)] and Pseudo dice [0.79]
            m = re.search(r"Pseudo dice \[(?:np\.float\d*\()?(\d+\.\d+)", line)
            if m and cur_ep is not None:
                data.setdefault(cur_ep, {})["pseudo_dice"] = float(m.group(1))
    if not data:
        return None
    epochs = sorted(data.keys())
    result = {
        "epoch":       np.array(epochs),
        "train_loss":  np.array([data[e].get("train_loss", np.nan) for e in epochs]),
        "val_loss":    np.array([data[e].get("val_loss",   np.nan) for e in epochs]),
        "pseudo_dice": np.array([data[e].get("pseudo_dice", np.nan) for e in epochs]),
    }
    return result


# Short display names for model identifiers used in CSV files
MODEL_SHORT = {
    # DS001
    "DS001_ResEncL_3D":                  "ResEncL\n(ft)",
    "DS001_ResEncL_3D_TTA":              "ResEncL\n(ft+TTA)",
    "DS001_DS002_ResEncL_3D_TTA":        "ResEncL\n(DS002 ft)",
    "DS001_DS003_CNN_3D_TTA":            "CNN-3D",
    "DS001_DS003_ResEncL_3D_TTA":        "ResEncL-3D",
    "DS001_DS003_ResEncL_25D_TTA_chfix": "2.5D",
    "DS001_DS003_CNN_25D_TTA_chfix":     "CNN-2.5D",
    "DS003_TopK_ResEncL_fold0_TTA":      "TopK",
    "DS007_5fold_TTA_binary":            "5-fold\n(bin,TTA)",
    "DS007_5fold_binary":                "5-fold\n(bin)",
    "DS001_DS003_final":                 "Ensemble\n(best)",
    "DS001_final":                       "Ensemble\n(final)",
    # DS002
    "DS002_ResEncL_3D":                  "ResEncL\n(ft)",
    "DS002_ResEncL_3D_TTA":              "ResEncL\n(ft+TTA)",
    "DS002_DS002_ResEncL_3D_TTA":        "ResEncL\n(DS002 ft)",
    "DS002_DS003_CNN_3D_TTA":            "CNN-3D",
    "DS002_DS003_ResEncL_3D_TTA":        "ResEncL-3D",
    "DS002_DS003_ResEncL_25D_TTA_chfix": "2.5D",
    "DS002_DS003_CNN_25D_TTA_chfix":     "CNN-2.5D",
    "DS002_final":                       "Ensemble\n(best)",
}


def short_model_name(model_name):
    """Compact label for fold-specific model identifiers used in ensemble plots."""
    explicit = MODEL_SHORT.get(model_name)
    if explicit is not None:
        return explicit.replace("\n", " ")

    patterns = [
        (r"^CNN_3D_fold(\d+)_TTA(?:_DS002)?$", "CNN_f{}"),
        (r"^ResEncL_3D_fold(\d+)_TTA(?:_DS002)?$", "RE3D_f{}"),
        (r"^25D_fold(\d+)_TTA(?:_DS002)?$", "25D_f{}"),
    ]
    for pattern, label_fmt in patterns:
        match = re.match(pattern, model_name)
        if match:
            return label_fmt.format(match.group(1))

    return model_name.replace("_", " ")[:14]


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 — Dataset Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def ch4_fig1_dataset_examples():
    """2×3 mosaic: FLAIR | T1 | FLAIR+GT overlay — one row per dataset."""
    tag = "ch4_fig1_dataset_examples"
    if not HAS_NIB:
        print(f"  [skip] {tag} — nibabel unavailable")
        return
    print(f"  {tag} …")

    configs = [
        dict(
            row_label="MSLesSeg 2024\n(isotropic ~1 mm³, skull-stripped)",
            flair=DATA / "Dataset001_MSLesSeg/imagesTs/MSL_0097_0000.nii.gz",
            t1=   DATA / "Dataset001_MSLesSeg/imagesTs/MSL_0097_0001.nii.gz",
            lbl=  DATA / "Dataset001_MSLesSeg/labelsTs/MSL_0097.nii.gz",
        ),
        dict(
            row_label="WMH 2017\n(anisotropic ~0.96×0.96×3 mm³, with skull)",
            flair=DATA / "Dataset002_WMH/imagesTs/WMH_0061_0000.nii.gz",
            t1=   DATA / "Dataset002_WMH/imagesTs/WMH_0061_0001.nii.gz",
            lbl=  DATA / "Dataset002_WMH/labelsTs/WMH_0061.nii.gz",
        ),
    ]

    overlay_cmap = ListedColormap(["none", "#ff4400"])
    col_titles   = ["FLAIR", "T1w", "FLAIR + GT lesion mask"]

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5))

    for row, cfg in enumerate(configs):
        flair_v, _ = load_nii(cfg["flair"], canonical=True)
        t1_v,    _ = load_nii(cfg["t1"], canonical=True)
        lbl_v,   _ = load_nii(cfg["lbl"], canonical=True)

        if flair_v is None or lbl_v is None:
            print(f"    [warn] missing data for row {row}")
            for col in range(3):
                axes[row, col].axis("off")
            continue

        flair_n = clip_norm(flair_v)
        t1_n    = clip_norm(t1_v) if t1_v is not None else np.zeros_like(flair_n)
        lbl_b   = (lbl_v > 0).astype(np.float32)

        sl = best_axial_slice(lbl_b)
        imgs = [flair_n[:, :, sl], t1_n[:, :, sl], flair_n[:, :, sl]]

        for col, (img, ctitle) in enumerate(zip(imgs, col_titles)):
            ax = axes[row, col]
            ax.imshow(np.rot90(img), cmap="gray", interpolation="nearest")
            if col == 2:
                ax.imshow(np.rot90(lbl_b[:, :, sl]),
                          cmap=overlay_cmap, alpha=0.55, interpolation="nearest")
            if row == 0:
                ax.set_title(ctitle, fontsize=10, pad=4)
            ax.axis("off")

    # Row labels as text (ylabel doesn't render when axis is off)
    for row, cfg in enumerate(configs):
        ax = axes[row, 0]
        ax.text(-0.05, 0.5, cfg["row_label"], fontsize=8.5,
                ha="right", va="center", transform=ax.transAxes)
    fig.subplots_adjust(left=0.18)
    savefig(fig, tag)


def ch4_fig2_lesion_volume_violin():
    """Violin + strip: per-subject total lesion volume for DS001 and DS002."""
    tag = "ch4_fig2_lesion_volume_violin"
    print(f"  {tag} …")
    df = pd.read_csv(EVAL / "lesion_extents.csv")
    df["dataset"] = np.where(df["subject"].str.startswith("MSL_"),
                             "MSLesSeg\n(DS001)", "WMH 2017\n(DS002)")
    vol = (df.groupby(["dataset", "subject"], sort=False)["vol_mm3"]
             .sum()
             .reset_index(name="vol_ml"))
    vol["vol_ml"] /= 1000.0

    groups    = [vol[vol["dataset"] == k]["vol_ml"].values
                 for k in ["MSLesSeg\n(DS001)", "WMH 2017\n(DS002)"]]
    positions = [1, 2]
    xlabels   = ["MSLesSeg\n(DS001)", "WMH 2017\n(DS002)"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    vp = ax.violinplot(groups, positions=positions, showmedians=True, widths=0.65)

    pal = [COLORS["DS001"], COLORS["DS002"]]
    for body, c in zip(vp["bodies"], pal):
        body.set_facecolor(c)
        body.set_alpha(0.65)
    for part in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part in vp:
            vp[part].set_color("black")
            vp[part].set_linewidth(1.3)

    # Overlay strip (jitter)
    rng = np.random.default_rng(0)
    for i, (g, c) in enumerate(zip(groups, pal), 1):
        jitter = rng.uniform(-0.08, 0.08, size=len(g))
        ax.scatter(i + jitter, g, s=14, color=c, alpha=0.55,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Total lesion volume per subject (mL)")
    ax.set_title("Lesion Volume Distribution by Dataset")
    ax.set_xlim(0.3, 2.7)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    savefig(fig, tag)


def ch4_fig3_lesion_size_histogram():
    """Grouped bar: lesion count by size category, DS001 vs DS002."""
    tag = "ch4_fig3_lesion_size_histogram"
    print(f"  {tag} …")
    df = pd.read_csv(EVAL / "lesion_extents.csv")
    df["dataset"] = np.where(df["subject"].str.startswith("MSL_"),
                             "MSLesSeg (DS001)", "WMH 2017 (DS002)")
    df["size_bin"] = pd.cut(df["vol_mm3"],
                            bins=SIZE_EDGES,
                            labels=SIZE_LABELS_PLAIN,
                            right=False)

    counts = (df.groupby(["dataset", "size_bin"], observed=True)
                .size()
                .unstack("size_bin")
                .reindex(columns=SIZE_LABELS_PLAIN, fill_value=0))

    x = np.arange(len(SIZE_LABELS_PLAIN))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, (ds, color) in enumerate(zip(counts.index,
                                        [COLORS["DS001"], COLORS["DS002"]])):
        vals = counts.loc[ds].values.astype(float)
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w,
                      label=ds, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 15,
                        f"{int(v):,}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_LABELS_PLAIN, fontsize=9)
    ax.set_xlabel("Lesion size category (mm³)")
    ax.set_ylabel("Number of lesions")
    ax.set_title("Lesion Size Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    savefig(fig, tag)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 — Training & Architecture Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def ch5_fig1_training_curves():
    """Loss + Pseudo Dice curves for CNN-3D fold 0 (representative run, from local logs)."""
    tag = "ch5_fig1_training_curves"
    print(f"  {tag} …")

    log = parse_training_logs(TRAINING_LOG_DIRS["CNN-3D"])
    if log is None:
        print(f"    [skip] {tag} — no local logs found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Early stopping epoch for CNN-3D fold 0
    es_epoch = 243

    # — loss panel
    ax1.plot(log["epoch"], log["train_loss"],
             color="#1f77b4", linewidth=1.0, alpha=0.85, label="Train")
    ax1.plot(log["epoch"], log["val_loss"],
             color="#ff7f0e", linewidth=1.0, label="Validation")
    ax1.axvline(es_epoch, color="#d62728", linestyle="--", linewidth=0.9,
                alpha=0.7, label=f"Early stop (ep {es_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Dice + CE)")
    ax1.set_title("CNN-3D, DS003, fold 0 — Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, linestyle="--")

    # — Pseudo Dice panel
    ax2.plot(log["epoch"], log["pseudo_dice"],
             color=COLORS["CNN-3D"], linewidth=1.0)
    ax2.axvline(es_epoch, color="#d62728", linestyle="--", linewidth=0.9,
                alpha=0.7, label=f"Early stop (ep {es_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pseudo Dice")
    ax2.set_title("CNN-3D, DS003, fold 0 — Pseudo Dice")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, linestyle="--")

    savefig(fig, tag)


def ch5_fig2_convergence_comparison():
    """Pseudo Dice convergence for CNN-3D, ResEncL-3D, ResEncL-2.5D from local logs."""
    tag = "ch5_fig2_convergence_comparison"
    print(f"  {tag} …")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for arch in ["CNN-3D", "ResEncL-3D", "ResEncL-2.5D"]:
        log = parse_training_logs(TRAINING_LOG_DIRS[arch])
        if log is None:
            print(f"    [warn] no local logs for {arch}")
            continue
        ax.plot(log["epoch"], log["pseudo_dice"], color=COLORS[arch],
                label=arch, linewidth=1.2, alpha=0.9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pseudo Dice")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 1.0)
    savefig(fig, tag)


def ch5_fig3_topk_vs_dicece():
    """Pseudo Dice: TopK loss (W&B) vs standard Dice+CE loss (local log) on ResEncL."""
    tag = "ch5_fig3_topk_vs_dicece"
    print(f"  {tag} …")

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Dice+CE: ResEncL-3D fold0 from local logs
    log = parse_training_logs(TRAINING_LOG_DIRS["ResEncL-3D"])
    if log is not None:
        ax.plot(log["epoch"], log["pseudo_dice"],
                color="#1f77b4", label="ResEncL (Dice+CE)", linewidth=1.2, alpha=0.9)

    # TopK: from W&B (trained remotely, no local log)
    if HAS_WANDB:
        df = load_wandb_history(WANDB_RUNS["TopK"], keys=("ema_fg_dice", "epoch"))
        if df is not None and not df.empty:
            ed = df.dropna(subset=["ema_fg_dice"])
            x  = ed["epoch"] if "epoch" in ed.columns else ed["_step"]
            ax.plot(x, ed["ema_fg_dice"], color=COLORS["TopK"],
                    label="ResEncL (TopK)", linewidth=1.2, alpha=0.9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice (Pseudo / EMA)")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 1.0)
    savefig(fig, tag)


def ch5_fig4_multisize_vs_binary():
    """Box plot: 5-fold binary vs multi-size labels on DS001 test set."""
    tag = "ch5_fig4_multisize_vs_binary"
    if not HAS_NIB:
        print(f"  [skip] {tag} — nibabel unavailable")
        return
    print(f"  {tag} …")

    lbl_dir = DATA / "Dataset001_MSLesSeg" / "labelsTs"
    results  = {}
    pairs    = [
        ("Binary labels",     PREDS / "DS007_5fold_TTA_binary"),
        ("Multi-size labels", PREDS / "DS007_5fold_TTA_multisize"),
    ]

    for cfg_label, fold_dir in pairs:
        dices = []
        if not fold_dir.exists():
            print(f"    [warn] {fold_dir.name} not found")
            continue
        for pred_f in sorted(fold_dir.glob("*.nii.gz")):
            gt_f = lbl_dir / pred_f.name
            if not gt_f.exists():
                continue
            pred_v, _ = load_nii(pred_f)
            gt_v,   _ = load_nii(gt_f)
            if pred_v is None or gt_v is None:
                continue
            dices.append(dice_coef(pred_v, gt_v))
        results[cfg_label] = dices

    if not results:
        print(f"    [skip] {tag} — no predictions found")
        return

    keys   = list(results.keys())
    values = [results[k] for k in keys]
    fig, ax = plt.subplots(figsize=(4, 4.5))
    bp = ax.boxplot(values, labels=keys, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 0.9},
                    capprops={"linewidth": 0.9})
    pal = [COLORS.get("Binary labels", "#e377c2"), COLORS.get("Multi-size labels", "#7f7f7f")]
    for patch, c in zip(bp["boxes"], pal):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    # Annotate mean DSC on each box
    for i, (k, v) in enumerate(zip(keys, values)):
        if v:
            mean_val = np.mean(v)
            med_val = np.median(v)
            ax.text(i + 1, med_val + 0.02, f"median={med_val:.3f}\nmean={mean_val:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("Dice Score (DSC)")
    ax.set_title("Multi-size vs Binary Labels\nDS001 Test Set (5-fold)")
    ax.set_ylim(0.5, 0.85)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    savefig(fig, tag)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 7 — Results & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def ch7_fig1_pipeline_progression():
    """Grouped bar: mean Dice per pipeline configuration for DS001 and DS002."""
    tag = "ch7_fig1_pipeline_progression"
    print(f"  {tag} …")

    ds1 = pd.read_csv(ANAL / "deep_comparison_MSLesSeg22.csv")
    ds2 = pd.read_csv(ANAL / "deep_comparison_WMH110.csv")

    PIPELINE_DS001 = [
        ("DS001_ResEncL_3D",                 "Baseline (fold 0)"),
        ("DS001_ResEncL_3D_TTA",             "Baseline + TTA"),
        ("DS001_DS003_CNN_3D_TTA",           "CNN-3D"),
        ("DS001_DS003_ResEncL_3D_TTA",       "ResEncL-3D"),
        ("DS001_DS003_ResEncL_25D_TTA_chfix","ResEncL-2.5D"),
        ("DS001_DS003_final",                "3-arch Ensemble"),
    ]
    # Best-2/arch ensemble values (manually computed, validation-selected)
    BEST2ARCH_DS001 = 0.7179
    BEST2ARCH_DS002 = 0.8032

    PIPELINE_DS002 = [
        ("DS002_ResEncL_3D",                 "Baseline (fold 0)"),
        ("DS002_ResEncL_3D_TTA",             "Baseline + TTA"),
        ("DS002_DS003_CNN_3D_TTA",           "CNN-3D"),
        ("DS002_DS003_ResEncL_3D_TTA",       "ResEncL-3D"),
        ("DS002_DS003_ResEncL_25D_TTA_chfix","ResEncL-2.5D"),
        ("DS002_final",                      "3-arch Ensemble"),
    ]

    def mean_dice(df, model):
        s = df[df["model"] == model]["dice"]
        return float(s.mean()) if len(s) else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.6))

    for ax, pipeline, df, title, color, ylo, yhi, best2arch_val in [
        (ax1, PIPELINE_DS001, ds1,
         "MSLesSeg 2024 (DS001)", COLORS["DS001"], 0.62, 0.76, BEST2ARCH_DS001),
        (ax2, PIPELINE_DS002, ds2,
         "WMH 2017 (DS002)",      COLORS["DS002"], 0.74, 0.82, BEST2ARCH_DS002),
    ]:
        models, xlabels = zip(*pipeline)
        vals = [mean_dice(df, m) for m in models]
        # Append Best-2/arch bar
        xlabels = list(xlabels) + ["Best-2/arch"]
        vals = list(vals) + [best2arch_val]
        x    = np.arange(len(xlabels))
        bar_colors = [color] * (len(xlabels) - 1) + ["#d62728"]  # red for final
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.82, edgecolor="white",
                      width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8, rotation=20, ha="right", rotation_mode="anchor")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.margins(x=0.04)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.0015, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)

    ax1.set_ylabel("Mean Dice (DSC)")
    fig.tight_layout(pad=1.0, w_pad=1.6)
    savefig(fig, tag)


def ch7_fig2_lesion_detection_by_size():
    """Sensitivity per lesion-size bin for best models on DS001."""
    tag = "ch7_fig2_lesion_detection_by_size"
    if not (HAS_NIB and HAS_SCIPY):
        print(f"  [skip] {tag} — nibabel/scipy unavailable")
        return
    print(f"  {tag} …  (computing connected components, may take ~30 s)")

    lbl_dir = DATA / "Dataset001_MSLesSeg" / "labelsTs"
    models  = {
        "CNN-3D":       PREDS / "DS001_DS003_CNN_3D_TTA",
        "ResEncL-3D":   PREDS / "DS001_DS003_ResEncL_3D_TTA",
        "ResEncL-2.5D": PREDS / "DS001_DS003_ResEncL_25D_TTA_chfix",
        "Ensemble":     PREDS / "DS001_DS003_final",
    }

    # Each element: (size_bin_idx, detected_bool)
    detection = {k: [[] for _ in SIZE_EDGES[:-1]] for k in models}

    for lbl_f in sorted(lbl_dir.glob("*.nii.gz")):
        gt_v,  aff = load_nii(lbl_f)
        if gt_v is None:
            continue
        # voxel volume in mm³
        vox_vol = float(np.abs(np.linalg.det(aff[:3, :3])))

        gt_bin   = (gt_v > 0).astype(np.int32)
        struct   = ndimage.generate_binary_structure(3, 1)
        labeled, n_cc = ndimage.label(gt_bin, structure=struct)

        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            vol_mm3 = float(cc_mask.sum()) * vox_vol
            bin_idx = int(np.searchsorted(SIZE_EDGES[1:-1], vol_mm3))

            for model_name, pred_dir in models.items():
                pred_f = pred_dir / lbl_f.name
                if not pred_f.exists():
                    continue
                pred_v, _ = load_nii(pred_f)
                if pred_v is None:
                    continue
                detected = bool(((pred_v > 0) & cc_mask).any())
                detection[model_name][bin_idx].append(detected)

    # Compute sensitivity per bin per model
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x   = np.arange(len(SIZE_LABELS_PLAIN))
    n_m = len(models)
    w   = 0.7 / n_m

    for i, (mname, det_bins) in enumerate(detection.items()):
        sens = [np.mean(b) if b else np.nan for b in det_bins]
        offset = (i - (n_m - 1) / 2) * w
        bars = ax.bar(x + offset, sens, width=w,
                      label=mname, color=COLORS.get(mname, f"C{i}"),
                      alpha=0.82, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_LABELS_PLAIN, fontsize=9)
    ax.set_xlabel("Lesion size (mm³)")
    ax.set_ylabel("Lesion-level Sensitivity")
    ax.set_title("Lesion Detection Rate by Size — DS001")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    savefig(fig, tag)


def ch7_fig3_subject_scatter():
    """Per-subject scatter: DSC vs GT lesion volume — DS001 and DS002."""
    tag = "ch7_fig3_subject_scatter"
    print(f"  {tag} …")

    ds1 = pd.read_csv(ANAL / "deep_comparison_MSLesSeg22.csv")
    ds2 = pd.read_csv(ANAL / "deep_comparison_WMH110.csv")

    # gt_vol in the CSV is in voxels. Convert to mL:
    # DS001: 1mm isotropic -> 1 voxel = 1 mm3 = 0.001 mL
    # DS002: ~0.96x0.96x3 mm -> ~2.77 mm3/voxel, but we just use mL label
    model_sel = [
        ("DS001_DS003_final",          ds1, "#1f77b4", "MSLesSeg (DS001)",      0.001),
        ("DS002_DS003_ResEncL_3D_TTA", ds2, "#ff7f0e", "WMH 2017 (DS002)",     0.00277),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    for ax, (model, df, color, title, vox_ml) in zip(axes, model_sel):
        sub = df[df["model"] == model].copy()
        sub["gt_vol_ml"] = sub["gt_vol"] * vox_ml

        valid = sub.dropna(subset=["gt_vol_ml", "dice"])
        ax.scatter(valid["gt_vol_ml"], valid["dice"],
                   s=32, color=color, alpha=0.72,
                   edgecolors="white", linewidths=0.4)

        if len(valid) >= 2:
            z  = np.polyfit(valid["gt_vol_ml"], valid["dice"], 1)
            px = np.linspace(valid["gt_vol_ml"].min(),
                             valid["gt_vol_ml"].max(), 150)
            ax.plot(px, np.polyval(z, px), "--", color="#444444",
                    linewidth=1.1, alpha=0.75, label="Linear fit")
            r  = np.corrcoef(valid["gt_vol_ml"], valid["dice"])[0, 1]
            ax.text(0.97, 0.04, f"r = {r:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

        ax.set_xlabel("GT lesion volume (mL)")
        ax.set_ylabel("Dice" if ax is axes[0] else "")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle="--")

    savefig(fig, tag)


def ch7_fig4_qualitative_grid():
    """3-row qualitative grid: bad / average / good segmentation on DS001."""
    tag = "ch7_fig4_qualitative_grid"
    if not HAS_NIB:
        print(f"  [skip] {tag} — nibabel unavailable")
        return
    print(f"  {tag} …")

    img_dir  = DATA  / "Dataset001_MSLesSeg" / "imagesTs"
    lbl_dir  = DATA  / "Dataset001_MSLesSeg" / "labelsTs"
    pred_dir = PREDS / "DS001_DS003_final"

    cases = [
        ("Bad\n(DSC ≈ 0.44)",     "MSL_0099"),
        ("Average\n(DSC ≈ 0.69)", "MSL_0101"),
        ("Good\n(DSC ≈ 0.83)",    "MSL_0109"),
    ]

    gt_cmap   = ListedColormap(["none", "#00cc44"])   # green — GT
    pred_cmap = ListedColormap(["none", "#ff3300"])   # red   — pred

    col_titles = ["FLAIR", "GT (green) + Prediction (red)", "Prediction only"]

    fig, axes = plt.subplots(len(cases), 3,
                             figsize=(8.5, len(cases) * 2.6))

    for row, (quality, case) in enumerate(cases):
        flair_v, _ = load_nii(img_dir  / f"{case}_0000.nii.gz")
        gt_v,    _ = load_nii(lbl_dir  / f"{case}.nii.gz")
        pred_f     = pred_dir / f"{case}.nii.gz"
        pred_v, _  = load_nii(pred_f) if pred_f.exists() else (None, None)

        if flair_v is None or gt_v is None:
            print(f"    [warn] missing data for {case}")
            for col in range(3):
                axes[row, col].axis("off")
            continue

        flair_n  = clip_norm(flair_v)
        gt_b     = (gt_v > 0).astype(np.float32)
        pred_b   = ((pred_v > 0).astype(np.float32)
                    if pred_v is not None else np.zeros_like(gt_b))

        sl    = best_axial_slice(gt_b)
        fl_s  = np.rot90(flair_n[:, :, sl])
        gt_s  = np.rot90(gt_b[:,   :, sl])
        pr_s  = np.rot90(pred_b[:,  :, sl])

        axes[row, 0].imshow(fl_s, cmap="gray", interpolation="nearest")

        axes[row, 1].imshow(fl_s,  cmap="gray", interpolation="nearest")
        axes[row, 1].imshow(gt_s,  cmap=gt_cmap,   alpha=0.55, interpolation="nearest")
        axes[row, 1].imshow(pr_s,  cmap=pred_cmap,  alpha=0.45, interpolation="nearest")

        axes[row, 2].imshow(fl_s,  cmap="gray", interpolation="nearest")
        axes[row, 2].imshow(pr_s,  cmap=pred_cmap,  alpha=0.55, interpolation="nearest")

        for col in range(3):
            axes[row, col].axis("off")

    # Row labels via text (ylabel is invisible with axis off)
    for row, (quality, _) in enumerate(cases):
        ax = axes[row, 0]
        ax.text(-0.05, 0.5, quality, fontsize=9,
                ha="right", va="center", transform=ax.transAxes)

    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontsize=9.5, pad=4)
    fig.subplots_adjust(left=0.15)
    savefig(fig, tag)


def ch7_fig5_postprocessing():
    """Side-by-side bar: voxel Dice and lesion F1 per post-processing strategy."""
    tag = "ch7_fig5_postprocessing"
    print(f"  {tag} …")

    df = pd.read_csv(EVAL / "postprocessing" / "full_postprocessing_comparison.csv")

    # Shorten strategy strings for display
    _STRAT_SHORT = {
        "TTA_thr_0.30":         "thr=0.30",
        "TTA_thr_0.35":         "thr=0.35",
        "TTA_thr_0.40":         "thr=0.40",
        "TTA_thr_0.45":         "thr=0.45",
        "TTA_thr_0.30_rm_3mm3": "thr=0.30\n+rm 3mm³",
        "TTA_thr_0.30_rm_5mm3": "thr=0.30\n+rm 5mm³",
        "TTA_thr_0.30_rm_10mm3":"thr=0.30\n+rm 10mm³",
        "TTA_thr_0.35_rm_3mm3": "thr=0.35\n+rm 3mm³",
        "TTA_thr_0.35_rm_5mm3": "thr=0.35\n+rm 5mm³",
        "TTA_thr_0.40_rm_3mm3": "thr=0.40\n+rm 3mm³",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.4), sharey=False)

    for ax, ds_tag, title, color in [
        (axes[0], "DS001", "MSLesSeg 2024 (DS001)", COLORS["DS001"]),
        (axes[1], "DS002", "WMH 2017 (DS002)",      COLORS["DS002"]),
    ]:
        sub  = df[df["dataset"] == ds_tag].copy()
        sub["short"] = sub["strategy"].map(_STRAT_SHORT).fillna(sub["strategy"])
        sub  = sub.sort_values("dice_mean", ascending=False)

        x = np.arange(len(sub))
        w = 0.38
        ax.bar(x - w / 2, sub["dice_mean"], w,
               label="Voxel Dice", color=color,       alpha=0.82, edgecolor="white")
        ax.bar(x + w / 2, sub["l_f1"],     w,
               label="Lesion F1",  color="#d62728",   alpha=0.72, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["short"].values, fontsize=6.5, rotation=35,
                           ha="right", rotation_mode="anchor")
        ax.set_title(title)
        ax.set_ylabel("Score" if ax is axes[0] else "")
        dmin = min(sub["dice_mean"].min(), sub["l_f1"].min()) - 0.02
        dmax = max(sub["dice_mean"].max(), sub["l_f1"].max()) + 0.02
        ax.set_ylim(max(0, dmin), min(1, dmax))
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.legend(fontsize=8)

    fig.tight_layout(pad=1.0, w_pad=1.4)
    savefig(fig, tag)


def ch7_fig6_ensemble_heatmap():
    """Heatmap of best pairwise (size=2) ensemble Dice — DS001 and DS002."""
    tag = "ch7_fig6_ensemble_heatmap"
    print(f"  {tag} …")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))

    for ax, csv_name, title in [
        (axes[0], "DS001_best_per_combo.csv", "MSLesSeg 2024 (DS001)"),
        (axes[1], "DS002_best_per_combo.csv", "WMH 2017 (DS002)"),
    ]:
        df    = pd.read_csv(EVAL / "exhaustive_ensembles" / csv_name)
        pairs = df[df["combo_size"] == 2].copy()

        # Parse "model_a + model_b" → two columns
        split = pairs["combo"].str.split(r"\s*\+\s*", n=1, expand=True)
        pairs = pairs.copy()
        pairs["m0"] = split[0].str.strip()
        pairs["m1"] = split[1].str.strip() if 1 in split.columns else None
        pairs = pairs.dropna(subset=["m0", "m1"])

        # Build short-name index
        all_models = sorted(set(pairs["m0"]) | set(pairs["m1"]))
        short_names = [short_model_name(m) for m in all_models]
        idx = {m: i for i, m in enumerate(all_models)}
        n   = len(all_models)

        mat = np.full((n, n), np.nan)
        for _, row in pairs.iterrows():
            i = idx.get(row["m0"])
            j = idx.get(row["m1"])
            if i is not None and j is not None:
                mat[i, j] = mat[j, i] = row["best_dice"]

        # diagonal = single-model best dice (size=1)
        singles = df[df["combo_size"] == 1].set_index("combo")["best_dice"].to_dict()
        for m, i in idx.items():
            if m in singles:
                mat[i, i] = singles[m]

        vmin = np.nanmin(mat) - 0.002
        vmax = np.nanmax(mat) + 0.002
        im   = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(short_names, fontsize=6)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=5.5, color="black")
        plt.colorbar(im, ax=ax, label="Best Dice", fraction=0.046, pad=0.04)
        ax.set_title(title)

    fig.tight_layout(pad=1.2, w_pad=2.8)
    savefig(fig, tag)


def ch7_fig7_fivefold_boxplot():
    """Box plots of per-fold Dice distributions for CNN-3D and 2.5D — DS001."""
    tag = "ch7_fig7_fivefold_boxplot"
    if not HAS_NIB:
        print(f"  [skip] {tag} — nibabel unavailable")
        return
    print(f"  {tag} …  (loading NIfTIs, may take ~1–2 min)")

    lbl_dir = DATA / "Dataset001_MSLesSeg" / "labelsTs"

    def compute_fold_dices(arch_prefix):
        all_dices = []
        for fold in range(5):
            fold_dir = PREDS / f"{arch_prefix}_fold{fold}_TTA"
            if not fold_dir.exists():
                all_dices.append(None)
                continue
            dices = []
            for lbl_f in sorted(lbl_dir.glob("*.nii.gz")):
                pred_f = fold_dir / lbl_f.name
                if not pred_f.exists():
                    continue
                gt_v,   _ = load_nii(lbl_f)
                pred_v, _ = load_nii(pred_f)
                if gt_v is None or pred_v is None:
                    continue
                dices.append(dice_coef(pred_v, gt_v))
            all_dices.append(dices if dices else None)
        return all_dices

    archs = [("CNN-3D",       "CNN_3D"),
             ("ResEncL-3D",   "ResEncL_3D"),
             ("ResEncL-2.5D", "25D")]

    fig, axes = plt.subplots(1, len(archs), figsize=(12, 4.5), sharey=True)

    for ax, (arch_label, prefix) in zip(axes, archs):
        print(f"    computing {arch_label} per-fold Dice …")
        all_dices  = compute_fold_dices(prefix)
        valid      = [(f"Fold {i}", d) for i, d in enumerate(all_dices) if d]
        if not valid:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        labels, data = zip(*valid)
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5},
                        whiskerprops={"linewidth": 0.9},
                        capprops={"linewidth": 0.9},
                        flierprops={"markersize": 4})
        color = COLORS[arch_label]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_title(arch_label)
        ax.set_ylabel("Dice" if ax is axes[0] else "")
        ax.set_ylim(0.3, 0.9)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        # annotate medians
        for i, d in enumerate(data):
            med = np.median(d)
            ax.text(i + 1, med + 0.02, f"{med:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

    savefig(fig, tag)


def ch7_fig8_fold_diversity():
    """Scatter: fold-0 Dice vs fold-1 Dice per subject (fold agreement check)."""
    tag = "ch7_fig8_fold_diversity"
    if not HAS_NIB:
        print(f"  [skip] {tag} — nibabel unavailable")
        return
    print(f"  {tag} …")

    lbl_dir = DATA / "Dataset001_MSLesSeg" / "labelsTs"

    archs = [("CNN-3D",       "CNN_3D",  COLORS["CNN-3D"]),
             ("ResEncL-2.5D", "25D",     COLORS["ResEncL-2.5D"])]

    fig, axes = plt.subplots(1, len(archs), figsize=(8.5, 4.5))

    for ax, (arch_label, prefix, color) in zip(axes, archs):
        f0_dir = PREDS / f"{prefix}_fold0_TTA"
        f1_dir = PREDS / f"{prefix}_fold1_TTA"
        if not (f0_dir.exists() and f1_dir.exists()):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        f0_d, f1_d = [], []
        for lbl_f in sorted(lbl_dir.glob("*.nii.gz")):
            gt_v, _ = load_nii(lbl_f)
            p0, _   = load_nii(f0_dir / lbl_f.name)
            p1, _   = load_nii(f1_dir / lbl_f.name)
            if gt_v is None or p0 is None or p1 is None:
                continue
            f0_d.append(dice_coef(p0, gt_v))
            f1_d.append(dice_coef(p1, gt_v))

        ax.scatter(f0_d, f1_d, s=40, color=color, alpha=0.75,
                   edgecolors="white", linewidths=0.4)
        lo, hi = 0, 1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.5)
        r = np.corrcoef(f0_d, f1_d)[0, 1] if len(f0_d) > 1 else np.nan
        ax.text(0.96, 0.04, f"r = {r:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.set_xlabel("Fold 0 Dice")
        ax.set_ylabel("Fold 1 Dice" if ax is axes[0] else "")
        ax.set_title(arch_label)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle="--")

    savefig(fig, tag)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_FIGURES = [
    # tag,                      function
    ("ch4_fig1", ch4_fig1_dataset_examples),
    ("ch4_fig2", ch4_fig2_lesion_volume_violin),
    ("ch4_fig3", ch4_fig3_lesion_size_histogram),
    ("ch5_fig1", ch5_fig1_training_curves),
    ("ch5_fig2", ch5_fig2_convergence_comparison),
    ("ch5_fig3", ch5_fig3_topk_vs_dicece),
    ("ch5_fig4", ch5_fig4_multisize_vs_binary),
    ("ch7_fig1", ch7_fig1_pipeline_progression),
    ("ch7_fig2", ch7_fig2_lesion_detection_by_size),
    ("ch7_fig3", ch7_fig3_subject_scatter),
    ("ch7_fig4", ch7_fig4_qualitative_grid),
    ("ch7_fig5", ch7_fig5_postprocessing),
    ("ch7_fig6", ch7_fig6_ensemble_heatmap),
    ("ch7_fig7", ch7_fig7_fivefold_boxplot),
    ("ch7_fig8", ch7_fig8_fold_diversity),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-nifti", action="store_true",
                        help="Skip figures that require NIfTI I/O")
    parser.add_argument("--skip-wandb", action="store_true",
                        help="Skip figures that require W&B API")
    parser.add_argument("--only", metavar="TAG",
                        help="Run only the figure with this prefix (e.g. ch4_fig1)")
    args = parser.parse_args()

    global HAS_NIB, HAS_WANDB
    if args.skip_nifti:
        HAS_NIB = False
    if args.skip_wandb:
        HAS_WANDB = False

    width = 60
    print("=" * width)
    print("  Thesis Figures Generator")
    print(f"  Output: {OUT}")
    print("=" * width)

    for fig_tag, fn in ALL_FIGURES:
        if args.only and not fig_tag.startswith(args.only):
            continue
        try:
            fn()
        except Exception as exc:
            print(f"  [ERROR] {fig_tag}: {exc}")
            import traceback; traceback.print_exc()

    print("=" * width)
    print("  Done.")
    print("=" * width)


if __name__ == "__main__":
    main()
