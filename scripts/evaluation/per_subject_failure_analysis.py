#!/usr/bin/env python3
"""
Per-subject failure mode analysis for DS001 (MSLesSeg) test set.

Computes per-subject DSC for the validation-selected Best-2/arch ensemble
(6 models: CNN_fold{1,3} + RE3D_fold{1,3} + 25D_fold{1,3}), then correlates
segmentation performance with clinical metadata (Age, EDSS, Lesion Volume,
Lesion Number, Sex, MS Type) extracted from the original MSLesSeg dataset.

Outputs:
  - thesis_outputs/figures/ch7_fig12_failure_mode_analysis.pdf
  - results/evaluation/per_subject_ds001_clinical.csv
"""
import json
import sys
import zipfile
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_test_sets import compute_metrics

# ── Paths ────────────────────────────────────────────────────────────────────
MAPPING_JSON = REPO_ROOT / "data/nnUNet_raw/Dataset001_MSLesSeg/mapping.json"
GT_DIR = REPO_ROOT / "data/nnUNet_raw/Dataset001_MSLesSeg/labelsTs"
PRED_DIR = REPO_ROOT / "results/predictions"
CLINICAL_ZIP = REPO_ROOT / "data/original/mslesseg/MSLesSeg Dataset.zip"
CLINICAL_ZIP_PATH = "MSLesSeg Dataset/info_dataset/clinical_data.csv"

OUT_FIG = REPO_ROOT / "thesis_outputs/figures/ch7_fig12_failure_mode_analysis.pdf"
OUT_CSV = REPO_ROOT / "results/evaluation/per_subject_ds001_clinical.csv"

# Best-2/arch models (selected by validation EMA fg_dice)
ENSEMBLE_MODELS = [
    "CNN_3D_fold1",
    "CNN_3D_fold3",
    "ResEncL_3D_fold1",
    "ResEncL_3D_fold3",
    "25D_fold1",
    "25D_fold3",
]
THRESHOLD = 0.50


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_mapping():
    """Load MSL case_id -> Patient mapping for test cases."""
    with open(MAPPING_JSON) as f:
        mapping = json.load(f)
    return {
        entry["case_id"]: entry["patient"]
        for entry in mapping
        if entry["split"] == "test"
    }


def load_clinical_data():
    """Extract and parse clinical_data.csv from the MSLesSeg zip."""
    with zipfile.ZipFile(CLINICAL_ZIP) as zf:
        raw = zf.read(CLINICAL_ZIP_PATH).decode("utf-8-sig")

    # Semicolon-separated, European decimal notation (commas)
    # Header: Patient;Timepoint;Age;Sex;MS Type;EDSS;Lesion Number;Lesion Volume;
    # Note: "Lesion Number" column is empty; actual count is in trailing column
    df = pd.read_csv(StringIO(raw), sep=";", decimal=",")
    # Drop unnamed trailing column from trailing semicolon
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # The "Lesion Number" column is empty; the actual lesion count is lost
    # due to CSV formatting. Lesion Volume is populated correctly.
    return df


def load_prob(npz_path):
    """Load softmax probability for foreground class from .npz file."""
    d = np.load(npz_path)
    key = "softmax" if "softmax" in d.files else "probabilities"
    arr = d[key][1].astype(np.float32)
    return np.transpose(arr, (2, 1, 0))


def compute_per_subject_metrics():
    """Compute DSC and full metrics for each test subject using the ensemble."""
    case_to_patient = load_mapping()
    gt_files = sorted(GT_DIR.glob("*.nii.gz"))
    rows = []

    for gt_path in gt_files:
        case_id = gt_path.name.replace(".nii.gz", "")
        if case_id not in case_to_patient:
            continue

        # Load and average ensemble probabilities
        probs = []
        for model in ENSEMBLE_MODELS:
            pred_dir = PRED_DIR / f"{model}_TTA"
            npz = pred_dir / f"{case_id}.npz"
            if not npz.exists():
                npz = pred_dir / f"{case_id}.nii.gz.npz"
            if not npz.exists():
                print(f"  [WARN] Missing softmax: {npz}")
                continue
            probs.append(load_prob(npz))

        if len(probs) != len(ENSEMBLE_MODELS):
            print(f"  [WARN] Incomplete predictions for {case_id}, skipping")
            continue

        avg_prob = np.mean(probs, axis=0)
        pred = (avg_prob >= THRESHOLD).astype(np.uint8)

        gt_img = nib.load(str(gt_path))
        gt = (gt_img.get_fdata() > 0.5).astype(np.uint8)
        pixdim = list(gt_img.header.get_zooms()[:3])

        m = compute_metrics(pred, gt, pixdim)
        m["case_id"] = case_id
        m["patient"] = case_to_patient[case_id]
        rows.append(m)

    return pd.DataFrame(rows)

def merge_with_clinical(metrics_df):
    """Merge per-subject metrics with clinical metadata."""
    clinical = load_clinical_data()

    # Test patients have a single timepoint (T1 in the CSV)
    # Filter to T1 only (test patients all have 1 timepoint)
    clinical_t1 = clinical[clinical["Timepoint"] == "T1"].copy()

    # Rename for merge
    clinical_t1 = clinical_t1.rename(columns={
        "Patient": "patient",
        "Age": "age",
        "Sex": "sex",
        "MS Type": "ms_type",
        "EDSS": "edss",
        "Lesion Volume": "lesion_volume_mm3",
    })
    cols = ["patient", "age", "sex", "ms_type", "edss", "lesion_volume_mm3"]
    clinical_t1 = clinical_t1[[c for c in cols if c in clinical_t1.columns]]

    # Ensure numeric columns are actually numeric
    for col in ["age", "edss", "lesion_volume_mm3"]:
        if col in clinical_t1.columns:
            clinical_t1[col] = pd.to_numeric(clinical_t1[col], errors="coerce")

    df = pd.merge(metrics_df, clinical_t1, on="patient", how="left")
    return df


def generate_figure(df):
    """Create 4-panel scatter plot: DSC vs Age, EDSS, Lesion Volume, Lesion Count."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Per-Subject Segmentation Performance vs Clinical Variables\n"
        "(Best-2/arch Ensemble, DS001 Test Set, n=22)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # (a) DSC vs Age
    ax = axes[0, 0]
    valid = df.dropna(subset=["age"])
    ax.scatter(valid["age"], valid["dice"], c="#2196F3", s=60, edgecolors="k", linewidths=0.5, zorder=5)
    if len(valid) >= 3:
        rho, pval = stats.spearmanr(valid["age"], valid["dice"])
        z = np.polyfit(valid["age"], valid["dice"], 1)
        x_line = np.linspace(valid["age"].min(), valid["age"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5)
        ax.set_title(f"(a) DSC vs Age  ($\\rho$={rho:.3f}, p={pval:.3f})")
    else:
        ax.set_title("(a) DSC vs Age")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)

    # (b) DSC vs EDSS
    ax = axes[0, 1]
    valid = df.dropna(subset=["edss"])
    ax.scatter(valid["edss"], valid["dice"], c="#FF9800", s=60, edgecolors="k", linewidths=0.5, zorder=5)
    if len(valid) >= 3:
        rho, pval = stats.spearmanr(valid["edss"], valid["dice"])
        z = np.polyfit(valid["edss"], valid["dice"], 1)
        x_line = np.linspace(valid["edss"].min(), valid["edss"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5)
        ax.set_title(f"(b) DSC vs EDSS  ($\\rho$={rho:.3f}, p={pval:.3f})")
    else:
        ax.set_title("(b) DSC vs EDSS")
    ax.set_xlabel("EDSS Score")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)

    # (c) DSC vs Lesion Volume (log scale)
    ax = axes[1, 0]
    valid = df.dropna(subset=["lesion_volume_mm3"])
    valid_pos = valid[valid["lesion_volume_mm3"] > 0]
    ax.scatter(valid_pos["lesion_volume_mm3"], valid_pos["dice"], c="#4CAF50", s=60, edgecolors="k", linewidths=0.5, zorder=5)
    if len(valid_pos) >= 3:
        rho, pval = stats.spearmanr(valid_pos["lesion_volume_mm3"], valid_pos["dice"])
        log_vol = np.log10(valid_pos["lesion_volume_mm3"])
        z = np.polyfit(log_vol, valid_pos["dice"], 1)
        x_line = np.linspace(log_vol.min(), log_vol.max(), 50)
        ax.plot(10**x_line, np.polyval(z, x_line), "k--", alpha=0.5)
        ax.set_title(f"(c) DSC vs Lesion Volume  ($\\rho$={rho:.3f}, p={pval:.3f})")
    else:
        ax.set_title("(c) DSC vs Lesion Volume")
    ax.set_xscale("log")
    ax.set_xlabel("Lesion Volume (mm³)")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)

    # (d) DSC vs GT Volume (from segmentation evaluation)
    ax = axes[1, 1]
    valid = df.dropna(subset=["gt_volume_ml"])
    valid_pos = valid[valid["gt_volume_ml"] > 0]
    ax.scatter(valid_pos["gt_volume_ml"], valid_pos["dice"], c="#9C27B0", s=60, edgecolors="k", linewidths=0.5, zorder=5)
    if len(valid_pos) >= 3:
        rho, pval = stats.spearmanr(valid_pos["gt_volume_ml"], valid_pos["dice"])
        log_vol = np.log10(valid_pos["gt_volume_ml"])
        z = np.polyfit(log_vol, valid_pos["dice"], 1)
        x_line = np.linspace(log_vol.min(), log_vol.max(), 50)
        ax.plot(10**x_line, np.polyval(z, x_line), "k--", alpha=0.5)
        ax.set_title(f"(d) DSC vs GT Volume  ($\\rho$={rho:.3f}, p={pval:.3f})")
    else:
        ax.set_title("(d) DSC vs GT Volume (mL)")
    ax.set_xscale("log")
    ax.set_xlabel("Ground Truth Volume (mL)")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {OUT_FIG}")


def print_summary(df):
    """Print correlation summary and worst/best cases."""
    print("\n" + "=" * 65)
    print("  Per-Subject DSC Summary (n={})".format(len(df)))
    print("=" * 65)
    print(f"  Mean DSC: {df['dice'].mean():.4f} +/- {df['dice'].std():.4f}")
    print(f"  Median:   {df['dice'].median():.4f}")
    print(f"  Range:    [{df['dice'].min():.4f}, {df['dice'].max():.4f}]")

    # Worst 5
    print("\n  --- Worst 5 cases ---")
    worst = df.nsmallest(5, "dice")
    for _, row in worst.iterrows():
        extras = []
        if pd.notna(row.get("age")):
            extras.append(f"Age={row['age']:.0f}")
        if pd.notna(row.get("edss")):
            extras.append(f"EDSS={row['edss']}")
        if pd.notna(row.get("lesion_volume_mm3")):
            extras.append(f"Vol={row['lesion_volume_mm3']:.0f}mm³")
        if pd.notna(row.get("sex")):
            extras.append(f"Sex={row['sex']}")
        ex = ", ".join(extras)
        print(f"    {row['case_id']} ({row['patient']}): DSC={row['dice']:.4f}  [{ex}]")

    # Best 5
    print("\n  --- Best 5 cases ---")
    best = df.nlargest(5, "dice")
    for _, row in best.iterrows():
        extras = []
        if pd.notna(row.get("age")):
            extras.append(f"Age={row['age']:.0f}")
        if pd.notna(row.get("edss")):
            extras.append(f"EDSS={row['edss']}")
        if pd.notna(row.get("lesion_volume_mm3")):
            extras.append(f"Vol={row['lesion_volume_mm3']:.0f}mm³")
        ex = ", ".join(extras)
        print(f"    {row['case_id']} ({row['patient']}): DSC={row['dice']:.4f}  [{ex}]")

    # Spearman correlations
    print("\n  --- Spearman Correlations (DSC vs ...) ---")
    for col, label in [
        ("age", "Age"),
        ("edss", "EDSS"),
        ("lesion_volume_mm3", "Lesion Volume (mm³)"),
        ("gt_volume_ml", "GT Volume (mL)"),
    ]:
        if col in df.columns:
            valid = df.dropna(subset=[col, "dice"])
            if len(valid) >= 3:
                rho, pval = stats.spearmanr(valid[col], valid["dice"])
                sig = "*" if pval < 0.05 else ""
                print(f"    {label:<22s}: rho={rho:+.3f}, p={pval:.4f} {sig}")

    # Sex comparison
    if "sex" in df.columns:
        print("\n  --- DSC by Sex ---")
        for sex, g in df.groupby("sex"):
            print(f"    {sex}: DSC={g['dice'].mean():.4f} +/- {g['dice'].std():.4f} (n={len(g)})")

    # MS Type comparison
    if "ms_type" in df.columns:
        print("\n  --- DSC by MS Type ---")
        for mt, g in df.groupby("ms_type"):
            print(f"    {mt}: DSC={g['dice'].mean():.4f} +/- {g['dice'].std():.4f} (n={len(g)})")


def main():
    print("=" * 65)
    print("  Per-Subject Failure Mode Analysis — DS001 Test Set")
    print("  Ensemble: Best-2/arch (6 models, thr=0.50)")
    print("=" * 65)

    print("\n  Computing per-subject metrics...")
    metrics_df = compute_per_subject_metrics()
    print(f"  Computed metrics for {len(metrics_df)} test subjects")

    print("\n  Merging with clinical data...")
    df = merge_with_clinical(metrics_df)

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"  Saved CSV: {OUT_CSV}")

    print_summary(df)
    generate_figure(df)


if __name__ == "__main__":
    main()
