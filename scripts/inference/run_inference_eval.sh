#!/usr/bin/env bash
set -euo pipefail

# ── Environment check ─────────────────────────────────────────────────────────
for var in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Set $var environment variable (standard nnUNet setup)" >&2
        exit 1
    fi
done

export nnUNet_compile=false

DATASET=Dataset004_Combined_MultiSize
TRAINER=nnUNetTrainer_WandB
PLANS=nnUNetResEncUNetLPlans
CONFIG=3d_fullres

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

INPUT_DIR=$nnUNet_raw/$DATASET/imagesTs
OUTPUT_DIR=$REPO_ROOT/results/inference_output/DS004_5fold_multisize
GT_DIR=$nnUNet_raw/$DATASET/labelsTs

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Step 1: nnU-Net 5-fold ensemble inference"
echo "========================================"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Time:   $(date)"
echo ""

nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -d 4 \
    -tr "$TRAINER" \
    -p "$PLANS" \
    -c "$CONFIG" \
    -f 0 1 2 3 4 \
    --disable_tta \
    -npp 3 \
    -nps 3

echo ""
echo "Inference complete: $(date)"
echo ""

echo "========================================"
echo "  Step 2: Merge + Evaluate"
echo "========================================"

python3 "$(dirname "$0")/merge_and_evaluate.py" \
    --pred_dir "$OUTPUT_DIR" \
    --gt_dir "$GT_DIR"

echo ""
echo "Pipeline complete: $(date)"
