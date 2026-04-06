#!/bin/bash
# ==========================================================================
# run_ds003_improved_5fold.sh -- Improved loss function experiments on DS003
# ==========================================================================
#
# Trains DS003 (Combined, binary labels, 153 cases) with improved losses
# designed to beat MadSeg (0.714 Dice on MSLesSeg).
#
# Experiments (run sequentially):
#   1. ResEncL + TopK loss (hardest 10% voxels) -- 5-fold
#   2. CNN    + TopK loss                       -- 5-fold
#   3. ResEncL + Focal loss (gamma=2)           -- 5-fold
#
# Usage:
#   screen -S ds003improved
#   source venv/bin/activate
#   bash scripts/training/run_ds003_improved_5fold.sh 2>&1 | tee ds003_improved.log
#   # Ctrl+A, D to detach
#
# Resume: Re-run the script. Completed folds (checkpoint_final.pth) skipped.
#
# Data: DS003 raw + preprocessed must exist before running.
# ==========================================================================
set -euo pipefail

# -- Environment (set these before running, or use scripts/setup/setup_env.sh) -
# export nnUNet_raw=/path/to/nnUNet_raw
# export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
# export nnUNet_results=/path/to/nnUNet_results
if [ -z "${nnUNet_raw:-}" ]; then echo "ERROR: Set nnUNet_raw env var"; exit 1; fi
if [ -z "${nnUNet_preprocessed:-}" ]; then echo "ERROR: Set nnUNet_preprocessed env var"; exit 1; fi
if [ -z "${nnUNet_results:-}" ]; then echo "ERROR: Set nnUNet_results env var"; exit 1; fi
export nnUNet_compile=false
export WANDB_MODE=online

DATASET="Dataset003_Combined"
CONFIG="3d_fullres"

# -- Helper function -------------------------------------------------------
run_5fold() {
    local TRAINER=$1
    local PLANS=$2
    local TAG=$3

    echo ""
    echo "================================================================"
    echo "  ${TAG}: ${TRAINER} + ${PLANS}"
    echo "  Dataset: ${DATASET}  Config: ${CONFIG}"
    echo "  Started: $(date)"
    echo "================================================================"

    for FOLD in 0 1 2 3 4; do
        local FOLD_DIR="${nnUNet_results}/${DATASET}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

        echo ""
        echo "--- ${TAG} fold ${FOLD}/4 --- $(date)"

        if [ -f "${FOLD_DIR}/checkpoint_final.pth" ]; then
            echo "SKIP: already completed"
            continue
        fi

        RESUME_FLAG=""
        if [ -f "${FOLD_DIR}/checkpoint_latest.pth" ] || [ -f "${FOLD_DIR}/checkpoint_best.pth" ]; then
            RESUME_FLAG="--c"
        fi

        python -u -m nnunetv2.run.run_training ${DATASET} ${CONFIG} ${FOLD} \
            -tr ${TRAINER} -p ${PLANS} ${RESUME_FLAG}

        echo "DONE: ${TAG} fold ${FOLD} at $(date)"
    done

    echo ""
    echo "=== ${TAG} ALL 5 FOLDS COMPLETE === $(date)"
}

# -- Experiment 1: ResEncL + TopK (highest priority) -----------------------
run_5fold "nnUNetTrainer_WandB_TopK" "nnUNetResEncUNetLPlans" "EXP1-ResEncL-TopK"

# -- Experiment 2: CNN + TopK (architectural diversity) --------------------
run_5fold "nnUNetTrainer_WandB_TopK" "nnUNetPlans" "EXP2-CNN-TopK"

# -- Experiment 3: ResEncL + Focal (alternative hard-mining) ---------------
run_5fold "nnUNetTrainer_WandB_Focal" "nnUNetResEncUNetLPlans" "EXP3-ResEncL-Focal"

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE -- $(date)"
echo "  Results in: ${nnUNet_results}/${DATASET}/"
echo "================================================================"
