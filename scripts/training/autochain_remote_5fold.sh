#!/bin/bash
set -e

# Set these environment variables before running, or use scripts/setup/setup_env.sh
# export nnUNet_raw=/path/to/nnUNet_raw
# export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
# export nnUNet_results=/path/to/nnUNet_results
if [ -z "${nnUNet_raw:-}" ]; then echo "ERROR: Set nnUNet_raw env var"; exit 1; fi
if [ -z "${nnUNet_preprocessed:-}" ]; then echo "ERROR: Set nnUNet_preprocessed env var"; exit 1; fi
if [ -z "${nnUNet_results:-}" ]; then echo "ERROR: Set nnUNet_results env var"; exit 1; fi
export nnUNet_compile=false
export WANDB_MODE=online

echo "=========================================="
echo "  AUTOCHAIN: ResEncL_3D folds 1-4 + ResEncL_25D folds 1-4"
echo "  Started: $(date)"
echo "=========================================="

# --- Phase 1: ResEncL_3D folds 1-4 ---
for FOLD in 1 2 3 4; do
    FOLD_DIR="$nnUNet_results/Dataset003_Combined/nnUNetTrainer_WandB__nnUNetResEncUNetLPlans__3d_fullres/fold_${FOLD}"
    FINAL="$FOLD_DIR/checkpoint_final.pth"

    if [ -f "$FINAL" ]; then
        echo "[SKIP] ResEncL_3D fold $FOLD already done."
        continue
    fi

    echo "[START] ResEncL_3D fold $FOLD at $(date)"

    RESUME=""
    if [ -f "$FOLD_DIR/checkpoint_latest.pth" ]; then
        RESUME="--c"
        echo "  Resuming from checkpoint..."
    fi

    python -u -m nnunetv2.run.run_training 3 3d_fullres $FOLD \
        -tr nnUNetTrainer_WandB -p nnUNetResEncUNetLPlans --npz $RESUME \
        2>&1 | tee "resencl3d_fold${FOLD}.log"

    echo "[DONE] ResEncL_3D fold $FOLD at $(date)"
done

echo "=========================================="
echo "  ResEncL_3D folds 1-4 COMPLETE at $(date)"
echo "  Starting ResEncL_25D folds 1-4..."
echo "=========================================="

# --- Phase 2: ResEncL_25D folds 1-4 ---
for FOLD in 1 2 3 4; do
    FOLD_DIR="$nnUNet_results/Dataset003_Combined/nnUNetTrainer_25D__nnUNetResEncUNetLPlans__3d_fullres/fold_${FOLD}"
    FINAL="$FOLD_DIR/checkpoint_final.pth"

    if [ -f "$FINAL" ]; then
        echo "[SKIP] ResEncL_25D fold $FOLD already done."
        continue
    fi

    echo "[START] ResEncL_25D fold $FOLD at $(date)"

    RESUME=""
    if [ -f "$FOLD_DIR/checkpoint_latest.pth" ]; then
        RESUME="--c"
        echo "  Resuming from checkpoint..."
    fi

    python -u -m nnunetv2.run.run_training 3 3d_fullres $FOLD \
        -tr nnUNetTrainer_25D -p nnUNetResEncUNetLPlans --npz $RESUME \
        2>&1 | tee "resencl25d_fold${FOLD}.log"

    echo "[DONE] ResEncL_25D fold $FOLD at $(date)"
done

echo "=========================================="
echo "  ALL TRAINING COMPLETE at $(date)"
echo "  ResEncL_3D folds 1-4"
echo "  ResEncL_25D folds 1-4"
echo "=========================================="
