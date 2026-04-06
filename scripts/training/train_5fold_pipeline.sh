#!/usr/bin/env bash
# =============================================================================
# train_5fold_pipeline.sh -- 5-Fold CV Pipeline for Best 3D Architecture
# =============================================================================
#
# Trains ResEncL-3D (nnUNetResEncUNetLPlans) with nnUNetTrainer_WandB
# on all 5 folds for DS004 (Combined with multi-size lesion labels).
#
# Multi-size labeling:
#   - Class 0: Background
#   - Class 1: Small lesions (<100mm^3)
#   - Class 2: Medium lesions (100-1000mm^3)
#   - Class 3: Large lesions (>1000mm^3)
#
# Uses nnUNetTrainer_WandB (VRAM probing, gradient checkpointing,
# Rician noise, elastic deform, dropout, bf16, W&B logging)
#
# Test data is NEVER touched: nnU-Net CV splits only use imagesTr cases.
# imagesTs/ is completely separate and not referenced by splits_final.json.
#
# Usage:
#   screen -S fivefold
#   source venv/bin/activate
#   bash scripts/training/train_5fold_pipeline.sh 2>&1 | tee fivefold.log
#
# Resume from specific dataset/fold:
#   bash scripts/training/train_5fold_pipeline.sh --start-ds 1 --start-fold 3
#
# =============================================================================

set -euo pipefail

# -- Environment (set these before running, or use scripts/setup/setup_env.sh) -
# export nnUNet_raw=/path/to/nnUNet_raw
# export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
# export nnUNet_results=/path/to/nnUNet_results
if [ -z "${nnUNet_raw:-}" ]; then echo "ERROR: Set nnUNet_raw env var"; exit 1; fi
if [ -z "${nnUNet_preprocessed:-}" ]; then echo "ERROR: Set nnUNet_preprocessed env var"; exit 1; fi
if [ -z "${nnUNet_results:-}" ]; then echo "ERROR: Set nnUNet_results env var"; exit 1; fi
export nnUNet_compile=false          # torch.compile hangs on PyTorch 2.5+Linux
export WANDB_MODE=online             # real-time W&B dashboard
export KMP_DUPLICATE_LIB_OK=TRUE

# -- Training configuration ------------------------------------------------
TRAINER="nnUNetTrainer_WandB"
PLANS="nnUNetResEncUNetLPlans"       # ResEncL architecture
CONFIG="3d_fullres"
NUM_FOLDS=5

# DS004 = Combined dataset with multi-size lesion labels (small/medium/large)
DATASETS=("Dataset004_Combined_MultiSize")
DS_IDS=(4)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# -- Parse arguments -------------------------------------------------------
START_DS=0
START_FOLD=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-ds)    START_DS=$2;   shift 2;;
        --start-fold)  START_FOLD=$2; shift 2;;
        *)             echo "Unknown arg: $1"; exit 1;;
    esac
done

echo ""
echo "==================================================================="
echo "  5-FOLD CV PIPELINE -- ResEncL-3D Multi-Size Labels"
echo "==================================================================="
echo "  Trainer:  ${TRAINER}"
echo "  Plans:    ${PLANS}"
echo "  Config:   ${CONFIG}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Folds:    0-$((NUM_FOLDS - 1))"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
echo "  GPU:      ${GPU_NAME}"
echo "  Time:     $(timestamp)"
echo "  Resume:   DS index=${START_DS}, fold=${START_FOLD}"
echo "==================================================================="
echo ""

# -- Disk space check ------------------------------------------------------
check_disk() {
    local avail_gb
    avail_gb=$(df -BG --output=avail ~ | tail -1 | tr -d ' G')
    echo "[$(timestamp)] Disk space: ${avail_gb} GB available"
    if (( avail_gb < 10 )); then
        echo "[$(timestamp)] CRITICAL: Less than 10 GB free! Aborting."
        exit 1
    elif (( avail_gb < 20 )); then
        echo "[$(timestamp)] WARNING: Less than 20 GB free. Monitor closely."
    fi
}

# -- Preprocessing ---------------------------------------------------------
ensure_preprocessed() {
    local dataset=$1
    local ds_id=$2
    local preproc_dir="${nnUNet_preprocessed}/${dataset}/nnUNetPlans_3d_fullres"

    if [[ -d "${preproc_dir}" ]]; then
        local n_files
        n_files=$(find "${preproc_dir}" -name "*.b2nd" 2>/dev/null | wc -l)
        echo "[$(timestamp)] Preprocessed data exists for ${dataset} (${n_files} .b2nd files)"
        return 0
    fi

    echo "[$(timestamp)] Preprocessing ${dataset} (id=${ds_id})..."
    check_disk

    python -u -m nnunetv2.experiment_planning.plan_and_preprocess \
        -d "${ds_id}" \
        --verify_dataset_integrity \
        -c 3d_fullres \
        --clean

    echo "[$(timestamp)] Preprocessing complete for ${dataset}"
}

# -- Train a single fold ---------------------------------------------------
train_fold() {
    local dataset=$1
    local fold=$2
    local result_dir="${nnUNet_results}/${dataset}/${TRAINER}__${PLANS}__${CONFIG}"
    local fold_dir="${result_dir}/fold_${fold}"

    # Check if already completed
    if [[ -f "${fold_dir}/checkpoint_final.pth" ]]; then
        echo "[$(timestamp)] SKIP: ${dataset} fold ${fold} -- already completed"
        return 0
    fi

    # Check for checkpoint to resume
    local resume_flag=""
    if [[ -f "${fold_dir}/checkpoint_latest.pth" ]] || \
       [[ -f "${fold_dir}/checkpoint_best.pth" ]]; then
        resume_flag="--c"
        echo "[$(timestamp)] Resuming ${dataset} fold ${fold} from existing checkpoint"
    fi

    check_disk

    echo ""
    echo "+--------------------------------------------------------------+"
    echo "|  Training: ${dataset}"
    echo "|  Fold: ${fold}/4  |  Trainer: ${TRAINER}"
    echo "|  Plans: ${PLANS}  |  Config: ${CONFIG}"
    echo "|  Time: $(timestamp)"
    echo "+--------------------------------------------------------------+"
    echo ""

    local start_time
    start_time=$(date +%s)

    # Run training
    python -u -m nnunetv2.run.run_training \
        "${dataset}" "${CONFIG}" "${fold}" \
        -tr "${TRAINER}" \
        -p "${PLANS}" \
        ${resume_flag}

    local elapsed=$(( $(date +%s) - start_time ))
    local hours=$(( elapsed / 3600 ))
    local mins=$(( (elapsed % 3600) / 60 ))
    echo ""
    echo "[$(timestamp)] ${dataset} fold ${fold} COMPLETED in ${hours}h ${mins}m"
    echo ""
}

# -- Main pipeline ---------------------------------------------------------
TOTAL_FOLDS=$(( ${#DATASETS[@]} * NUM_FOLDS ))
RUN_COUNT=0
FAILED=()

for ds_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$ds_idx]}"
    ds_id="${DS_IDS[$ds_idx]}"

    # Skip datasets before --start-ds
    if (( ds_idx < START_DS )); then
        echo "[$(timestamp)] Skipping ${dataset} (--start-ds ${START_DS})"
        RUN_COUNT=$(( RUN_COUNT + NUM_FOLDS ))
        continue
    fi

    echo ""
    echo "==================================================================="
    echo "  Dataset: ${dataset} (id=${ds_id})"
    echo "==================================================================="

    # Ensure preprocessed data exists
    ensure_preprocessed "${dataset}" "${ds_id}"

    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        # Skip folds before --start-fold (only for first non-skipped dataset)
        if (( ds_idx == START_DS && fold < START_FOLD )); then
            echo "[$(timestamp)] Skipping fold ${fold} (--start-fold ${START_FOLD})"
            RUN_COUNT=$(( RUN_COUNT + 1 ))
            continue
        fi

        RUN_COUNT=$(( RUN_COUNT + 1 ))
        echo "[$(timestamp)] === Progress: ${RUN_COUNT}/${TOTAL_FOLDS} ==="

        if ! train_fold "${dataset}" "${fold}"; then
            FAILED+=("${dataset}__fold_${fold}")
            echo "[$(timestamp)] FAILED: ${dataset} fold ${fold} -- continuing to next"
        fi
    done
done

# -- Summary ---------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  5-FOLD PIPELINE COMPLETE -- $(timestamp)"
echo "  Total runs attempted: ${RUN_COUNT}"
echo "  Failed: ${#FAILED[@]}"
if (( ${#FAILED[@]} > 0 )); then
    echo "  Failed folds:"
    for f in "${FAILED[@]}"; do
        echo "    - ${f}"
    done
fi
echo "==================================================================="
