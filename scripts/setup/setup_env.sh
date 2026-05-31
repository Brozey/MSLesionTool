#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh
# Initialise nnU-Net v2 environment variables and directory scaffold.
#
# Usage:
#   source setup_env.sh              # uses default base
#   source setup_env.sh /data/nnunet # custom base directory
# ==============================================================================
set -euo pipefail

# -- Configurable root -----------------------------------------------------
BASE_DIR="${1:-$HOME/nnunet_workspace}"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${BASE_DIR}/nnUNet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

# -- Create top-level trees ------------------------------------------------
for DIR in "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"; do
    mkdir -p "$DIR"
    echo "[setup] Created  $DIR"
done

# -- Build dataset sub-folders inside nnUNet_raw ---------------------------
DATASETS=("Dataset500_RawFLAIR" "Dataset501_SkullStrippedFLAIR")
SUBDIRS=("imagesTr" "imagesTs" "labelsTr" "labelsTs")

for DS in "${DATASETS[@]}"; do
    for SUB in "${SUBDIRS[@]}"; do
        mkdir -p "${nnUNet_raw}/${DS}/${SUB}"
    done
    echo "[setup] Scaffolded  ${nnUNet_raw}/${DS}"
done

echo ""
echo "=============================================================="
echo "  nnU-Net v2 environment ready"
echo "  nnUNet_raw            = $nnUNet_raw"
echo "  nnUNet_preprocessed   = $nnUNet_preprocessed"
echo "  nnUNet_results        = $nnUNet_results"
echo "=============================================================="
