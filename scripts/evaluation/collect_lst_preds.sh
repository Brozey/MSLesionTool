#!/bin/bash
# collect_lst_preds.sh -- Collect LST-AI predictions into a single directory
#
# Usage:
#   export LST_OUTPUT_DIR=/path/to/lst_ai/output
#   export LST_PRED_DIR=/path/to/lst_ai/predictions
#   bash collect_lst_preds.sh

export PATH=/usr/bin:/usr/local/bin:$PATH

if [ -z "${LST_OUTPUT_DIR:-}" ]; then echo "ERROR: Set LST_OUTPUT_DIR env var"; exit 1; fi
if [ -z "${LST_PRED_DIR:-}" ]; then echo "ERROR: Set LST_PRED_DIR env var"; exit 1; fi

mkdir -p "$LST_PRED_DIR"
for d in "$LST_OUTPUT_DIR"/MSL_*; do
    subj=$(basename "$d")
    cp "$d/space-flair_seg-lst.nii.gz" "$LST_PRED_DIR/${subj}.nii.gz"
done
echo "Copied $(ls "$LST_PRED_DIR"/*.nii.gz | wc -l) predictions"
