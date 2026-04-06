#!/bin/bash
# launch_ds003_improved.sh -- Screen wrapper for improved loss experiments
# Run from remote: screen -dmS ds003improved ~/ms_lesion_seg/launch_ds003_improved.sh
cd ~/ms_lesion_seg
source venv/bin/activate
echo "Activated venv: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Starting DS003 improved loss training pipeline..."
bash scripts/training/run_ds003_improved_5fold.sh 2>&1 | tee ds003_improved.log
echo "Pipeline finished at $(date)"
