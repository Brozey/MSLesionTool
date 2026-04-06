# run_fold_all_chain.ps1
# Chains: wait for 2.5D fold 1 -> 2.5D fold_all -> CNN 3D fold_all
# Launch in a terminal with conda activated.

# Set these environment variables before running:
# $env:nnUNet_raw = 'path/to/nnUNet_raw'
# $env:nnUNet_preprocessed = 'path/to/nnUNet_preprocessed'
# $env:nnUNet_results = 'path/to/nnUNet_results'
if (-not $env:nnUNet_raw) { Write-Error "Set nnUNet_raw env var"; exit 1 }
if (-not $env:nnUNet_preprocessed) { Write-Error "Set nnUNet_preprocessed env var"; exit 1 }
if (-not $env:nnUNet_results) { Write-Error "Set nnUNet_results env var"; exit 1 }
$env:WANDB_MODE = "online"

$fold1Final = "$env:nnUNet_results\Dataset003_Combined\nnUNetTrainer_25D__nnUNetPlans__3d_fullres\fold_1\checkpoint_final.pth"

# -- Step 1: Wait for 2.5D fold 1 to finish --
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Waiting for 2.5D fold 1 to finish..."
Write-Host "  Watching: $fold1Final"

while (-not (Test-Path $fold1Final)) {
    Start-Sleep -Seconds 120
}
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | 2.5D fold 1 COMPLETE! checkpoint_final.pth found."
Start-Sleep -Seconds 30  # Brief pause to let file handles close

# -- Step 2: Train 2.5D fold_all (1000 epochs, no validation) --
Write-Host ""
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Starting 2.5D fold_all training (1000 epochs)..."
Write-Host "  Trainer: nnUNetTrainer_25D | Plans: nnUNetPlans | Config: 3d_fullres"
nnUNetv2_train 003 3d_fullres all -tr nnUNetTrainer_25D
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | 2.5D fold_all COMPLETE!"

# -- Step 3: Train CNN 3D fold_all (1000 epochs, no validation) --
Write-Host ""
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Starting CNN 3D fold_all training (1000 epochs)..."
Write-Host "  Trainer: nnUNetTrainer_WandB | Plans: nnUNetPlans | Config: 3d_fullres"
nnUNetv2_train 003 3d_fullres all -tr nnUNetTrainer_WandB
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | CNN 3D fold_all COMPLETE!"

Write-Host ""
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | ALL FOLD_ALL TRAINING DONE!"
Write-Host "  - 2.5D fold_all: $env:nnUNet_results\Dataset003_Combined\nnUNetTrainer_25D__nnUNetPlans__3d_fullres\fold_all\"
Write-Host "  - CNN 3D fold_all: $env:nnUNet_results\Dataset003_Combined\nnUNetTrainer_WandB__nnUNetPlans__3d_fullres\fold_all\"
