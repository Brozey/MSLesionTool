# Retrain 2.5D folds 1, 2, 4 to 1000 epochs (they stopped prematurely via curve-fit)
# Early stopping is DISABLED in the trainer (PATIENCE=9999, CURVE_FIT_MIN=9999)
# After these finish, RESTORE early stopping settings!

# Set these environment variables before running:
# $env:nnUNet_raw = 'path/to/nnUNet_raw'
# $env:nnUNet_preprocessed = 'path/to/nnUNet_preprocessed'
# $env:nnUNet_results = 'path/to/nnUNet_results'
if (-not $env:nnUNet_raw) { Write-Error "Set nnUNet_raw env var"; exit 1 }
if (-not $env:nnUNet_preprocessed) { Write-Error "Set nnUNet_preprocessed env var"; exit 1 }
if (-not $env:nnUNet_results) { Write-Error "Set nnUNet_results env var"; exit 1 }
$env:WANDB_MODE = 'online'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:nnUNet_compile = 'false'

Write-Host '=== RETRAIN: 2.5D folds 1, 2, 4 to 1000 epochs ==='
Write-Host '  Early stopping: DISABLED'

foreach ($fold in 1, 2, 4) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[START] Fold $fold at $ts (resuming from checkpoint)"

    & python -m nnunetv2.run.run_training 3 3d_fullres $fold -tr nnUNetTrainer_25D -p nnUNetPlans --npz --c

    $exitCode = $LASTEXITCODE
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if ($exitCode -ne 0) {
        Write-Host "[ERROR] Fold $fold crashed (exit $exitCode) at $ts. Continuing..."
    } else {
        Write-Host "[DONE] Fold $fold at $ts"
    }
}

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host '========================================'
Write-Host "  Retrain complete at $ts"
Write-Host '  REMEMBER: Restore early stopping in nnUNetTrainer_25D.py!'
Write-Host '    EARLY_STOP_PATIENCE = 150'
Write-Host '    CURVE_FIT_MIN_EPOCHS = 60'
Write-Host '========================================'
