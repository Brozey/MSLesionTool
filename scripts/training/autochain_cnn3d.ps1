# Auto-chain: waits for fold 1 to finish, then runs folds 2-4
# Polls every 60s for fold_1/checkpoint_final.pth
# DS008 FLAIR-only SKIPPED — prioritizing 3-arch 5-fold ensemble

# Set these environment variables before running:
# $env:nnUNet_raw = 'path/to/nnUNet_raw'
# $env:nnUNet_preprocessed = 'path/to/nnUNet_preprocessed'
# $env:nnUNet_results = 'path/to/nnUNet_results'
if (-not $env:nnUNet_raw) { Write-Error "Set nnUNet_raw env var"; exit 1 }
if (-not $env:nnUNet_preprocessed) { Write-Error "Set nnUNet_preprocessed env var"; exit 1 }
if (-not $env:nnUNet_results) { Write-Error "Set nnUNet_results env var"; exit 1 }
$env:WANDB_MODE = 'online'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:nnUNet_n_proc_DA = '12'

$ds003Base = "$env:nnUNet_results\Dataset003_Combined\nnUNetTrainer_WandB__nnUNetPlans__3d_fullres"
$fold1Final = Join-Path $ds003Base 'fold_1\checkpoint_final.pth'

Write-Host '=== AUTOCHAIN: CNN 3D folds 1-4 on DS003 ==='
Write-Host "  Monitoring: $fold1Final"
Write-Host '  Polling every 60 seconds...'

while (-not (Test-Path $fold1Final)) {
    Start-Sleep -Seconds 60
}

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "=== Fold 1 DONE at $ts! Starting folds 2-4 ==="

# DS003 CNN 3D folds 2, 3, 4
foreach ($fold in 2, 3, 4) {
    $foldDir = Join-Path $ds003Base "fold_$fold"
    $finalPth = Join-Path $foldDir 'checkpoint_final.pth'

    if (Test-Path $finalPth) {
        Write-Host "[SKIP] DS003 Fold $fold already complete."
        continue
    }

    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[START] DS003 CNN 3D Fold $fold at $ts"

    $args = @('3', '3d_fullres', "$fold", '-tr', 'nnUNetTrainer_WandB', '-p', 'nnUNetPlans', '--npz')
    $latestPth = Join-Path $foldDir 'checkpoint_latest.pth'
    if (Test-Path $latestPth) {
        $args += '--c'
        Write-Host '  Resuming from checkpoint...'
    }

    & python -m nnunetv2.run.run_training @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] DS003 Fold $fold crashed (exit $LASTEXITCODE). Continuing..."
    } else {
        $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Host "[DONE] DS003 Fold $fold at $ts"
    }
}

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host '========================================'
Write-Host "  CNN 3D 5-fold COMPLETE at $ts"
Write-Host '========================================'
