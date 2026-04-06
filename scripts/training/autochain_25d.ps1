# Auto-chain: 2.5D training folds 0-4 on DS003
# Uses nnUNetTrainer_25D with curve-fit early stopping, W&B, VRAM probing
# Runs sequentially: fold 0 -> 1 -> 2 -> 3 -> 4

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

$ds003Base = "$env:nnUNet_results\Dataset003_Combined\nnUNetTrainer_25D__nnUNetPlans__3d_fullres"

Write-Host '=== AUTOCHAIN: 2.5D folds 0-4 on DS003 ==='
Write-Host "  Results dir: $ds003Base"

foreach ($fold in 0, 1, 2, 3, 4) {
    $foldDir = Join-Path $ds003Base "fold_$fold"
    $finalPth = Join-Path $foldDir 'checkpoint_final.pth'

    if (Test-Path $finalPth) {
        Write-Host "[SKIP] DS003 Fold $fold already complete."
        continue
    }

    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[START] DS003 2.5D Fold $fold at $ts"

    $args = @('3', '3d_fullres', "$fold", '-tr', 'nnUNetTrainer_25D', '-p', 'nnUNetPlans', '--npz')
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
        Write-Host "[DONE] DS003 2.5D Fold $fold at $ts"
    }
}

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host '========================================'
Write-Host "  2.5D 5-fold COMPLETE at $ts"
Write-Host '========================================'
