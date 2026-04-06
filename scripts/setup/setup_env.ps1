# ==============================================================================
# setup_env.ps1
# PowerShell equivalent: set nnU-Net v2 env vars and build directory scaffold.
#
# Usage:
#   . .\setup_env.ps1                          # default base
#   . .\setup_env.ps1 -BaseDir "D:\nnunet"     # custom base
# ==============================================================================
param(
    [string]$BaseDir = "$env:USERPROFILE\nnunet_workspace"
)

# -- Set environment variables (session-wide) --------------------------------
$env:nnUNet_raw            = Join-Path $BaseDir "nnUNet_raw"
$env:nnUNet_preprocessed   = Join-Path $BaseDir "nnUNet_preprocessed"
$env:nnUNet_results        = Join-Path $BaseDir "nnUNet_results"

# Persist to user-level so new shells inherit them
[System.Environment]::SetEnvironmentVariable("nnUNet_raw",          $env:nnUNet_raw,          "User")
[System.Environment]::SetEnvironmentVariable("nnUNet_preprocessed", $env:nnUNet_preprocessed, "User")
[System.Environment]::SetEnvironmentVariable("nnUNet_results",      $env:nnUNet_results,      "User")

# -- Create top-level directories -------------------------------------------
foreach ($dir in @($env:nnUNet_raw, $env:nnUNet_preprocessed, $env:nnUNet_results)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "[setup] Created  $dir"
}

# -- Build dataset sub-folders ----------------------------------------------
$datasets = @("Dataset500_RawFLAIR", "Dataset501_SkullStrippedFLAIR")
$subdirs  = @("imagesTr", "imagesTs", "labelsTr", "labelsTs")

foreach ($ds in $datasets) {
    foreach ($sub in $subdirs) {
        $p = Join-Path $env:nnUNet_raw "$ds\$sub"
        New-Item -ItemType Directory -Path $p -Force | Out-Null
    }
    Write-Host "[setup] Scaffolded  $(Join-Path $env:nnUNet_raw $ds)"
}

Write-Host ""
Write-Host "================================================================="
Write-Host "  nnU-Net v2 environment ready"
Write-Host "  nnUNet_raw            = $env:nnUNet_raw"
Write-Host "  nnUNet_preprocessed   = $env:nnUNet_preprocessed"
Write-Host "  nnUNet_results        = $env:nnUNet_results"
Write-Host "================================================================="
