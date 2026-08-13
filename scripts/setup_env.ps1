# Create the Python environment for the ML backend.
#
# PyTorch is installed separately from requirements.txt because the correct
# wheel depends on the CUDA runtime, and pip cannot choose for you.
#
#   .\scripts\setup_env.ps1            # CUDA 12.4 build
#   .\scripts\setup_env.ps1 -Cpu       # CPU-only build

param(
    [switch]$Cpu,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv" -ForegroundColor Cyan
    & $Python -m venv .venv
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip

$index = if ($Cpu) {
    "https://download.pytorch.org/whl/cpu"
} else {
    "https://download.pytorch.org/whl/cu124"
}

Write-Host "Installing PyTorch from $index" -ForegroundColor Cyan
& $venvPython -m pip install torch torchvision torchaudio --index-url $index

Write-Host "Installing the remaining requirements" -ForegroundColor Cyan
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install -r requirements-dev.txt

Write-Host "`nChecking the environment" -ForegroundColor Cyan
& $venvPython scripts\check_env.py

Write-Host "`nNext: python scripts\download_checkpoint.py" -ForegroundColor Green
