<#
Install the repository dependencies on Windows.
Run this script from the repository root inside the desired Python virtual environment.
#>

$ErrorActionPreference = 'Stop'

function Get-PythonExe {
    if (Test-Path ".venv\Scripts\python.exe") {
        return Resolve-Path ".venv\Scripts\python.exe"
    }
    return "python"
}

$python = Get-PythonExe
Write-Host "Using Python executable: $python"
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing base dependencies..."
& $python -m pip install -r requirements-base.txt

Write-Host "Select a hardware target for PyTorch installation:"
Write-Host "  1) NVIDIA CUDA (recommended for an NVIDIA GPU)"
Write-Host "  2) CPU-only"
Write-Host "  3) ROCm (AMD GPU/Linux only; use only if you have a compatible ROCm setup)"
Write-Host "  4) Skip PyTorch install"
$choice = Read-Host "Enter selection [1-4]"

switch ($choice) {
    '1' {
        Write-Host "Installing PyTorch and TorchVision for NVIDIA CUDA..."
        & $python -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121
    }
    '2' {
        Write-Host "Installing CPU-only PyTorch and TorchVision..."
        & $python -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
    }
    '3' {
        Write-Host "Installing ROCm-compatible PyTorch and TorchVision..."
        & $python -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/rocm7.8
    }
    default {
        Write-Host "Skipping PyTorch install. You can install PyTorch manually later."
    }
}

Write-Host "Installation complete. If you installed PyTorch successfully, you can now run the pipeline scripts."
