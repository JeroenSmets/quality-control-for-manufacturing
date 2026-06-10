#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements-base.txt

echo "Select a hardware target for PyTorch installation:"
echo "  1) NVIDIA CUDA (recommended for an NVIDIA GPU)"
echo "  2) CPU-only"
echo "  3) ROCm (AMD GPU/Linux only; use only if you have a compatible ROCm setup)"
echo "  4) Skip PyTorch install"
read -rp "Enter selection [1-4]: " choice

case "$choice" in
  1)
    echo "Installing PyTorch and TorchVision for NVIDIA CUDA..."
    python3 -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121
    ;;
  2)
    echo "Installing CPU-only PyTorch and TorchVision..."
    python3 -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
    ;;
  3)
    echo "Installing ROCm-compatible PyTorch and TorchVision..."
    python3 -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/rocm7.8
    ;;
  *)
    echo "Skipping PyTorch install. Install PyTorch manually later."
    ;;
esac

echo "Installation complete."
