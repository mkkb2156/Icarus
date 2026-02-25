#!/bin/bash
# Icarus Environment Setup Script
#
# Prerequisites:
#   - Ubuntu 22.04
#   - NVIDIA GPU with RT Cores (RTX 3070+ recommended, RTX 4090 ideal)
#   - NVIDIA Driver >= 535
#   - Python 3.10
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh
#
# This script will:
#   1. Create a Python virtual environment
#   2. Install Isaac Lab (includes Isaac Sim)
#   3. Install RL frameworks (rl_games)
#   4. Install project dependencies
#   5. Verify the installation

set -e

echo "=== Icarus Environment Setup ==="
echo ""

# ─── Step 1: Check prerequisites ─────────────────────────────────────

echo "[1/6] Checking prerequisites..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "WARNING: Python 3.10 is required for Isaac Sim. Found: Python $PYTHON_VERSION"
    echo "Install Python 3.10: sudo apt install python3.10 python3.10-venv"
    echo ""
fi

# Check NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  NVIDIA Driver: $DRIVER_VERSION"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME"
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  VRAM: $GPU_MEMORY"
else
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

echo ""

# ─── Step 2: Create virtual environment ──────────────────────────────

echo "[2/6] Creating Python virtual environment..."

if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists, skipping"
fi

source .venv/bin/activate
pip install --upgrade pip

echo ""

# ─── Step 3: Install Isaac Lab (includes Isaac Sim) ──────────────────

echo "[3/6] Installing Isaac Lab + Isaac Sim (this may take 10-20 minutes)..."
echo "  NOTE: First run will download ~30GB of assets"

pip install 'isaaclab[isaacsim,all]==2.3.2' --extra-index-url https://pypi.nvidia.com

# Install CUDA-enabled PyTorch
pip install -U torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo ""

# ─── Step 4: Install RL frameworks ───────────────────────────────────

echo "[4/6] Installing RL frameworks..."

pip install rl_games==1.6.1

echo ""

# ─── Step 5: Install project dependencies ────────────────────────────

echo "[5/6] Installing Icarus project..."

pip install -e ".[dev]"

# Install demo dependencies
pip install -r demo/requirements.txt

echo ""

# ─── Step 6: Verify installation ─────────────────────────────────────

echo "[6/6] Verifying installation..."

python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run Phase 0 baseline (verify Isaac Lab works):"
echo "     python -m isaaclab.train --task Isaac-Quadcopter-Direct-v0 --num_envs 4096 --headless --max_iterations 1000"
echo "  3. Quick test of FacadeDrone environment:"
echo "     python scripts/train.py --num_envs 64 --max_iterations 10"
echo "  4. Full training:"
echo "     python scripts/train.py --num_envs 4096 --headless --max_iterations 5000"
echo "  5. Launch demo dashboard:"
echo "     cd demo && streamlit run app.py"
