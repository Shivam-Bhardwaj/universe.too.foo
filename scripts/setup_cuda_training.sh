#!/bin/bash
# CUDA Training Environment Setup
# Based on whitepaper Section 7.1
#
# This script configures the environment for CUDA-accelerated
# Gaussian splatting training using tch-rs (LibTorch bindings).

set -e

echo "=== Universe CUDA Training Setup ==="
echo ""

# 1. Check CUDA availability
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA Toolkit 11.8 or 12.x"
    echo "  Ubuntu: sudo apt install nvidia-cuda-toolkit"
    echo "  Or download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "  CUDA version: $CUDA_VERSION"

# 2. Check for PyTorch (for bundled libtorch)
echo ""
echo "Checking PyTorch installation..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Installing with CUDA support..."
    echo "  This may take a few minutes..."

    # Detect CUDA version for PyTorch wheel
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    if [ "$CUDA_MAJOR" -ge "12" ]; then
        pip3 install torch --index-url https://download.pytorch.org/whl/cu121
    else
        pip3 install torch --index-url https://download.pytorch.org/whl/cu118
    fi
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "  PyTorch version: $TORCH_VERSION"

TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
echo "  PyTorch path: $TORCH_PATH"

# 3. Check CUDA availability in PyTorch
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo ""
    echo "WARNING: PyTorch CUDA not available!"
    echo "  This usually means the PyTorch version doesn't match your CUDA version."
    echo "  Try reinstalling PyTorch with the correct CUDA index URL."
    echo ""
fi

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "  CUDA available: $CUDA_AVAILABLE (${GPU_COUNT} device(s))"

if [ "$GPU_COUNT" -gt "0" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo "  GPU 0: $GPU_NAME"
fi

# 4. Set environment variables
echo ""
echo "Setting environment variables..."

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$TORCH_PATH/lib:$LD_LIBRARY_PATH"

# Handle potential library preload issues
if [ -f "$TORCH_PATH/lib/libtorch_cuda.so" ]; then
    export LD_PRELOAD="$TORCH_PATH/lib/libtorch_cuda.so"
fi

echo "  LIBTORCH_USE_PYTORCH=1"
echo "  LD_LIBRARY_PATH includes: $TORCH_PATH/lib"

# 5. Build with torch feature
echo ""
echo "Building universe-train with CUDA support..."
echo "  cargo build --release --features torch -p universe-train"
echo ""

cd "$(dirname "$0")/.."
cargo build --release --features torch -p universe-train

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To train with CUDA backend, run:"
echo "  export LIBTORCH_USE_PYTORCH=1"
echo "  export LD_LIBRARY_PATH=\"$TORCH_PATH/lib:\$LD_LIBRARY_PATH\""
echo ""
echo "  cargo run --release --features torch -p universe-cli -- train-all \\"
echo "      --input universe_gaia_poc \\"
echo "      --output universe_gaia_poc_trained \\"
echo "      --backend torch-cuda \\"
echo "      --iterations 1000"
echo ""
echo "Or source this script to set up the environment:"
echo "  source scripts/setup_cuda_training.sh"
