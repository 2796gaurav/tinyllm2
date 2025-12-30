#!/bin/bash
# TinyGuardrail Production Setup
# Verifies environment and prepares for training

set -e

echo "================================================================================"
echo "🔧 TinyGuardrail Production Setup"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "   Python: $PYTHON_VERSION"

REQUIRED_PYTHON="3.8"
if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}❌ Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}   ✅ Python version OK${NC}"
echo ""

# Check HF_TOKEN
echo "🔑 Checking HuggingFace token..."
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}❌ HF_TOKEN not set!${NC}"
    echo ""
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN='hf_your_token_here'"
    echo ""
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo ""
    exit 1
else
    echo -e "${GREEN}   ✅ HF_TOKEN found${NC}"
    # Mask token for security
    MASKED_TOKEN="${HF_TOKEN:0:7}...${HF_TOKEN: -4}"
    echo "   Token: $MASKED_TOKEN"
fi
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✅ Dependencies installed${NC}"
else
    echo -e "${RED}   ❌ Dependency installation failed${NC}"
    exit 1
fi
echo ""

# Verify key packages
echo "🔍 Verifying key packages..."

PACKAGES=(
    "torch"
    "transformers"
    "datasets"
    "huggingface_hub"
    "optuna"
    "onnx"
    "onnxruntime"
    "sklearn"
    "pandas"
    "numpy"
)

for package in "${PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}   ✅ $package ($VERSION)${NC}"
    else
        echo -e "${RED}   ❌ $package not found${NC}"
        exit 1
    fi
done
echo ""

# Check CUDA
echo "🖥️  Checking GPU availability..."
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')" 2>/dev/null)
    echo -e "${GREEN}   ✅ CUDA available${NC}"
    echo "   GPU: $GPU_NAME"
    echo "   Memory: ${GPU_MEM} GB"
else
    echo -e "${YELLOW}   ⚠️  No CUDA GPU found - training will be slow on CPU${NC}"
    echo "   Consider using Google Colab or cloud GPU"
fi
echo ""

# Test HuggingFace access
echo "🌐 Testing HuggingFace access..."
python -c "
from src.data.real_benchmark_loader import verify_hf_access
try:
    verify_hf_access()
    print('${GREEN}   ✅ HuggingFace access verified${NC}')
except Exception as e:
    print('${RED}   ❌ HuggingFace access failed${NC}')
    print(f'   Error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Please check:"
    echo "  1. HF_TOKEN is valid"
    echo "  2. You have internet connection"
    echo "  3. You accepted dataset licenses on HuggingFace"
    exit 1
fi
echo ""

# Create output directories
echo "📁 Creating output directories..."
mkdir -p outputs/production_final
mkdir -p outputs/optimized
mkdir -p outputs/dry_run
mkdir -p results/benchmarks
mkdir -p checkpoints/production_final
mkdir -p logs/production_final
echo -e "${GREEN}   ✅ Directories created${NC}"
echo ""

# Test model initialization
echo "🏗️  Testing model initialization..."
python -c "
import torch
from src.models.dual_branch import TinyGuardrail, DualBranchConfig

config = DualBranchConfig(
    vocab_size=30522,
    d_model=384,
    router_threshold=0.3,
)

model = TinyGuardrail(config)
print('${GREEN}   ✅ Model initializes correctly${NC}')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
echo ""

# Summary
echo "================================================================================"
echo "✅ SETUP COMPLETE - READY FOR PRODUCTION TRAINING"
echo "================================================================================"
echo ""
echo "📊 System Summary:"
echo "   Python:     $PYTHON_VERSION"
echo "   PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:       $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "   HF Access:  ✅ Verified"
echo ""
echo "🚀 Next steps:"
echo ""
echo "   Option 1: Quick dry run (5 minutes)"
echo "     python dry_run.py"
echo ""
echo "   Option 2: Full pipeline (6-8 hours)"
echo "     ./run_end_to_end.sh"
echo ""
echo "   Option 3: Training only (4-6 hours)"
echo "     python train_production_final.py --output_dir outputs/production --use_wandb"
echo ""
echo "================================================================================"


