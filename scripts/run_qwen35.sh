#!/bin/bash
set -e

# Session: Qwen3.5-9B baseline eval + layer-by-layer probe
# Tests early fusion architecture for spatial decoding bottleneck
# ~3 hours on A100 80GB, no training needed

echo "=== Qwen3.5-9B Early Fusion Bottleneck Test ==="
echo "Start time: $(date)"

# ---- Conda fix ----
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda deactivate 2>/dev/null || true
fi

# ---- GPU check ----
echo "Checking GPU..."
CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(1).cuda()
    del t
    print('GPU allocation test: OK')
else:
    print('ERROR: No CUDA GPU')
    exit(1)
"

# ---- Deps ----
echo "Installing dependencies..."
# Qwen3.5 requires bleeding-edge transformers
pip install --break-system-packages -q "transformers @ git+https://github.com/huggingface/transformers.git@main" 2>/dev/null || true
pip install --break-system-packages -q qwen-vl-utils peft scikit-learn torchvision pillow 2>/dev/null || true

# Verify
python3 -c "
import transformers
print(f'transformers {transformers.__version__}')
from transformers import Qwen3_5ForConditionalGeneration
print('Qwen3_5ForConditionalGeneration: available')
" || {
    echo "FATAL: Cannot import Qwen3_5ForConditionalGeneration"
    echo "Try: pip install --break-system-packages 'transformers @ git+https://github.com/huggingface/transformers.git@main'"
    exit 1
}

# ---- Dataset ----
if [ ! -d "dataset/train" ]; then
    echo "Generating dataset..."
    python3 generate_dataset.py --n-train 1000 --n-test 200 --seed 42 --output-dir dataset
fi

if [ ! -f "embeddings/ground_truth.npy" ]; then
    mkdir -p embeddings
    python3 -c "
import json, numpy as np
with open('dataset/test/metadata.jsonl') as f:
    samples = [json.loads(l) for l in f]
gt = {s['idx']: s['diameter_mm'] for s in samples}
np.save('embeddings/ground_truth.npy', gt)
print(f'Saved ground truth for {len(gt)} samples')
"
fi

# ---- Eval + Probe ----
echo "=========================================="
echo "  Qwen3.5-9B: Baseline Eval + Layer Probe"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 eval_qwen35.py --all

# ---- Push ----
echo "=========================================="
echo "  Pushing Results"
echo "=========================================="
git add results/qwen35_baseline/ embeddings/qwen35_* embeddings/ground_truth.npy 2>/dev/null || true
git commit -m "Qwen3.5-9B early fusion: layer-by-layer spatial probe" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
