#!/bin/bash
set -e

# Session: Qwen3-VL-8B baseline eval + probe
# Tests if the decoding bottleneck is universal
# ~2 hours on A100 80GB, no training needed

echo "=== Qwen3-VL Bottleneck Test ==="
echo "Start time: $(date)"

# ---- Conda fix ----
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda deactivate 2>/dev/null || true
fi

# ---- GPU check ----
echo "Checking GPU..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Quick allocation test
    t = torch.zeros(1).cuda()
    del t
    print('GPU allocation test: OK')
else:
    print('ERROR: No CUDA GPU')
    exit(1)
"

# ---- Deps ----
echo "Installing dependencies..."
pip install --break-system-packages -q "transformers>=4.57.0" qwen-vl-utils peft scikit-learn 2>/dev/null || true

# Verify transformers version
python3 -c "
import transformers
v = transformers.__version__
print(f'transformers {v}')
major, minor = int(v.split('.')[0]), int(v.split('.')[1])
if major < 4 or (major == 4 and minor < 57):
    print(f'ERROR: Need transformers >= 4.57.0, got {v}')
    print('Try: pip install --break-system-packages git+https://github.com/huggingface/transformers')
    exit(1)
print('Version OK')
"

# Verify Qwen3VL class exists
python3 -c "
from transformers import Qwen3VLForConditionalGeneration
print('Qwen3VLForConditionalGeneration: available')
" || {
    echo "Qwen3VL not in transformers. Installing from source..."
    pip install --break-system-packages -q git+https://github.com/huggingface/transformers
    python3 -c "from transformers import Qwen3VLForConditionalGeneration; print('OK')" || {
        echo "FATAL: Cannot import Qwen3VLForConditionalGeneration"
        exit 1
    }
}

# ---- Dataset ----
if [ ! -d "dataset/train" ]; then
    echo "Generating dataset..."
    python3 generate_dataset.py --n-train 1000 --n-test 200 --seed 42 --output-dir dataset
fi

# Ensure ground_truth.npy exists for probe
if [ ! -f "embeddings/ground_truth.npy" ]; then
    mkdir -p embeddings
    python3 -c "
import json, numpy as np
from pathlib import Path
with open('dataset/test/metadata.jsonl') as f:
    samples = [json.loads(l) for l in f]
gt = {s['idx']: s['diameter_mm'] for s in samples}
np.save('embeddings/ground_truth.npy', gt)
print(f'Saved ground truth for {len(gt)} samples')
"
fi

# ---- Run eval + probe ----
echo "=========================================="
echo "  Qwen3-VL-8B: Baseline Eval + Probe"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 eval_qwen3vl.py --all

# ---- Push ----
echo "=========================================="
echo "  Pushing Results"
echo "=========================================="
git add results/qwen3vl_baseline/ embeddings/qwen3vl_* embeddings/ground_truth.npy 2>/dev/null || true
git commit -m "Qwen3-VL-8B baseline: bottleneck 1 test" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
