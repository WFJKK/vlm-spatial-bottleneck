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
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: Cannot import torch"; exit 1
}

GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 5000 ]; then
    echo "ERROR: GPU has stale memory. Restart instance."; exit 1
fi

# ---- Deps ----
# Qwen3-VL requires transformers >= 4.57.0
pip install --break-system-packages -q "transformers>=4.57.0" qwen-vl-utils peft scikit-learn huggingface_hub 2>/dev/null || true

# Verify transformers version
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"

# ---- Dataset ----
if [ ! -d "dataset/train" ]; then
    python3 generate_dataset.py --n-train 1000 --n-test 200 --seed 42 --output-dir dataset
fi

# Need ground_truth.npy for probe comparison
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

# ---- Eval + Probe ----
echo "=========================================="
echo "  Qwen3-VL-8B: Baseline Eval + Probe"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 eval_qwen3vl.py --all

# ---- Push ----
echo "=========================================="
echo "  Pushing Results"
echo "=========================================="
git add results/qwen3vl_baseline/ embeddings/qwen3vl_*
git commit -m "Qwen3-VL-8B baseline: bottleneck 1 test" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
