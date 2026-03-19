#!/bin/bash
set -e

# Session: Frozen vision GRPO + attention analysis
# Budget: 6 hours on A100 80GB
#
# Time estimates:
#   Setup + dataset:              15 min
#   Download adapters from HF:    5 min
#   Frozen vision GRPO:           3 hrs
#   Frozen vision eval:           15 min
#   Attention analysis:           20 min
#   Push results:                 5 min
#   Total:                        ~4 hrs

echo "=== Frozen Vision + Attention Session ==="
echo "Start time: $(date)"
echo ""

# ---- Conda fix ----
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda deactivate 2>/dev/null || true
fi
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: Cannot import torch"
    exit 1
}

# ---- GPU check ----
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "GPU memory in use: ${GPU_MEM}MiB"
if [ "$GPU_MEM" -gt 5000 ]; then
    echo "ERROR: GPU has stale memory. Restart instance."
    exit 1
fi

# ---- Deps ----
pip install --break-system-packages -q qwen-vl-utils peft scikit-learn huggingface_hub 2>/dev/null || true

# ---- HuggingFace login ----
huggingface-cli login --token $HF_TOKEN

# ---- Dataset ----
if [ ! -d "dataset/train" ]; then
    echo "Generating dataset..."
    python3 generate_dataset.py --n-train 1000 --n-test 200 --seed 42 --output-dir dataset
fi

# ---- Download existing adapters from HuggingFace ----
echo "=========================================="
echo "  Downloading adapters from HuggingFace"
echo "=========================================="
python3 -c "
from huggingface_hub import snapshot_download
import os

adapters = {
    'checkpoints_sft/final': 'WFJKK/spatial-rl-sft',
    'checkpoints/final': 'WFJKK/spatial-rl-grpo-answer',
    'checkpoints_sft_rl/final': 'WFJKK/spatial-rl-sft-then-rl',
}

for local_path, repo_id in adapters.items():
    if os.path.exists(local_path):
        print(f'  {local_path} exists, skipping')
        continue
    try:
        os.makedirs(local_path, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local_path)
        print(f'  Downloaded {repo_id} -> {local_path}')
    except Exception as e:
        print(f'  Failed to download {repo_id}: {e}')
"

# ---- Phase 1: Frozen Vision GRPO ----
if [ ! -d "checkpoints_frozen_vision/final" ]; then
    echo "=========================================="
    echo "  Phase 1: Frozen Vision GRPO (~3 hrs)"
    echo "=========================================="
    python3 train_grpo_frozen_vision.py

    # Upload to HuggingFace
    echo "  Uploading adapter..."
    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('WFJKK/spatial-rl-grpo-frozen-vision', exist_ok=True)
api.upload_folder(folder_path='checkpoints_frozen_vision/final', repo_id='WFJKK/spatial-rl-grpo-frozen-vision')
print('  Uploaded successfully')
" 2>/dev/null || echo "  Upload failed (non-fatal)"
else
    echo "Frozen vision checkpoint exists, skipping"
fi

# ---- Phase 2: Evaluate frozen vision ----
echo "=========================================="
echo "  Phase 2: Evaluate Frozen Vision"
echo "=========================================="
python3 evaluate.py --checkpoint-dir checkpoints_frozen_vision/final --tag grpo_frozen_vision

# Also re-evaluate other models if results don't exist
if [ ! -d "results/baseline" ]; then
    python3 evaluate.py --baseline
fi
if [ ! -d "results/grpo_answer" ]; then
    python3 evaluate.py --checkpoint-dir checkpoints/final --tag grpo_answer
fi
if [ ! -d "results/sft_then_rl" ]; then
    python3 evaluate.py --checkpoint-dir checkpoints_sft_rl/final --tag sft_then_rl --sft-base checkpoints_sft/final
fi

python3 evaluate.py --compare

# ---- Phase 3: Attention Analysis ----
echo "=========================================="
echo "  Phase 3: Attention Analysis"
echo "=========================================="
python3 analyze_attention.py --all

# ---- Push everything ----
echo "=========================================="
echo "  Pushing Results"
echo "=========================================="
git add results/ attention_maps/
git add -f checkpoints_frozen_vision/training_log.jsonl 2>/dev/null || true
git commit -m "frozen vision GRPO + attention analysis" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
