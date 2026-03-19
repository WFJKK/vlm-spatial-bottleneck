#!/bin/bash
set -e

# INSTANCE B: Multi-epoch SFT + GRPO with KL penalty
# ~4.5 hours on A100 80GB

echo "=== Instance B: Multi-epoch SFT + KL GRPO ==="
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

pip install --break-system-packages -q qwen-vl-utils peft scikit-learn huggingface_hub 2>/dev/null || true
huggingface-cli login --token $HF_TOKEN

if [ ! -d "dataset/train" ]; then
    python3 generate_dataset.py --n-train 1000 --n-test 200 --seed 42 --output-dir dataset
fi

# ---- Multi-epoch SFT (~90 min) ----
echo "=========================================="
echo "  3-Epoch SFT Training"
echo "=========================================="
python3 train_sft_3epoch.py

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('WFJKK/spatial-rl-sft-3epoch', exist_ok=True)
api.upload_folder(folder_path='checkpoints_sft_3epoch/final', repo_id='WFJKK/spatial-rl-sft-3epoch')
print('Uploaded')
" 2>/dev/null || echo "Upload failed (non-fatal)"

# ---- Evaluate 3-epoch SFT ----
python3 evaluate.py --checkpoint-dir checkpoints_sft_3epoch/final --tag sft_3epoch

# Push intermediate results
git add results/
git add -f checkpoints_sft_3epoch/training_log.jsonl 2>/dev/null || true
git commit -m "Instance B: 3-epoch SFT results" 2>/dev/null || true
git push 2>/dev/null || true

# ---- GRPO with KL Penalty (~3 hrs) ----
echo "=========================================="
echo "  GRPO with KL Penalty Training"
echo "=========================================="
python3 train_grpo_kl.py

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('WFJKK/spatial-rl-grpo-kl', exist_ok=True)
api.upload_folder(folder_path='checkpoints_kl/final', repo_id='WFJKK/spatial-rl-grpo-kl')
print('Uploaded')
" 2>/dev/null || echo "Upload failed (non-fatal)"

# ---- Evaluate KL GRPO ----
python3 evaluate.py --checkpoint-dir checkpoints_kl/final --tag grpo_kl

# ---- Full comparison ----
echo "=========================================="
echo "  Comparison (what this instance has)"
echo "=========================================="
python3 evaluate.py --compare

# ---- Push everything ----
git add results/
git add -f checkpoints_kl/training_log.jsonl 2>/dev/null || true
git commit -m "Instance B: KL GRPO + 3-epoch SFT results" 2>/dev/null || true
git push 2>/dev/null || true

echo "=== Instance B Complete ==="
echo "End time: $(date)"
