#!/bin/bash
set -e

# INSTANCE A: Frozen vision GRPO + attention analysis
# ~4 hours on A100 80GB

echo "=== Instance A: Frozen Vision + Attention ==="
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

# ---- Download existing adapters for eval + attention ----
echo "Downloading adapters..."
python3 -c "
from huggingface_hub import snapshot_download
import os
for local, repo in [
    ('checkpoints_sft/final', 'WFJKK/spatial-rl-sft'),
    ('checkpoints/final', 'WFJKK/spatial-rl-grpo-answer'),
    ('checkpoints_sft_rl/final', 'WFJKK/spatial-rl-sft-then-rl'),
]:
    if not os.path.exists(local):
        os.makedirs(local, exist_ok=True)
        try:
            snapshot_download(repo_id=repo, local_dir=local)
            print(f'  Downloaded {repo}')
        except Exception as e:
            print(f'  Failed: {e}')
    else:
        print(f'  {local} exists')
"

# ---- Frozen Vision GRPO (~3 hrs) ----
echo "=========================================="
echo "  Frozen Vision GRPO Training"
echo "=========================================="
python3 train_grpo_frozen_vision.py

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('WFJKK/spatial-rl-grpo-frozen-vision', exist_ok=True)
api.upload_folder(folder_path='checkpoints_frozen_vision/final', repo_id='WFJKK/spatial-rl-grpo-frozen-vision')
print('Uploaded')
" 2>/dev/null || echo "Upload failed (non-fatal)"

# ---- Evaluate frozen vision ----
echo "=========================================="
echo "  Evaluations"
echo "=========================================="
python3 evaluate.py --checkpoint-dir checkpoints_frozen_vision/final --tag grpo_frozen_vision
python3 evaluate.py --baseline
python3 evaluate.py --checkpoint-dir checkpoints/final --tag grpo_answer
python3 evaluate.py --checkpoint-dir checkpoints_sft/final --tag sft
python3 evaluate.py --checkpoint-dir checkpoints_sft_rl/final --tag sft_then_rl --sft-base checkpoints_sft/final
python3 evaluate.py --compare

# ---- Attention Analysis ----
echo "=========================================="
echo "  Attention Analysis"
echo "=========================================="
python3 analyze_attention.py --all

# ---- Push ----
git add results/ attention_maps/
git add -f checkpoints_frozen_vision/training_log.jsonl 2>/dev/null || true
git commit -m "Instance A: frozen vision GRPO + attention" 2>/dev/null || true
git push 2>/dev/null || true

echo "=== Instance A Complete ==="
echo "End time: $(date)"
