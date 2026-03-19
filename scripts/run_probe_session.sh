#!/bin/bash
set -e

# Session: Retrain all models + probe embeddings
# Budget: 6 hours on A100 80GB
#
# Time estimates:
#   Setup + dataset:        15 min
#   SFT training:           30 min
#   GRPO answer-only:       3 hrs
#   SFT→RL:                 30 min (only RL stage, SFT reused)
#   Extract embeddings:     20 min (4 models × 200 images)
#   Train probes:           1 min
#   Eval all models:        30 min
#   Total:                  ~5 hrs

echo "=== Probing Session ==="
echo "Start time: $(date)"
echo ""

# ---- Conda fix for vast.ai ----
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

# ---- Helper: upload adapter to HuggingFace ----
upload_adapter() {
    local CKPT_DIR=$1
    local REPO_NAME=$2
    echo "  Uploading $CKPT_DIR to WFJKK/$REPO_NAME..."
    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('WFJKK/$REPO_NAME', exist_ok=True)
api.upload_folder(folder_path='$CKPT_DIR', repo_id='WFJKK/$REPO_NAME')
print('  Uploaded successfully')
" 2>/dev/null || echo "  Upload failed (non-fatal)"
}

# ---- Phase 1: SFT ----
if [ ! -d "checkpoints_sft/final" ]; then
    echo "=========================================="
    echo "  Phase 1: SFT Training (~30 min)"
    echo "=========================================="
    python3 train_sft.py
    upload_adapter "checkpoints_sft/final" "spatial-rl-sft"
else
    echo "SFT checkpoint exists, skipping"
fi

# ---- Phase 2: GRPO answer-only ----
if [ ! -d "checkpoints/final" ]; then
    echo "=========================================="
    echo "  Phase 2: GRPO Answer-Only Training (~3 hrs)"
    echo "=========================================="
    python3 train_grpo_custom.py
    upload_adapter "checkpoints/final" "spatial-rl-grpo-answer"
else
    echo "GRPO answer-only checkpoint exists, skipping"
fi

# ---- Phase 3: SFT→RL ----
if [ ! -d "checkpoints_sft_rl/final" ]; then
    echo "=========================================="
    echo "  Phase 3: SFT→RL Training (~30 min)"
    echo "=========================================="
    python3 train_grpo_from_sft.py
    upload_adapter "checkpoints_sft_rl/final" "spatial-rl-sft-then-rl"
else
    echo "SFT→RL checkpoint exists, skipping"
fi

# ---- Phase 4: Evaluate all models (if not done) ----
echo "=========================================="
echo "  Phase 4: Evaluations"
echo "=========================================="
if [ ! -d "results/baseline" ]; then
    python3 evaluate.py --baseline
fi
if [ ! -d "results/sft" ]; then
    python3 evaluate.py --checkpoint-dir checkpoints_sft/final --tag sft
fi
if [ ! -d "results/grpo_answer" ]; then
    python3 evaluate.py --checkpoint-dir checkpoints/final --tag grpo_answer
fi
if [ ! -d "results/sft_then_rl" ]; then
    python3 evaluate.py --checkpoint-dir checkpoints_sft_rl/final --tag sft_then_rl --sft-base checkpoints_sft/final
fi

# ---- Phase 5: Extract embeddings + probe ----
echo "=========================================="
echo "  Phase 5: Embedding Probe"
echo "=========================================="
python3 probe_embeddings.py --all

# ---- Push everything ----
echo "=========================================="
echo "  Pushing Results"
echo "=========================================="
git add results/ embeddings/
git add -f checkpoints*/training_log.jsonl 2>/dev/null || true
git commit -m "probe results + all evaluations" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
echo "Adapters saved to:"
echo "  https://huggingface.co/WFJKK/spatial-rl-sft"
echo "  https://huggingface.co/WFJKK/spatial-rl-grpo-answer"
echo "  https://huggingface.co/WFJKK/spatial-rl-sft-then-rl"
