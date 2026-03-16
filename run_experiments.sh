#!/bin/bash
set -e

# Session: SFT→RL + CoT GRPO
# Budget: 6 hours on A100 80GB
#
# Time estimates:
#   Setup + dataset:     15 min
#   SFT training:        30 min (needed for SFT→RL)
#   SFT→RL training:    2.5 hrs (answer-only, 1000 steps)
#   SFT→RL eval:         15 min
#   CoT GRPO training:  ~2.5 hrs (will run until time pressure)
#   CoT eval:            15 min
#   Total:              ~6 hrs

echo "=== Experiment Session ==="
echo "Start time: $(date)"
echo ""

# ---- Deactivate conda env if active (vast.ai auto-activates one) ----
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Deactivating conda env: $CONDA_DEFAULT_ENV"
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda deactivate 2>/dev/null || true
fi

# Verify we can import torch
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: Cannot import torch. If in conda env, run: deactivate"
    exit 1
}

# ---- Check GPU is clean ----
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "GPU memory in use: ${GPU_MEM}MiB"
if [ "$GPU_MEM" -gt 5000 ]; then
    echo "ERROR: GPU has ${GPU_MEM}MiB in use. Stale process?"
    echo "Run: fuser -k /dev/nvidia0  OR restart instance"
    exit 1
fi
echo ""

# ---- Deps ----
pip install --break-system-packages -q qwen-vl-utils peft 2>/dev/null || true

# ---- Dataset ----
if [ ! -d "dataset/train" ]; then
    echo "Generating dataset..."
    python3 generate_dataset.py --n-train 1000 --n-test 200 --seed 42 --output-dir dataset
fi

# ---- Phase 1: SFT (needed as base for SFT→RL) ----
if [ ! -d "checkpoints_sft/final" ]; then
    echo "=========================================="
    echo "  Phase 1: SFT Training (~30 min)"
    echo "=========================================="
    python3 train_sft.py
    echo ""
else
    echo "SFT checkpoint already exists, skipping"
fi

# ---- Phase 2: SFT→RL (GRPO from SFT checkpoint) ----
echo "=========================================="
echo "  Phase 2: SFT→RL Training (~2.5 hrs)"
echo "=========================================="
python3 train_grpo_from_sft.py
echo ""

echo "  Phase 2b: SFT→RL Evaluation"
python3 evaluate.py --checkpoint-dir checkpoints_sft_rl/final --tag sft_then_rl --sft-base checkpoints_sft/final
echo ""

# Push SFT→RL results immediately
git add results/
git add -f checkpoints_sft_rl/training_log.jsonl 2>/dev/null || true
git commit -m "SFT->RL results" 2>/dev/null || true
git push 2>/dev/null || true
echo "SFT→RL results pushed"
echo ""

# ---- Phase 3: CoT GRPO ----
echo "=========================================="
echo "  Phase 3: CoT GRPO Training"
echo "  (will run until stopped or complete)"
echo "=========================================="
# Clean stale checkpoints from previous 4-bit run (invalid)
if [ -d "checkpoints_cot" ]; then
    echo "  Removing stale CoT checkpoints from previous run"
    rm -rf checkpoints_cot
fi
python3 train_grpo_cot.py
echo ""

echo "  Phase 3b: CoT GRPO Evaluation"
if [ -d "checkpoints_cot/final" ]; then
    python3 evaluate.py --checkpoint-dir checkpoints_cot/final --tag grpo_cot
else
    LATEST=$(ls -d checkpoints_cot/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST" ]; then
        echo "Using latest checkpoint: $LATEST"
        python3 evaluate.py --checkpoint-dir "$LATEST" --tag grpo_cot_partial
    fi
fi
echo ""

# ---- Final comparison ----
echo "=========================================="
echo "  Final Comparison"
echo "=========================================="
python3 evaluate.py --compare
echo ""

# ---- Push everything ----
git add results/
git add -f checkpoints_sft_rl/training_log.jsonl checkpoints_cot/training_log.jsonl 2>/dev/null || true
git commit -m "all experiment results" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
