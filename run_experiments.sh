#!/bin/bash
set -e

# Full experiment: SFT baseline + CoT GRPO
# Budget: 6 hours on A100 80GB
#
# Run inside tmux:
#   tmux new -s exp2
#   bash run_experiments.sh

echo "=== Experiment Session 2 ==="
echo "Start time: $(date)"
echo ""

# ---- 1. SFT Baseline (expect ~30 min) ----
echo "=========================================="
echo "  Phase 1: SFT Baseline Training"
echo "=========================================="
python3 train_sft.py
echo ""

echo "  Phase 1b: SFT Evaluation"
python3 evaluate.py --checkpoint-dir checkpoints_sft/final --tag sft
echo ""

# Save SFT results immediately
# Save SFT results immediately
git add results/
git add -f checkpoints_sft/training_log.jsonl 2>/dev/null || true
git commit -m "SFT baseline results" 2>/dev/null || true
git push 2>/dev/null || true
echo "SFT results pushed to GitHub"
echo ""

# ---- 2. CoT GRPO (expect ~3.5 hrs) ----
echo "=========================================="
echo "  Phase 2: CoT GRPO Training"
echo "=========================================="
python3 train_grpo_cot.py
echo ""

echo "  Phase 2b: CoT GRPO Evaluation"
python3 evaluate.py --checkpoint-dir checkpoints_cot/final --tag grpo_cot
echo ""

# ---- 3. Final comparison ----
echo "=========================================="
echo "  Final Comparison"
echo "=========================================="
python3 evaluate.py --compare
echo ""

# ---- 4. Push everything ----
git add results/
git add -f checkpoints_sft/training_log.jsonl checkpoints_cot/training_log.jsonl 2>/dev/null || true
git commit -m "SFT + CoT GRPO experiment results" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== All experiments complete ==="
echo "End time: $(date)"
