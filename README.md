# Can RL Improve Spatial Vision in VLMs?

Proof-of-concept: testing whether GRPO can teach a vision-language model to
visually estimate hole diameters from technical drawings using a scale bar.

## Research question

SFT teaches VLMs to imitate reasoning templates but cannot demonstrate the
perceptual process of converting pixel distances to mm via a scale bar.
Can RL (specifically GRPO), with spatial accuracy as the reward signal,
induce genuine spatial measurement capability?

## Task

- **Input**: Image of a single hole on a plate with a scale bar
- **Output**: Diameter in mm (answer-only, no chain-of-thought)
- **Correct strategy**: `diameter_mm = hole_pixels × (scale_bar_mm / scale_bar_pixels)`

## Anti-shortcut dataset design

The dataset is deliberately constructed so **no statistical shortcut** can
approximate the correct answer:

| Shortcut | Old dataset | This dataset |
|---|---|---|
| Guess dataset mean | 2.61 mm MAE | 6.63 mm MAE |
| Guess nearest nominal | **0.28 mm MAE** | n/a (continuous) |
| Pixel regression | 1.40 mm MAE | 5.86 mm MAE |
| OCR plate title | 0.00 mm MAE | n/a (no title) |
| Scale bar formula | 0.00 mm MAE | 0.007 mm MAE |

Key design choices:
- Continuous uniform diameters (3–30mm), no nominal clusters
- Variable zoom (pixels_per_mm) independent of diameter
- 8 different scale bar values (5–50mm)
- No text leaks (no plate dimensions, no dimension labels)
- Single hole per image
- Variable canvas sizes

**Matched pair diagnostic**: 40 test pairs where the hole is pixel-identical
but the scale bar differs. If the model gives the same answer for both
images in a pair, it is definitively not using the scale bar.

## Interpretation guide

| MAE | Matched pairs | Interpretation |
|---|---|---|
| >6mm | Same answers | Model is guessing (baseline) |
| ~5.5mm | Same answers | Pixel regression, not spatial reasoning |
| <4mm | Answers diverge | Evidence of scale bar usage |
| <2mm | Strong divergence | Strong spatial reasoning |
| <0.5mm | Correct divergence | Near-perfect scale bar reasoning |

## Setup

```bash
# On A100 80GB (required for FP8 config)
bash setup.sh
```

This installs dependencies, generates the dataset (1000 train + 200 test +
40 matched pairs), and runs shortcut verification.

## Usage

```bash
# 1. Baseline evaluation (before training)
python3 evaluate.py --baseline

# 2. Test reward function (no GPU needed)
python3 train_grpo.py --test-reward

# 3. GRPO training
python3 train_grpo.py

# 4. Resume if interrupted
python3 train_grpo.py --resume

# 5. Post-training evaluation
python3 evaluate.py --checkpoint final

# 6. Compare results
python3 evaluate.py --compare
```

## Infrastructure

- **Model**: Ministral 3 8B Instruct (FP8 weights, auto-dequant to bf16)
- **Training**: TRL GRPOTrainer, vLLM colocate mode, LoRA rank 64
- **GPU**: A100 80GB (compute cap 8.0 required for FineGrainedFP8Config)
- **DO NOT** use Blackwell or RTX PRO GPUs

## Reward function

Continuous relative error: `reward = -abs(predicted - ground_truth) / ground_truth`

- Perfect answer: 0.0
- Off by 50%: -0.5
- Off by 100%: -1.0
- Unparseable / negative: -5.0 (clipped floor)

Continuous (not stepwise) to preserve full ranking information for GRPO's
group-relative advantage computation.

## Time budget

Designed for a 6-hour A100 80GB session:
- Setup + dataset: 15 min
- Baseline eval: 15 min
- GRPO training: ~4 hrs (1 epoch, N=4 generations)
- Post-training eval: 15 min
- Push results: 5 min

## Known risks and TODO

- [ ] TRL GRPOTrainer + Mistral VLM + vLLM colocate is untested — may need
  debugging on first run
- [ ] Image handling in TRL's dataset pipeline for multimodal may need
  adaptation (processor-specific chat template)
- [ ] The prompt format (how images are passed in messages) may differ
  between Mistral and what TRL expects
- [ ] Run small test (10 samples, 1 epoch) before full training
- [ ] Verify model can parse task at all before GRPO (baseline eval first)

## File structure

```
generate_dataset.py   # Shortcut-proof dataset generator
train_grpo.py         # GRPO training with TRL
evaluate.py           # Baseline + post-training evaluation
setup.sh              # One-shot setup script
dataset/              # Generated images + metadata (after setup.sh)
checkpoints/          # Training checkpoints (after training)
results/              # Evaluation results (after evaluate.py)
```
