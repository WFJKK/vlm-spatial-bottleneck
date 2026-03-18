# Can RL Teach Vision-Language Models to See Spatially?

**TL;DR:** VLMs don't fail at spatial measurement because they can't see — their vision encoders already encode spatial information at R²=0.94. They fail because the language model can't decode it into correct numbers. RL teaches consistent spatial reasoning, SFT teaches precise output, combining them (SFT→RL) gets the best result. Chain-of-thought reasoning is unfaithful — the model writes plausible spatial descriptions disconnected from its actual visual processing.

## The Core Finding

We probed the vision encoder's patch embeddings across four training conditions:

| Model | Probe MAE (mm) | Probe R² | Model Output MAE (mm) |
|---|---|---|---|
| Baseline (no training) | 4.31 | 0.942 | 7.72 |
| SFT | 4.37 | 0.922 | 3.08 |
| GRPO answer-only | 4.46 | 0.940 | 3.53 |
| SFT → RL | 4.39 | 0.925 | 2.29 |

**The vision encoder is identical across all conditions.** Training changes the language model, not the vision encoder. The spatial information is already there (R²=0.94) — the model just doesn't use it until trained.

**Nonlinear probes can't close the gap.** MLPs up to 512×256 neurons on mean-pooled embeddings get 4.9mm — worse than the linear probe's 4.31mm (overfitting). The 4.31mm→2.29mm gap requires cross-patch attention: the language model comparing scale bar patches to hole patches. No pointwise probe can replicate this.

This directly answers MeasureBench's open question: "should we pursue better model architectures and visual encoding schemes?" Answer: not necessarily — the current architecture already encodes the spatial information. The gap is in how the language model decodes it.

## Five Findings

### 1. The vision encoder already sees spatially.
A linear probe on patch embeddings from the *untrained* model predicts diameter at 4.31mm MAE. The model's actual output is 7.72mm. The vision encoder encodes spatial information better than the language model can use it.

### 2. RL discovers spatial reasoning that nobody programmed.
GRPO, with only a "your number was wrong" reward, taught the model to relate hole size to scale bar size. Matched pair diagnostic proves this causally: pixel-identical holes with different scale bars → model gives different answers (r=0.75). No explicit supervision on how to use the scale bar.

### 3. SFT and RL learn different things.
SFT outputs precise decimals (17.64mm) but ignores the scale bar 21% of the time. GRPO always uses the scale bar but rounds to multiples of 5 (20mm). SFT learns output calibration, RL learns spatial consistency. Only the matched pair diagnostic reveals this — MAE alone cannot.

### 4. Chain-of-thought reasoning is unfaithful.
CoT GRPO (5.42mm) performed worse than answer-only GRPO (3.53mm). The model reads the scale bar label correctly 100% of the time and writes reasoning like "the circle appears to be approximately half the scale bar." But "half" is a template — when the true proportion is 1/6, it still writes "half." The verbal description of spatial proportions is unreliable regardless of training method (SFT or RL).

### 5. SFT→RL is complementary.
The standard pipeline (SFT first, then GRPO) achieves the best result (2.29mm) by combining SFT's decimal precision with GRPO's spatial consistency. GRPO alone suffers mode collapse toward round numbers because the base model's token distribution strongly favors them. SFT breaks this prior before RL refines the spatial strategy.

## Results

| Method | MAE (mm) | Scale Bar Correlation | Same Answer % | Compute |
|---|---|---|---|---|
| Baseline | 7.72 | -0.31 (ignores) | — | 0 |
| CoT GRPO | 5.42 | -0.06 (ignores) | 0% | High |
| GRPO answer-only | 3.53 | 0.75 (uses) | 0% | High |
| SFT | 3.08 | 0.58 (uses) | 21% | Low |
| **SFT → RL** | **2.29** | **0.65 (uses)** | **5%** | **High** |

**Pixel regression floor: 5.86mm** — anything below this requires using the scale bar.
**Linear probe ceiling: 4.31mm** — what a linear readout of the vision encoder achieves.
**Perfect: 0.007mm** — using the scale bar formula correctly.

## Methodology

### Shortcut-Proof Dataset
Most spatial benchmarks have exploitable shortcuts. We verified every possible one:

```
Correlation with ground truth diameter:
  hole_pixels alone                    r = +0.44  (not enough)
  scale_bar_mm alone                   r = -0.00  (useless)
  CORRECT: hole_px × sb_mm / sb_px    r = +1.00  (only path)
```

Continuous uniform diameters (3–30mm), variable zoom independent of diameter, 8 scale bar values (5–50mm), no text leaks.

### Matched Pair Diagnostic
40 image pairs where the hole is pixel-identical but the scale bar differs. If the model gives the same answer → ignoring the scale bar. If answers diverge correctly → causally using the scale bar. General-purpose method for testing whether a VLM uses any specific visual feature.

### Linear Probing
Extract mean-pooled patch embeddings from the vision encoder's merger layer (boundary between vision and language). Train Ridge regression to predict diameter. Compare across training conditions to determine if the vision encoder changed or only the language model.

## Connection to Prior Work

**MeasureBench (Oct 2025)** applied GRPO to gauge reading, found RL helps on synthetic but not real images, asked whether architectural changes are needed. Our probing shows the architecture is sufficient — spatial information is already encoded. The bottleneck is the language model.

**SpatialVLM (CVPR 2024)** argued spatial limitations come from datasets, not architecture. Partially confirmed: training dramatically improves performance without architectural changes. But the CoT failure shows a limit — training can teach the model to *use* spatial features but not to *verbalize* them faithfully.

**DeepSeek-R1** showed RL produces emergent text reasoning. We find this partially extends to vision: RL produces emergent scale bar usage (answer-only) but NOT emergent faithful spatial reasoning (CoT). Visual proportional reasoning cannot be reliably verbalized.

**VLAA-Thinker (2025)** found SFT before GRPO degrades performance on general reasoning. Our spatial task shows the opposite — SFT→RL outperforms either alone — because they solve complementary subproblems (precision vs consistency).

## Architecture

```
Task:    Image of hole + scale bar  →  diameter in mm
Model:   Qwen2.5-VL-7B-Instruct + LoRA (rank 64, all-linear)
Reward:  -|predicted - ground_truth| / ground_truth
Method:  Custom GRPO loop (4 generations, per-completion backward)
Compute: A100/A800 80GB, ~$5 per experiment
```

## Reproducing

```bash
git clone https://github.com/WFJKK/Multimodal-GRPO.git
cd Multimodal-GRPO
bash setup.sh                         # Generate dataset
python3 train_sft.py                  # SFT (~30 min)
python3 train_grpo_custom.py          # GRPO answer-only (~3 hrs)
python3 train_grpo_from_sft.py        # SFT→RL (~2.5 hrs)
python3 evaluate.py --compare         # Results table
python3 probe_embeddings.py --all     # Vision encoder probe
```

Pretrained adapters available on HuggingFace:
- [WFJKK/spatial-rl-sft](https://huggingface.co/WFJKK/spatial-rl-sft)
- [WFJKK/spatial-rl-grpo-answer](https://huggingface.co/WFJKK/spatial-rl-grpo-answer)
- [WFJKK/spatial-rl-sft-then-rl](https://huggingface.co/WFJKK/spatial-rl-sft-then-rl)

## Limitations

- **No KL penalty** in GRPO — causes mode collapse toward round numbers.
- **Synthetic only** — no real-world transfer test.
- **Single task** — hole diameter with scale bar. Methodology generalizes but demonstrated once.
- **Compute not matched** — GRPO uses ~10x more compute than SFT per step.
- **LoRA on all-linear** — includes vision encoder layers, but probing shows they didn't change meaningfully. A controlled experiment with vision-frozen LoRA would confirm this.

## Repository Structure

```
generate_dataset.py            # Shortcut-proof dataset generator
train_sft.py                   # Supervised fine-tuning
train_grpo_custom.py           # Answer-only GRPO
train_grpo_cot.py              # Chain-of-thought GRPO
train_grpo_from_sft.py         # SFT→RL pipeline
train_grpo_frozen_vision.py    # GRPO with frozen vision encoder
evaluate.py                    # Evaluation + matched pair diagnostic
probe_embeddings.py            # Vision encoder probing
analyze_attention.py           # Patch-level attention analysis
results/                       # All experimental results
embeddings/                    # Extracted embeddings + probe results
```

## Citation

```
@misc{kames2026spatial,
  title={Can RL Teach Vision-Language Models to See Spatially?},
  author={Kames, Joshua},
  year={2026},
  url={https://github.com/WFJKK/Multimodal-GRPO}
}
```
