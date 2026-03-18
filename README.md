# Can RL Teach Vision-Language Models to See Spatially?

**TL;DR:** VLMs don't fail at spatial measurement because they can't see — their vision encoders already encode spatial information at R²=0.94. They fail because the language model can't *compute* the right answer from features it already receives. Training method matters less than dataset design: 3-epoch SFT on a shortcut-proof dataset (1.62mm MAE) beats all RL variants. The real contribution is the evaluation methodology — matched pair diagnostics and shortcut verification reveal what accuracy alone cannot.

## The Core Finding

We probed the vision encoder's patch embeddings and extracted attention maps across training conditions:

| Model | Probe MAE | Probe R² | Output MAE | Matched Pair |
|---|---|---|---|---|
| Baseline (no training) | 4.31mm | 0.942 | 7.72mm | Ignores scale bar |
| SFT (1 epoch) | 4.37mm | 0.922 | 3.08mm | Partial (r=0.58) |
| GRPO answer-only | 4.46mm | 0.940 | 3.53mm | Uses (r=0.75) |
| SFT → RL | 4.39mm | 0.925 | 2.29mm | Uses (r=0.65) |
| **SFT (3 epochs)** | — | — | **1.62mm** | **Uses (r=0.85)** |

- **Vision encoder embeddings:** identical across all training conditions
- **Attention patterns:** r=0.885 correlation between baseline and trained models
- **All improvement:** in the language model's downstream computation

The vision encoder already sees spatially. Training teaches the language model to compute correctly from what it already receives and already attends to.

## Complete Results

| Method | MAE (mm) | Matched Pair | Scale Bar | Compute |
|---|---|---|---|---|
| Baseline | 7.72 | r=-0.31 | ✗ Ignores | 0 |
| CoT GRPO (500 steps) | 5.42 | r=-0.06 | ✗ Ignores | High |
| GRPO answer-only | 3.53 | r=0.75 | ✓ Uses | High |
| GRPO frozen vision | 3.24 | r=0.34 | ~ Partial | High |
| SFT (1 epoch) | 3.08 | r=0.58 | ✓ Uses | Low |
| GRPO + KL penalty | 2.75 | — | ✓ Uses | High |
| SFT → RL | 2.29 | r=0.65 | ✓ Uses | High |
| **SFT (3 epochs)** | **1.62** | **r=0.85** | **✓ Uses** | **Low** |

Reference baselines:
- Pixel regression (no scale bar): 5.86mm
- Linear probe on vision embeddings: 4.31mm
- Perfect (scale bar formula): 0.007mm

## Six Findings

### 1. The vision encoder already encodes spatial information.
A linear probe on patch embeddings from the untrained model gets 4.31mm MAE (R²=0.94). The model's actual output is 7.72mm. The vision encoder is better at encoding spatial information than the language model is at decoding it. Nonlinear probes (MLP up to 512×256) cannot improve over the linear probe — the 4.31mm→1.62mm gap requires cross-patch attention that pointwise probes fundamentally cannot replicate.

### 2. Training method matters less than dataset design.
3-epoch SFT beats every RL variant at a fraction of the compute. RL's perceived advantage at 1 epoch was a compute artifact, not a fundamental difference in learning. On a shortcut-proof dataset, SFT is forced into genuine spatial reasoning because no easier path exists. When text shortcuts are available (as in our earlier compliance experiment), SFT takes shortcuts instead. The dataset determines what the model learns, not the training algorithm.

### 3. Chain-of-thought reasoning is unfaithful for spatial perception.
CoT GRPO (5.42mm) performed worse than answer-only GRPO (3.53mm). The model reads scale bar labels correctly 100% of the time and writes plausible reasoning ("the circle appears to be approximately half the scale bar"). But "half" is a template — when the true proportion is 1/6, it still writes "half." Spatial proportions exist as continuous features in embeddings; discretizing them into words loses the precision that makes them useful. The language model computes spatial relationships better than it can describe them.

The reasoning strategy evolves over training but never becomes faithful:

| Training Phase | Strategy |
|---|---|
| Steps 1–50 | "Measure the diameter" (92%), "multiply" (58%) |
| Steps 50–200 | "Compare size" emerges (36%) |
| Steps 400+ | "Compare size" dominates (78%), "estimate" (63%) |

### 4. Attention patterns don't change after training.
Mean attention correlation between baseline and GRPO-trained model: r=0.885. The model looks at the same patches before and after training. The improvement is entirely in what the language model *computes* from the attended features, not in where it looks.

### 5. Vision encoder LoRA helps consistency but not encoding.
Freezing the vision encoder (LoRA only on language model) gets similar MAE (3.24mm vs 3.53mm) but weaker matched pair correlation (0.34 vs 0.75). Linear probes show identical embeddings either way. The vision encoder LoRA doesn't change spatial encoding — it changes patch saliency in a way that helps the language model use the scale bar more consistently.

### 6. GRPO without KL penalty causes mode collapse.
Vanilla GRPO collapses to round numbers (83% multiples of 5). SFT→RL collapses to repeated decimals (e.g., 3.42, 9.42, 24.32). KL penalty (β=0.1) partially mitigates this (2.75mm MAE vs 3.53mm). But 3-epoch SFT avoids the problem entirely — direct supervision on precise targets naturally produces diverse, precise outputs.

## Methodology

### Shortcut-Proof Dataset
Most spatial benchmarks have exploitable shortcuts. Our previous compliance dataset was 98% solvable by guessing one of six nominal diameters. We verified every possible shortcut:

```
Correlation with ground truth diameter:
  hole_pixels alone                    r = +0.44  (not enough)
  scale_bar_mm alone                   r = -0.00  (useless)
  CORRECT: hole_px × sb_mm / sb_px    r = +1.00  (only path)
```

Continuous uniform diameters (3–30mm), variable zoom independent of diameter, 8 scale bar values (5–50mm), no text leaks.

### Matched Pair Diagnostic
40 image pairs where the hole is pixel-identical but the scale bar differs. If the model gives the same answer → ignoring the scale bar. If answers diverge correctly → causally using the scale bar. This reveals what accuracy alone cannot: at 1 epoch, SFT and GRPO had similar MAE but fundamentally different spatial strategies.

### Linear Probing + Attention Analysis
Extract mean-pooled patch embeddings from the vision encoder's merger layer. Train Ridge regression to predict diameter. Compare across training conditions. Extract attention weights from language model layers to compare where trained vs untrained models look. Both analyses converge: the vision encoder and attention are unchanged; all improvement is in downstream computation.

## Connection to Prior Work

**MeasureBench (Oct 2025)** applied GRPO for gauge reading, asked whether architectural changes are needed. Our probing shows: no — the architecture already encodes spatial information. The bottleneck is training the language model to use it, and even SFT suffices with a clean dataset.

**SpatialVLM (CVPR 2024)** argued spatial limitations come from datasets, not architecture. Strongly confirmed: dataset design (eliminating shortcuts) matters more than training method (SFT vs RL).

**DeepSeek-R1** showed RL produces emergent text reasoning. We find this does NOT extend to spatial perception: RL produces emergent scale bar usage but NOT faithful spatial reasoning via CoT. The language model computes spatial relationships better than it can verbalize them.

**VLAA-Thinker (2025)** found SFT before GRPO degrades general reasoning. Our result adds nuance: for spatial tasks with clean data, SFT alone with enough epochs outperforms all RL variants.

## Architecture

```
Task:    Image of hole + scale bar → diameter in mm
Model:   Qwen2.5-VL-7B-Instruct + LoRA (rank 64)
Reward:  -|predicted - ground_truth| / ground_truth
Method:  Custom GRPO loop (4 generations, per-completion backward)
Compute: A100/A800 80GB, ~$5 per experiment
```

## Reproducing

```bash
git clone https://github.com/WFJKK/Multimodal-GRPO.git
cd Multimodal-GRPO
bash setup.sh                              # Generate dataset
python3 train_sft_3epoch.py                # Best method: 3-epoch SFT
python3 evaluate.py --checkpoint-dir checkpoints_sft_3epoch/final --tag sft_3epoch
python3 evaluate.py --compare
python3 probe_embeddings.py --all          # Vision encoder probing
python3 analyze_attention.py --all         # Attention analysis
```

Pretrained adapters on HuggingFace:
- [WFJKK/spatial-rl-sft](https://huggingface.co/WFJKK/spatial-rl-sft) (1-epoch SFT)
- [WFJKK/spatial-rl-sft-3epoch](https://huggingface.co/WFJKK/spatial-rl-sft-3epoch) (3-epoch SFT, best)
- [WFJKK/spatial-rl-grpo-answer](https://huggingface.co/WFJKK/spatial-rl-grpo-answer)
- [WFJKK/spatial-rl-grpo-frozen-vision](https://huggingface.co/WFJKK/spatial-rl-grpo-frozen-vision)
- [WFJKK/spatial-rl-sft-then-rl](https://huggingface.co/WFJKK/spatial-rl-sft-then-rl)

## Limitations

- **Synthetic only** — no real-world transfer test
- **Single task** — hole diameter with scale bar
- **No KL penalty** in most GRPO runs — caused mode collapse
- **Compute not fully matched** — 3-epoch SFT uses ~90 min vs GRPO's ~3 hrs
- **200 test images** — larger test set would reduce variance in matched pair correlation

## Repository Structure

```
generate_dataset.py            # Shortcut-proof dataset generator
train_sft.py                   # 1-epoch SFT
train_sft_3epoch.py            # 3-epoch SFT (best result)
train_grpo_custom.py           # Answer-only GRPO
train_grpo_kl.py               # GRPO with KL penalty
train_grpo_cot.py              # Chain-of-thought GRPO
train_grpo_frozen_vision.py    # GRPO with frozen vision encoder
train_grpo_from_sft.py         # SFT→RL pipeline
evaluate.py                    # Evaluation + matched pair diagnostic
probe_embeddings.py            # Vision encoder probing
analyze_attention.py           # Patch-level attention analysis
results/                       # All experimental results
embeddings/                    # Probe results
attention_maps/                # Attention analysis
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
