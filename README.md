# Where VLMs Fail: The Spatial Decoding Bottleneck

VLMs can see spatially -- they just can't say what they see. We prove this across three model architectures, eight training methods, and 2,400 controlled evaluations.

## The Problem

Vision-language models fail at spatial measurement: reading gauges, estimating distances, extracting dimensions from technical drawings. MeasureBench (2025) found the best model achieves only 30% accuracy on instrument reading. The field assumes this is a *vision encoding* problem -- that patch-based ViTs can't encode spatial information precisely enough.

We show it's a *decoding* problem. The vision encoder already works. The language model can't use what it receives.

## Evidence

**Three models, same bottleneck.**

| Model | Architecture | Vision Probe | Output | Gap |
|---|---|---|---|---|
| Qwen2.5-VL-7B | ViT → Merger → LM | 4.31 mm | 7.72 mm | 3.41 mm |
| Qwen3-VL-8B | ViT → DeepStack → LM | 4.22 mm | 9.00 mm | 4.78 mm |
| Qwen3.5-9B | Early fusion | 4.55 mm | 8.62 mm | 4.07 mm |

*Vision probe:* linear regression on mean-pooled patch embeddings predicts diameter at ~4.3 mm MAE. The vision encoder encodes spatial information at R² ≥ 0.94 across all three models.

*Output:* the language model produces ~8 mm MAE -- worse than the probe on its own inputs. DeepStack and early fusion training don't help. The gap persists across architectural generations.

**Training closes the gap -- dataset design matters more than method.**

| Method | MAE (mm) | Matched Pair r | Compute |
|---|---|---|---|
| Baseline (no training) | 7.72 | −0.31 | -- |
| CoT GRPO | 5.42 | −0.06 | High |
| GRPO answer-only | 3.53 | 0.75 | High |
| GRPO + KL penalty | 2.75 | -- | High |
| SFT (1 epoch) | 3.08 | 0.58 | Low |
| SFT → RL | 2.29 | 0.65 | High |
| **SFT (3 epochs)** | **1.62** | **0.85** | **Low** |
| **Regression head (SFT-3epoch)** | **1.33** | **0.87** | **Low** |

3-epoch SFT on a shortcut-proof dataset outperforms all RL variants at a fraction of the compute. On a clean dataset, SFT is forced into genuine spatial reasoning. RL's perceived advantage is a dataset artifact: RL helps when SFT targets are poor, not when the evaluation is rigorous.

**The tokenization bottleneck is real and measurable.**

Replacing text decoding with a regression head on the SFT-3epoch model's last hidden state reduces MAE from 1.62 mm to 1.33 mm (18% improvement). Same model, same forward pass, same cross-patch attention. The only difference is the final step: a continuous regression head instead of text token generation. The model's internal representation encodes spatial information more precisely than it can express in tokens.

**Chain-of-thought is unfaithful for spatial perception.**

CoT GRPO (5.42 mm) performed *worse* than answer-only GRPO (3.53 mm). The model reads scale bar labels correctly 100% of the time but writes "approximately half the scale bar" regardless of the true proportion. Verbal descriptions of spatial relationships are unreliable templates, not faithful reports of visual processing. This aligns with MeasureBench's finding that "thinking" doesn't improve measurement, and extends it: CoT fails because spatial proportions can't be faithfully discretized into words.

**Attention patterns don't change after training.**

Baseline vs. GRPO attention correlation: r = 0.885. The model looks at the same patches before and after training. All improvement comes from downstream computation -- better arithmetic on the same attended features.

## Regression Head Experiment

The regression head isolates the tokenization bottleneck. The architecture is identical to how reward models work: run the full VLM forward pass (vision encoder → merger → LM with cross-patch attention), then replace the autoregressive text decoder with a small MLP (3584 → 256 → 64 → 1) that reads the LM's last hidden state and outputs a continuous scalar.

The LM still does the spatial reasoning: it processes the instruction, identifies the relevant drawing elements, and computes cross-patch attention between scale bar tokens and hole tokens. The regression head reads out the answer without the precision loss of number-to-token conversion.

| Variant | MAE (mm) | Matched Pair r | What it tests |
|---|---|---|---|
| SFT-3epoch → text tokens | 1.62 | 0.85 | Text decoding baseline |
| SFT-3epoch → regression head | 1.33 | 0.87 | Tokenization bottleneck |

The 0.29 mm gap between identical models with different output heads confirms that text tokenization is a measurable precision bottleneck for spatial outputs. The regression head (934K parameters, trained frozen on the 7B model) recovers precision that exists in the LM's internal state but is lost during text generation.

## How We Know It's Not Shortcuts

Most spatial benchmarks are exploitable. We verified every possible shortcut:

```
Correlation with ground truth diameter:
  hole_pixels alone                  r = +0.44  (insufficient)
  scale_bar_mm alone                 r = −0.00  (useless)
  CORRECT: hole_px × sb_mm / sb_px  r = +1.00  (only valid path)
```

**Matched pair diagnostic:** 40 image pairs with pixel-identical holes but different scale bars. If the model gives the same answer for both → not using the scale bar. If answers diverge correctly → causally verified scale bar usage. This method generalizes to any visual feature: swap it, check if the output changes.

At 1 epoch, SFT and GRPO had similar MAE but fundamentally different strategies -- only the matched pair diagnostic revealed this. MAE alone is insufficient for evaluating spatial reasoning.

## Layer-by-Layer Probe (Qwen3.5 Early Fusion)

Spatial information at image-token positions across 32 transformer layers (untrained):

| Layer | Probe MAE | R² |
|---|---|---|
| 0 | 5.42 mm | 0.475 |
| 8 | 4.84 mm | 0.729 |
| 16 | 4.95 mm | 0.869 |
| 28 | 4.55 mm | 0.998 |
| 32 | 4.57 mm | 1.000 |

MAE barely improves (5.42 → 4.55 mm) but R² rises from 0.475 to 1.000. The transformer layers reorganize spatial information into linearly separable representations without extracting new spatial content. The precision ceiling is set at the patch embedding stage.

## Nonlinear Probes Can't Close the Gap

MLPs up to 512×256 neurons on mean-pooled embeddings: 4.91 mm. Worse than the 4.31 mm linear probe (overfitting). The 4.31 → 1.62 mm gap requires cross-patch attention -- comparing scale bar tokens to hole tokens -- that no pointwise probe can replicate. The trained language model achieves 1.62 mm because it uses attention to compare specific patches, not because it applies a more complex function to the average embedding.

This also explains why the regression head must be placed *after* the LM, not on the vision encoder directly. The LM's cross-patch attention is essential for computing the spatial relationship. The regression head simply reads out the result without tokenization loss.

## Implications

**For VLM builders:** The vision encoder is not the bottleneck for spatial reasoning. Qwen3-VL's DeepStack improved MMMU scores but made spatial measurement *worse*. Architectural improvements targeting general multimodal benchmarks don't transfer to fine-grained spatial perception. The decoding pathway needs targeted training.

**For VLA/robotics:** The tokenization bottleneck directly affects vision-language-action models that predict continuous actions (coordinates, gripper widths) through autoregressive text generation. A regression head on the LM's hidden state recovers spatial precision that would otherwise be lost, suggesting action heads should read from internal representations rather than decoded tokens.

**For benchmark designers:** Matched pair diagnostics and shortcut verification should be standard. We found that even our own compliance benchmark was 98% solvable by text shortcuts. Shortcut elimination is prerequisite to meaningful evaluation.

**For practitioners:** If you need a VLM to extract measurements, spatial distances, or dimensional information from images -- train on shortcut-proof data. The method (SFT vs. RL) matters less than the dataset. Chain-of-thought will not help and may hurt. For maximum precision, use a regression head instead of text decoding.

## Reproducing

```
git clone https://github.com/WFJKK/vlm-spatial-bottleneck.git
cd vlm-spatial-bottleneck
bash setup.sh

python3 train_sft_3epoch.py                # Best text result: 1.62 mm
python3 train_regression_head.py            # Best overall: 1.33 mm (sft3_head_only)
python3 evaluate.py --compare              # Full comparison table
python3 probe_embeddings.py --all          # Vision encoder probing
python3 analyze_attention.py --all         # Attention analysis
python3 eval_qwen3vl.py --all             # Cross-model: Qwen3-VL
python3 eval_qwen35.py --all              # Cross-model: Qwen3.5
```

Pretrained adapters:
[WFJKK/spatial-rl-sft-3epoch](https://huggingface.co/WFJKK/spatial-rl-sft-3epoch) ·
[WFJKK/spatial-rl-grpo-answer](https://huggingface.co/WFJKK/spatial-rl-grpo-answer) ·
[WFJKK/spatial-rl-sft-then-rl](https://huggingface.co/WFJKK/spatial-rl-sft-then-rl) ·
[WFJKK/spatial-rl-sft](https://huggingface.co/WFJKK/spatial-rl-sft) ·
[WFJKK/spatial-rl-grpo-frozen-vision](https://huggingface.co/WFJKK/spatial-rl-grpo-frozen-vision)

**Requirements:** A100/H100 80 GB for training. Inference and probing fit on any GPU with ≥24 GB.

## Repository

```
generate_dataset.py              Shortcut-proof synthetic dataset with verification
evaluate.py                      Evaluation + matched pair diagnostic
probe_embeddings.py              Vision encoder linear probing
analyze_attention.py             Patch-level attention analysis

train_sft.py                     Supervised fine-tuning (1 epoch)
train_sft_3epoch.py              Supervised fine-tuning (3 epochs, best text result)
train_regression_head.py         Regression head on LM hidden states (best overall)
train_grpo_custom.py             GRPO answer-only
train_grpo_kl.py                 GRPO with KL penalty
train_grpo_cot.py                GRPO with chain-of-thought
train_grpo_from_sft.py           SFT → RL pipeline
train_grpo_frozen_vision.py      GRPO with frozen vision encoder

eval_qwen3vl.py                  Qwen3-VL-8B cross-model evaluation + probe
eval_qwen35.py                   Qwen3.5-9B cross-model evaluation + layer probe

run_regression_head.sh           Vast.ai run script for regression head experiment

results/                         All experimental results (JSON)
embeddings/                      Extracted embeddings + probe results
attention_maps/                  Attention analysis data
```

## Prior Work

**MeasureBench** (BAAI, 2025) benchmarked VLMs on instrument reading, found RL "encouraging" on synthetic data but not on real images, asked whether architectural changes are needed. We answer: no -- the architecture already encodes the information. The gap is in decoding.

**SpatialVLM** (Google, CVPR 2024) argued spatial limitations come from datasets. Confirmed: shortcut-free SFT achieves 1.62 mm without architectural changes.

**DeepSeek-R1** (DeepSeek, 2025) showed RL produces emergent text reasoning. We find this partially extends to vision: RL produces emergent scale bar usage but not faithful spatial CoT.

**VLM-RobustBench** (ICLR 2026 Workshop) found VLMs are "semantically strong but spatially fragile." Our probing explains why: spatial information exists in the vision encoder but the language model doesn't decode it.

## Citation

```
@misc{kames2026spatial,
  title   = {Where VLMs Fail: The Spatial Decoding Bottleneck},
  author  = {Kames, Joshua},
  year    = {2026},
  url     = {https://github.com/WFJKK/vlm-spatial-bottleneck}
}
```
