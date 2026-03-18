"""Qwen3-VL-8B baseline evaluation + vision encoder probe.

Tests whether the decoding bottleneck (gap between vision encoder probe
and model output) exists in next-generation VLMs.

Qwen2.5-VL-7B had:
  - Vision encoder probe: 4.31mm (R²=0.94)
  - Baseline output: 7.72mm
  - Gap: 3.41mm (bottleneck 1)

If Qwen3-VL-8B shows a similar gap: bottleneck 1 is universal.
If gap is small: DeepStack architecture solved the decoding problem.

Usage:
    python3 eval_qwen3vl.py --eval       # Run baseline eval + matched pairs
    python3 eval_qwen3vl.py --probe      # Extract embeddings + linear probe
    python3 eval_qwen3vl.py --all        # Both
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASET_DIR = "dataset"
RESULTS_DIR = "results/qwen3vl_baseline"
EMBEDDINGS_DIR = "embeddings"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SYSTEM_PROMPT = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)
USER_PROMPT = "What is the diameter of hole H1 in mm?"


def parse_number(text):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.search(r'(\d+\.?\d*)', text)
    return float(match.group(1)) if match else None


def load_model():
    """Load Qwen3-VL-8B for inference."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    return model, processor


def run_inference(model, processor, image_path):
    """Run inference on a single image, return predicted text."""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    # Trim input tokens
    generated = outputs[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    return text


def run_eval():
    """Run baseline evaluation on test set + matched pairs."""
    model, processor = load_model()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test metadata
    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]

    print(f"\n=== Qwen3-VL-8B Baseline Evaluation ===")
    print(f"Test samples: {len(samples)}")

    results = []
    maes = []
    for i, sample in enumerate(samples):
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
        text = run_inference(model, processor, image_path)
        pred = parse_number(text)

        gt = sample["diameter_mm"]
        ae = abs(pred - gt) if pred else None

        results.append({
            "idx": sample["idx"],
            "gt_mm": gt,
            "predicted_mm": pred,
            "raw_output": text,
            "ae_mm": ae,
        })

        if ae is not None:
            maes.append(ae)

        if (i + 1) % 10 == 0:
            current_mae = np.mean(maes) if maes else float("nan")
            print(f"    [{i+1}/{len(samples)}] MAE so far: {current_mae:.2f}mm "
                  f"(parsed: {len(maes)}/{i+1})")

    # Save results
    with open(os.path.join(RESULTS_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Compute metrics
    valid_aes = [r["ae_mm"] for r in results if r["ae_mm"] is not None]
    metrics = {
        "model": MODEL_ID,
        "n_total": len(results),
        "n_parsed": len(valid_aes),
        "parse_rate": len(valid_aes) / len(results),
        "mae_mm": round(np.mean(valid_aes), 3) if valid_aes else None,
        "median_ae_mm": round(np.median(valid_aes), 3) if valid_aes else None,
        "within_1mm": round(np.mean([a < 1 for a in valid_aes]), 3) if valid_aes else None,
        "within_2mm": round(np.mean([a < 2 for a in valid_aes]), 3) if valid_aes else None,
        "within_5mm": round(np.mean([a < 5 for a in valid_aes]), 3) if valid_aes else None,
    }

    print(f"\n=== Qwen3-VL-8B Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ---- Matched pair diagnostic ----
    matched_dir = Path(DATASET_DIR) / "test_matched"
    if matched_dir.exists():
        meta_matched = matched_dir / "metadata.jsonl"
        with open(meta_matched) as f:
            pairs_meta = [json.loads(l) for l in f]

        print(f"\n=== Matched Pair Diagnostic ({len(pairs_meta)} pairs) ===")

        pair_results = []
        for pm in pairs_meta:
            img_a = str(matched_dir / f"pair_{pm['pair_idx']:03d}_a.png")
            img_b = str(matched_dir / f"pair_{pm['pair_idx']:03d}_b.png")

            text_a = run_inference(model, processor, img_a)
            text_b = run_inference(model, processor, img_b)
            pred_a = parse_number(text_a)
            pred_b = parse_number(text_b)

            pair_results.append({
                "pair_idx": pm["pair_idx"],
                "gt_a": pm["diameter_a_mm"],
                "gt_b": pm["diameter_b_mm"],
                "pred_a": pred_a,
                "pred_b": pred_b,
                "same_answer": pred_a == pred_b if (pred_a and pred_b) else None,
            })

        # Analysis
        valid_pairs = [p for p in pair_results if p["pred_a"] and p["pred_b"]]
        if valid_pairs:
            gt_diffs = [p["gt_a"] - p["gt_b"] for p in valid_pairs]
            pred_diffs = [p["pred_a"] - p["pred_b"] for p in valid_pairs]
            same = sum(1 for p in valid_pairs if p["same_answer"])

            corr = np.corrcoef(gt_diffs, pred_diffs)[0, 1] if len(valid_pairs) > 2 else 0
            frac_same = same / len(valid_pairs)

            if corr > 0.5:
                interpretation = "USES scale bar"
            elif corr > 0.2:
                interpretation = "PARTIALLY uses scale bar"
            else:
                interpretation = "IGNORES scale bar"

            matched_metrics = {
                "n_pairs": len(pairs_meta),
                "n_valid": len(valid_pairs),
                "gt_diff_mean": round(np.mean([abs(d) for d in gt_diffs]), 2),
                "pred_diff_mean": round(np.mean([abs(d) for d in pred_diffs]), 2),
                "diff_correlation": round(corr, 4),
                "frac_same_answer": round(frac_same, 3),
                "interpretation": interpretation,
            }

            print(f"  Correlation: {corr:.4f}")
            print(f"  Same answer: {frac_same:.1%}")
            print(f"  Interpretation: {interpretation}")

            metrics["matched_pair"] = matched_metrics

        with open(os.path.join(RESULTS_DIR, "matched_pairs.json"), "w") as f:
            json.dump(pair_results, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {RESULTS_DIR}/")

    del model
    torch.cuda.empty_cache()
    return metrics


def run_probe():
    """Extract vision encoder embeddings and train linear probe."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    out_path = os.path.join(EMBEDDINGS_DIR, "qwen3vl_baseline.npy")

    if os.path.exists(out_path):
        print(f"Embeddings already exist at {out_path}, skipping extraction")
        X = np.load(out_path)
    else:
        print(f"Loading {MODEL_ID} for probing...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.eval()

        # Find the merger module
        merger_module = None
        for name, module in model.named_modules():
            if "merger" in name.lower():
                merger_module = module
                print(f"  Found merger at: {name}")
                break

        if merger_module is None:
            print("  WARNING: No merger found. Searching for visual encoder output...")
            for name, module in model.named_modules():
                if "visual" in name.lower():
                    print(f"    Found: {name}")

        meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
        with open(meta_path) as f:
            samples = [json.loads(l) for l in f]

        embeddings = []
        for i, sample in enumerate(samples):
            image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
            image = Image.open(image_path).convert("RGB")

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
            ]}]

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)

            vision_output = {}

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    vision_output["features"] = output[0].detach().cpu().float()
                elif isinstance(output, torch.Tensor):
                    vision_output["features"] = output.detach().cpu().float()

            hook_handle = merger_module.register_forward_hook(hook_fn)

            with torch.no_grad():
                try:
                    model.generate(**inputs, max_new_tokens=1)
                except Exception:
                    pass

            hook_handle.remove()

            if "features" in vision_output:
                features = vision_output["features"]
                if features.dim() == 3:
                    pooled = features[0].mean(dim=0)
                elif features.dim() == 2:
                    pooled = features.mean(dim=0)
                else:
                    pooled = features.flatten()
                embeddings.append(pooled.numpy())
            else:
                print(f"    Image {sample['idx']}: no features captured")
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[-1]))

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(samples)}")

        X = np.stack(embeddings)
        np.save(out_path, X)
        print(f"  Saved embeddings: {X.shape}")

        del model
        torch.cuda.empty_cache()

    # Load ground truth
    gt_data = np.load(os.path.join(EMBEDDINGS_DIR, "ground_truth.npy"), allow_pickle=True).item()
    gt = np.array([gt_data[i] for i in sorted(gt_data.keys())])

    # Linear probe
    probe = Ridge(alpha=1.0)
    scores = cross_val_score(probe, X, gt, cv=5, scoring="neg_mean_absolute_error")
    mae = -scores.mean()
    mae_std = scores.std()

    probe.fit(X, gt)
    r2 = probe.score(X, gt)

    print(f"\n{'='*70}")
    print(f"  Qwen3-VL-8B Vision Encoder Probe")
    print(f"{'='*70}")
    print(f"  Embedding dim: {X.shape[1]}")
    print(f"  Probe MAE: {mae:.2f}mm (±{mae_std:.2f})")
    print(f"  Probe R²: {r2:.3f}")
    print()
    print(f"  Comparison with Qwen2.5-VL-7B:")
    print(f"    Qwen2.5-VL probe:  4.31mm  R²=0.942")
    print(f"    Qwen3-VL probe:    {mae:.2f}mm  R²={r2:.3f}")
    print()

    # Load Qwen3-VL eval results if available
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        output_mae = metrics.get("mae_mm")
        if output_mae:
            gap = output_mae - mae
            print(f"  Bottleneck 1 analysis:")
            print(f"    Vision encoder probe: {mae:.2f}mm")
            print(f"    Model output:         {output_mae:.2f}mm")
            print(f"    Gap:                  {gap:.2f}mm")
            print()
            if gap > 2:
                print(f"    → Large gap: bottleneck 1 EXISTS in Qwen3-VL")
                print(f"    → The decoding problem persists despite DeepStack")
            elif gap > 0.5:
                print(f"    → Moderate gap: bottleneck 1 partially reduced")
                print(f"    → DeepStack helps but doesn't fully solve decoding")
            else:
                print(f"    → Small gap: bottleneck 1 mostly SOLVED")
                print(f"    → Qwen3-VL decodes spatial info near probe ceiling")

    # Save probe results
    probe_results = {
        "model": MODEL_ID,
        "embedding_dim": int(X.shape[1]),
        "probe_mae_mm": round(mae, 2),
        "probe_mae_std": round(mae_std, 2),
        "probe_r2": round(r2, 3),
    }
    with open(os.path.join(EMBEDDINGS_DIR, "qwen3vl_probe_results.json"), "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"\n✓ Probe results saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.eval:
        run_eval()
    if args.all or args.probe:
        run_probe()
    if not (args.all or args.eval or args.probe):
        parser.print_help()


if __name__ == "__main__":
    main()
