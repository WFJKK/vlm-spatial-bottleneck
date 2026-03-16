"""Evaluation for spatial measurement experiments.

Supports baseline, SFT, GRPO, and CoT GRPO evaluations.
Results saved to results/<tag>/ for comparison.

Usage:
    python3 evaluate.py --baseline
    python3 evaluate.py --checkpoint-dir checkpoints/final --tag grpo_answer
    python3 evaluate.py --checkpoint-dir checkpoints_sft/final --tag sft
    python3 evaluate.py --checkpoint-dir checkpoints_cot/final --tag grpo_cot
    python3 evaluate.py --compare
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASET_DIR = "dataset"
RESULTS_DIR = "results"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)
USER_PROMPT = "What is the diameter of hole H1 in mm?"


def parse_number(text):
    """Extract numeric value from model output (answer-only or final number in CoT)."""
    text = text.strip()
    match = re.search(r'ANSWER:\s*(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        pass
    numbers = re.findall(r'(\d+\.?\d*)', text)
    if numbers:
        return float(numbers[-1])
    return None


def load_model(checkpoint_dir=None, sft_base=None):
    """Load model with optional LoRA adapter.
    
    Args:
        checkpoint_dir: Path to LoRA adapter to evaluate
        sft_base: Path to SFT adapter to merge first (for SFT→RL evals)
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading base model: {MODEL_ID}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    # For SFT→RL: merge SFT adapter into base first
    if sft_base and os.path.exists(sft_base):
        from peft import PeftModel
        print(f"Merging SFT base from {sft_base}")
        model = PeftModel.from_pretrained(model, sft_base)
        model = model.merge_and_unload()

    # Apply evaluation adapter
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        from peft import PeftModel
        print(f"Loading LoRA adapter from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        model = model.merge_and_unload()
    elif checkpoint_dir:
        print(f"WARNING: {checkpoint_dir} not found, using base model")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def run_inference(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def evaluate_test_set(model, processor, split="test", resume_path=None):
    data_dir = Path(DATASET_DIR) / split
    with open(data_dir / "metadata.jsonl") as f:
        samples = [json.loads(l) for l in f]

    results = {}
    if resume_path and os.path.exists(resume_path):
        with open(resume_path) as f:
            for line in f:
                r = json.loads(line)
                results[r["idx"]] = r
        print(f"  Resuming from {len(results)}/{len(samples)} completed")

    pending = [s for s in samples if s["idx"] not in results]
    print(f"  Running inference on {len(pending)} samples...")

    for i, sample in enumerate(pending):
        image_path = str(data_dir / f"image_{sample['idx']:04d}.png")
        start = time.time()
        response = run_inference(model, processor, image_path)
        elapsed = time.time() - start

        predicted = parse_number(response)
        gt = sample["diameter_mm"]
        error = abs(predicted - gt) if predicted is not None else None

        result = {
            "idx": sample["idx"],
            "ground_truth_mm": gt,
            "predicted_mm": predicted,
            "raw_output": response,
            "error_mm": error,
            "relative_error": error / gt if error is not None else None,
            "scale_bar_mm": sample["scale_bar_mm"],
            "hole_px": sample["hole_px"],
            "sb_px": sample["sb_px"],
            "inference_time_s": round(elapsed, 2),
        }
        results[sample["idx"]] = result

        if resume_path:
            with open(resume_path, "a") as f:
                f.write(json.dumps(result) + "\n")

        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            valid = [r for r in results.values() if r["error_mm"] is not None]
            if valid:
                mae = np.mean([r["error_mm"] for r in valid])
                print(f"    [{i+1}/{len(pending)}] MAE so far: {mae:.2f}mm "
                      f"(parsed: {len(valid)}/{len(results)})")

    return results


def evaluate_matched_pairs(model, processor):
    data_dir = Path(DATASET_DIR) / "test_matched"
    meta_path = data_dir / "metadata.jsonl"
    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        pairs = [json.loads(l) for l in f]

    print(f"  Running matched pair diagnostic ({len(pairs)} pairs)...")
    pair_results = []

    for pair in pairs:
        pid = pair["pair_id"]
        img_a = str(data_dir / f"pair_{pid:03d}_a.png")
        img_b = str(data_dir / f"pair_{pid:03d}_b.png")

        resp_a = run_inference(model, processor, img_a)
        resp_b = run_inference(model, processor, img_b)

        pred_a = parse_number(resp_a)
        pred_b = parse_number(resp_b)
        gt_a, gt_b = pair["diam_a"], pair["diam_b"]
        gt_diff = abs(gt_a - gt_b)
        pred_diff = abs(pred_a - pred_b) if (pred_a and pred_b) else None

        pair_results.append({
            "pair_id": pid, "gt_a": gt_a, "gt_b": gt_b,
            "gt_diff": round(gt_diff, 2),
            "pred_a": pred_a, "pred_b": pred_b,
            "pred_diff": round(pred_diff, 2) if pred_diff else None,
            "raw_a": resp_a, "raw_b": resp_b,
            "sb_a": pair["sb_a"], "sb_b": pair["sb_b"],
            "hole_px": pair["hole_px"],
        })

    return pair_results


def compute_metrics(results):
    valid = [r for r in results.values() if r["error_mm"] is not None]
    if not valid:
        return {"n_total": len(results), "n_parsed": 0}

    errors = [r["error_mm"] for r in valid]
    rel_errors = [r["relative_error"] for r in valid]

    metrics = {
        "n_total": len(results), "n_parsed": len(valid),
        "parse_rate": len(valid) / len(results),
        "mae_mm": round(np.mean(errors), 3),
        "median_ae_mm": round(np.median(errors), 3),
        "mae_relative": round(np.mean(rel_errors), 4),
        "within_1mm": round(np.mean([e < 1.0 for e in errors]), 4),
        "within_2mm": round(np.mean([e < 2.0 for e in errors]), 4),
        "within_5mm": round(np.mean([e < 5.0 for e in errors]), 4),
    }

    for sb in sorted(set(r["scale_bar_mm"] for r in valid)):
        subset = [r for r in valid if r["scale_bar_mm"] == sb]
        if len(subset) >= 5:
            metrics[f"mae_sb{sb}mm"] = round(np.mean([r["error_mm"] for r in subset]), 3)

    return metrics


def compute_matched_metrics(pair_results):
    if not pair_results:
        return {}
    valid = [p for p in pair_results if p["pred_diff"] is not None]
    if not valid:
        return {"n_pairs": len(pair_results), "n_valid": 0}

    gt_diffs = [p["gt_diff"] for p in valid]
    pred_diffs = [p["pred_diff"] for p in valid]
    corr = np.corrcoef(gt_diffs, pred_diffs)[0, 1] if len(valid) > 2 else 0
    same_answer = np.mean([d < 1.0 for d in pred_diffs])

    return {
        "n_pairs": len(pair_results), "n_valid": len(valid),
        "gt_diff_mean": round(np.mean(gt_diffs), 2),
        "pred_diff_mean": round(np.mean(pred_diffs), 2),
        "diff_correlation": round(corr, 4),
        "frac_same_answer": round(same_answer, 4),
        "interpretation": (
            "USES scale bar" if corr > 0.5
            else "PARTIALLY uses scale bar" if corr > 0.2
            else "IGNORES scale bar"
        ),
    }


def run_evaluation(checkpoint_dir, tag, sft_base=None):
    print(f"\n{'='*60}")
    print(f"  Evaluation: {tag}")
    print(f"{'='*60}\n")

    model, processor = load_model(checkpoint_dir, sft_base=sft_base)

    results_dir = Path(RESULTS_DIR) / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    resume_path = str(results_dir / "test_results_partial.jsonl")
    results = evaluate_test_set(model, processor, "test", resume_path)

    metrics = compute_metrics(results)
    print(f"\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    pair_results = evaluate_matched_pairs(model, processor)
    matched_metrics = compute_matched_metrics(pair_results)
    if matched_metrics:
        print(f"\n=== Matched Pair Diagnostic ===")
        for k, v in matched_metrics.items():
            print(f"  {k}: {v}")

    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"test": metrics, "matched_pairs": matched_metrics}, f, indent=2)
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(list(results.values()), f, indent=2)
    if pair_results:
        with open(results_dir / "matched_pairs.json", "w") as f:
            json.dump(pair_results, f, indent=2)

    partial = results_dir / "test_results_partial.jsonl"
    if partial.exists():
        partial.unlink()

    print(f"\n✓ Results saved to {results_dir}/")


def compare_results():
    results_dir = Path(RESULTS_DIR)
    tags = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    if not tags:
        print("No results found.")
        return

    print(f"\n{'='*80}")
    print(f"  Comparison across {len(tags)} evaluations")
    print(f"{'='*80}\n")

    print(f"{'Tag':<20} {'MAE':>8} {'Med.AE':>8} {'<1mm':>8} {'<2mm':>8} {'<5mm':>8} "
          f"{'Parse%':>8} {'Matched':>15}")
    print("-" * 95)

    for tag in tags:
        metrics_path = results_dir / tag / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        m = data.get("test", {})
        mp = data.get("matched_pairs", {})

        print(f"{tag:<20} "
              f"{m.get('mae_mm', 'n/a'):>8} "
              f"{m.get('median_ae_mm', 'n/a'):>8} "
              f"{m.get('within_1mm', 'n/a'):>8} "
              f"{m.get('within_2mm', 'n/a'):>8} "
              f"{m.get('within_5mm', 'n/a'):>8} "
              f"{m.get('parse_rate', 'n/a'):>8} "
              f"{mp.get('interpretation', 'n/a'):>15}")

    print("-" * 95)
    print(f"{'mean-guess':<20} {'6.63':>8} {'6.63':>8} {'~0.04':>8} {'':>8} {'~0.36':>8} "
          f"{'n/a':>8} {'IGNORES':>15}")
    print(f"{'pixel-regression':<20} {'5.86':>8} {'':>8} {'':>8} {'':>8} {'':>8} "
          f"{'n/a':>8} {'IGNORES':>15}")
    print(f"{'perfect':<20} {'0.00':>8} {'0.00':>8} {'1.00':>8} {'1.00':>8} {'1.00':>8} "
          f"{'n/a':>8} {'USES':>15}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--sft-base", type=str, default=None,
                        help="SFT adapter to merge before loading checkpoint (for SFT→RL)")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_results()
    elif args.baseline:
        run_evaluation(checkpoint_dir=None, tag="baseline")
    elif args.checkpoint_dir:
        tag = args.tag or args.checkpoint_dir.replace("/", "_")
        run_evaluation(checkpoint_dir=args.checkpoint_dir, tag=tag, sft_base=args.sft_base)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
