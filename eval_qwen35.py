"""Qwen3.5-9B baseline eval + layer-by-layer probe.

Qwen3.5 uses early fusion — text and vision tokens are mixed from early layers.
Instead of probing one merger layer, we probe hidden states at image token 
positions across ALL transformer layers.

This gives a spatial information curve:
- If high from layer 0: spatial info enters through patch embedding
- If steadily improving: cross-modal attention builds spatial understanding
- If rises then falls: model builds then destroys spatial info for text generation
- If always low: early fusion doesn't help spatial encoding

API notes (from HuggingFace docs + issues):
- Model class: Qwen3_5ForConditionalGeneration
- Requires: pip install transformers from git main
- Image input via apply_chat_template with image paths (not file:// prefix)
- image_token_id = 248056

Usage:
    python3 eval_qwen35.py --eval        # Baseline eval + matched pairs
    python3 eval_qwen35.py --probe       # Layer-by-layer probe
    python3 eval_qwen35.py --all         # Both
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASET_DIR = "dataset"
RESULTS_DIR = "results/qwen35_baseline"
EMBEDDINGS_DIR = "embeddings"
MODEL_ID = "Qwen/Qwen3.5-9B"

PROMPT = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
    "\n\nWhat is the diameter of hole H1 in mm?"
)

IMAGE_TOKEN_ID = 248056  # From Qwen3.5 config


def parse_number(text):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.search(r'(\d+\.?\d*)', text)
    return float(match.group(1)) if match else None


def load_model():
    """Load Qwen3.5-9B."""
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

    print(f"Loading {MODEL_ID}...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def prepare_inputs(processor, image_path):
    """Prepare inputs using Qwen3.5 API."""
    abs_path = os.path.abspath(image_path)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": abs_path},
            {"type": "text", "text": PROMPT},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return inputs


def run_inference(model, processor, image_path):
    """Run inference, return text output."""
    inputs = prepare_inputs(processor, image_path)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text


def smoke_test(model, processor):
    """Verify everything works on one image."""
    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        sample = json.loads(f.readline())

    image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
    print(f"  Smoke test: image {sample['idx']}, gt={sample['diameter_mm']:.1f}mm")

    try:
        text = run_inference(model, processor, image_path)
        pred = parse_number(text)
        print(f"  Output: '{text}' -> {pred}mm")

        # Also check we can find image tokens
        inputs = prepare_inputs(processor, image_path)
        input_ids = inputs["input_ids"][0]
        image_mask = (input_ids == IMAGE_TOKEN_ID)
        n_image_tokens = image_mask.sum().item()
        
        # Also check mm_token_type_ids if available
        if "mm_token_type_ids" in inputs:
            mm_ids = inputs["mm_token_type_ids"][0]
            n_mm = (mm_ids > 0).sum().item()
            print(f"  Image tokens (token_id): {n_image_tokens}, mm_token_type_ids>0: {n_mm}")
        else:
            print(f"  Image tokens (token_id): {n_image_tokens}")
            
        if n_image_tokens == 0:
            print("  WARNING: No image tokens found with IMAGE_TOKEN_ID")
            print(f"  Checking input_ids for unique tokens...")
            unique = torch.unique(input_ids)
            # Look for tokens > 200000 (likely special tokens)
            special = unique[unique > 200000]
            print(f"  Special tokens (>200k): {special.tolist()}")

        print(f"  Smoke test PASSED")
        return True
    except Exception as e:
        print(f"  Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_eval():
    """Baseline evaluation + matched pairs."""
    model, processor = load_model()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n--- Smoke Test ---")
    if not smoke_test(model, processor):
        print("ABORTING: smoke test failed")
        return None

    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]

    print(f"\n=== Qwen3.5-9B Baseline Evaluation ===")
    print(f"Test samples: {len(samples)}")

    results = []
    maes = []
    for i, sample in enumerate(samples):
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
        try:
            text = run_inference(model, processor, image_path)
        except Exception as e:
            print(f"    Image {sample['idx']}: ERROR {e}")
            text = ""

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

    with open(os.path.join(RESULTS_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

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

    print(f"\n=== Qwen3.5-9B Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Matched pairs
    matched_dir = Path(DATASET_DIR) / "test_matched"
    if matched_dir.exists():
        with open(matched_dir / "metadata.jsonl") as f:
            pairs = [json.loads(l) for l in f]

        print(f"\n=== Matched Pair Diagnostic ({len(pairs)} pairs) ===")

        pair_results = []
        for j, pm in enumerate(pairs):
            pid = pm["pair_id"]
            img_a = str(matched_dir / f"pair_{pid:03d}_a.png")
            img_b = str(matched_dir / f"pair_{pid:03d}_b.png")

            try:
                text_a = run_inference(model, processor, img_a)
                text_b = run_inference(model, processor, img_b)
            except Exception as e:
                print(f"    Pair {pid}: ERROR {e}")
                continue

            pred_a = parse_number(text_a)
            pred_b = parse_number(text_b)

            pair_results.append({
                "pair_id": pid,
                "gt_a": pm["diam_a"], "gt_b": pm["diam_b"],
                "pred_a": pred_a, "pred_b": pred_b,
                "same_answer": pred_a == pred_b if (pred_a and pred_b) else None,
            })

            if (j + 1) % 10 == 0:
                print(f"    [{j+1}/{len(pairs)}] pairs done")

        valid = [p for p in pair_results if p["pred_a"] and p["pred_b"]]
        if valid:
            gt_d = [p["gt_a"] - p["gt_b"] for p in valid]
            pr_d = [p["pred_a"] - p["pred_b"] for p in valid]
            corr = np.corrcoef(gt_d, pr_d)[0, 1] if len(valid) > 2 else 0
            same = sum(1 for p in valid if p["same_answer"]) / len(valid)

            if corr > 0.5:
                interp = "USES scale bar"
            elif corr > 0.2:
                interp = "PARTIALLY uses scale bar"
            else:
                interp = "IGNORES scale bar"

            metrics["matched_pair"] = {
                "n_valid": len(valid),
                "diff_correlation": round(corr, 4),
                "frac_same_answer": round(same, 3),
                "interpretation": interp,
            }
            print(f"  Correlation: {corr:.4f}, Same: {same:.1%}, {interp}")

        with open(os.path.join(RESULTS_DIR, "matched_pairs.json"), "w") as f:
            json.dump(pair_results, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {RESULTS_DIR}/")
    del model
    torch.cuda.empty_cache()
    return metrics


def run_probe():
    """Layer-by-layer probe of hidden states at image token positions.
    
    For each transformer layer, extracts hidden states at image token positions,
    mean-pools them, and trains a linear probe to predict diameter.
    
    This reveals WHERE in the network spatial information lives.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    probe_path = os.path.join(EMBEDDINGS_DIR, "qwen35_layer_probes.json")

    print(f"Loading {MODEL_ID} for layer probing...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # Get number of layers
    config = model.config
    if hasattr(config, 'text_config'):
        n_layers = config.text_config.num_hidden_layers
    elif hasattr(config, 'num_hidden_layers'):
        n_layers = config.num_hidden_layers
    else:
        # Try to count layers from model
        n_layers = len([n for n, _ in model.named_modules() if '.layers.' in n and n.endswith('.self_attn')])
    
    print(f"  Model has {n_layers} transformer layers")

    # Test on first image to find image token positions
    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]

    first_path = str(Path(DATASET_DIR) / "test" / f"image_{samples[0]['idx']:04d}.png")
    inputs = prepare_inputs(processor, first_path)
    input_ids = inputs["input_ids"][0]

    # Find image token positions
    image_mask = (input_ids == IMAGE_TOKEN_ID)
    n_image_tokens = image_mask.sum().item()
    
    # Fallback: check mm_token_type_ids
    if n_image_tokens == 0 and "mm_token_type_ids" in inputs:
        mm_ids = inputs["mm_token_type_ids"][0]
        image_mask = (mm_ids > 0)
        n_image_tokens = image_mask.sum().item()
        print(f"  Using mm_token_type_ids for image tokens: {n_image_tokens}")
    else:
        print(f"  Found {n_image_tokens} image tokens via token_id")

    if n_image_tokens == 0:
        print("  ERROR: Cannot identify image tokens. Aborting probe.")
        del model
        torch.cuda.empty_cache()
        return

    # Test forward pass with output_hidden_states
    print("  Testing forward pass with hidden states...")
    test_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**test_inputs, output_hidden_states=True)

    n_hidden = len(outputs.hidden_states)
    hidden_dim = outputs.hidden_states[0].shape[-1]
    print(f"  Got {n_hidden} hidden states, dim={hidden_dim}")
    del outputs
    torch.cuda.empty_cache()

    # Subsample layers to probe (every 4th + first + last)
    layer_indices = sorted(set([0, 1, 2] + list(range(0, n_hidden, max(1, n_hidden // 8))) + [n_hidden - 1]))
    print(f"  Probing {len(layer_indices)} layers: {layer_indices}")

    # Extract embeddings per layer for all test images
  
    n_probe_images = min(100, len(samples))
    print(f"\n  Extracting embeddings for {n_probe_images} images across {len(layer_indices)} layers...")

    # Store embeddings: layer_idx -> list of mean-pooled vectors
    layer_embeddings = {li: [] for li in layer_indices}

    for i in range(n_probe_images):
        sample = samples[i]
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")

        try:
            inputs = prepare_inputs(processor, image_path)
            input_ids = inputs["input_ids"][0]
            
            # Get image mask for this specific image
            img_mask = (input_ids == IMAGE_TOKEN_ID)
            if img_mask.sum() == 0 and "mm_token_type_ids" in inputs:
                img_mask = (inputs["mm_token_type_ids"][0] > 0)

            batch_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**batch_inputs, output_hidden_states=True)

            for li in layer_indices:
                hs = outputs.hidden_states[li][0]  # [seq_len, hidden_dim]
                # Extract only image token hidden states
                image_hs = hs[img_mask.to(hs.device)]  # [n_image_tokens, hidden_dim]
                pooled = image_hs.mean(dim=0).cpu().float().numpy()
                layer_embeddings[li].append(pooled)

            del outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Image {sample['idx']}: ERROR {e}")
            # Fill with zeros
            for li in layer_indices:
                if layer_embeddings[li]:
                    layer_embeddings[li].append(np.zeros_like(layer_embeddings[li][-1]))

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_probe_images}")

    del model
    torch.cuda.empty_cache()

    # Load ground truth
    gt_path = os.path.join(EMBEDDINGS_DIR, "ground_truth.npy")
    if os.path.exists(gt_path):
        gt_data = np.load(gt_path, allow_pickle=True).item()
    else:
        gt_data = {s["idx"]: s["diameter_mm"] for s in samples}
        np.save(gt_path, gt_data)

    gt = np.array([gt_data[samples[i]["idx"]] for i in range(n_probe_images)])

    # Train probes per layer
    print(f"\n{'='*70}")
    print(f"  Qwen3.5-9B Layer-by-Layer Probe Results")
    print(f"  (spatial info at image token positions per layer)")
    print(f"{'='*70}")
    print(f"  {'Layer':>6} {'MAE (mm)':>10} {'±std':>8} {'R²':>8}")
    print(f"  {'-'*34}")

    probe_results = {}
    for li in layer_indices:
        X = np.stack(layer_embeddings[li])
        
        if X.shape[0] < 20:
            print(f"  {li:>6} {'skip':>10} (too few samples)")
            continue

        probe = Ridge(alpha=1.0)
        scores = cross_val_score(probe, X, gt, cv=5, scoring="neg_mean_absolute_error")
        mae = -scores.mean()
        mae_std = scores.std()

        probe.fit(X, gt)
        r2 = probe.score(X, gt)

        probe_results[str(li)] = {
            "layer": li,
            "mae_mm": round(mae, 2),
            "mae_std": round(mae_std, 2),
            "r2": round(r2, 3),
        }
        print(f"  {li:>6} {mae:>10.2f} {mae_std:>8.2f} {r2:>8.3f}")

    print()
    
    # Interpret the curve
    if probe_results:
        maes = [(int(k), v["mae_mm"]) for k, v in probe_results.items()]
        maes.sort()
        first_mae = maes[0][1]
        last_mae = maes[-1][1]
        best_layer, best_mae = min(maes, key=lambda x: x[1])

        print(f"  First layer probe: {first_mae:.2f}mm")
        print(f"  Best layer probe:  {best_mae:.2f}mm (layer {best_layer})")
        print(f"  Last layer probe:  {last_mae:.2f}mm")
        print()

        if best_mae < first_mae - 0.5 and best_layer > maes[0][0]:
            print(f"  → Spatial info BUILDS through layers (improves from {first_mae:.1f} to {best_mae:.1f}mm)")
            if last_mae > best_mae + 0.5:
                print(f"  → Then DEGRADES toward output ({best_mae:.1f} -> {last_mae:.1f}mm)")
                print(f"  → Model builds spatial understanding then overwrites it for text generation")
            else:
                print(f"  → Maintained through to output layer")
        elif abs(first_mae - last_mae) < 0.5:
            print(f"  → Spatial info is FLAT across layers (~{np.mean([m for _, m in maes]):.1f}mm)")
            print(f"  → Entered through patch embedding, not improved by cross-modal attention")
        else:
            print(f"  → Complex pattern — see layer-by-layer results above")

    # Compare with Qwen2.5-VL and Qwen3-VL
    print()
    print(f"  Comparison:")
    print(f"    Qwen2.5-VL-7B merger probe:   4.31mm  R²=0.942")
    print(f"    Qwen3-VL-8B merger probe:      4.22mm  R²=0.728")
    if probe_results:
        print(f"    Qwen3.5-9B best layer probe:  {best_mae:.2f}mm  R²={probe_results[str(best_layer)]['r2']:.3f}")

    # Load eval results if available
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        output_mae = metrics.get("mae_mm")
        if output_mae and probe_results:
            gap = output_mae - best_mae
            print()
            print(f"  === BOTTLENECK ANALYSIS ===")
            print(f"  Best probe:   {best_mae:.2f}mm (layer {best_layer})")
            print(f"  Model output: {output_mae:.2f}mm")
            print(f"  Gap:          {gap:.2f}mm")
            print()
            print(f"  Qwen2.5-VL gap: 3.41mm")
            print(f"  Qwen3-VL gap:   4.78mm")
            print(f"  Qwen3.5 gap:    {gap:.2f}mm")

    with open(probe_path, "w") as f:
        json.dump(probe_results, f, indent=2)
    print(f"\n✓ Layer probe results saved to {probe_path}")


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
