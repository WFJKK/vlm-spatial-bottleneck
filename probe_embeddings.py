"""Linear probing: did RL change the vision encoder or just the language model?

Extracts patch embeddings from the vision encoder for each test image,
then trains a linear probe to predict diameter. Compares probe accuracy
across baseline, SFT, GRPO, and SFT→RL models.

Two analyses:
  1. Probe on each model's embeddings → predict ground truth
     If GRPO probe > baseline probe: vision encoder improved
     If equal: only language model changed

  2. Probe on BASE embeddings → predict each model's outputs
     If base embeddings predict GRPO outputs well: spatial info was
     already there, language model learned to use it
     If poorly: vision encoder must have changed

Usage:
    python3 probe_embeddings.py --extract    # Extract embeddings (needs GPU)
    python3 probe_embeddings.py --probe      # Train probes (CPU is fine)
    python3 probe_embeddings.py --all        # Both
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)
USER_PROMPT = "What is the diameter of hole H1 in mm?"

# Models to probe (tag -> checkpoint_dir, sft_base)
MODELS = {
    "baseline": {"checkpoint": None, "sft_base": None},
    "sft": {"checkpoint": "checkpoints_sft/final", "sft_base": None},
    "grpo_answer": {"checkpoint": "checkpoints/final", "sft_base": None},
    "sft_then_rl": {"checkpoint": "checkpoints_sft_rl/final", "sft_base": "checkpoints_sft/final"},
}


def load_model_for_probing(checkpoint=None, sft_base=None):
    """Load model, return the full model + processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    if sft_base and os.path.exists(sft_base):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_base)
        model = model.merge_and_unload()

    if checkpoint and os.path.exists(checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def extract_vision_embeddings(model, processor, image_path):
    """Extract patch embeddings from vision encoder before language model.

    Returns a 1D vector (mean-pooled over patches).
    """
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    # Hook into the vision encoder output
    vision_output = {}

    def hook_fn(module, input, output):
        # Qwen2.5-VL: visual encoder outputs image features
        if isinstance(output, tuple):
            vision_output["features"] = output[0].detach().cpu()
        else:
            vision_output["features"] = output.detach().cpu()

    # Find the vision encoder's final layer
    # Qwen2.5-VL structure: model.visual.merger or model.visual
    hook_handle = None
    if hasattr(model, 'visual'):
        # Hook the merger (projects vision features to language model dimension)
        if hasattr(model.visual, 'merger'):
            hook_handle = model.visual.merger.register_forward_hook(hook_fn)
        else:
            hook_handle = model.visual.register_forward_hook(hook_fn)
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        if hasattr(model.model.visual, 'merger'):
            hook_handle = model.model.visual.merger.register_forward_hook(hook_fn)
        else:
            hook_handle = model.model.visual.register_forward_hook(hook_fn)

    if hook_handle is None:
        raise RuntimeError("Could not find vision encoder to hook")

    with torch.no_grad():
        # Just run the forward pass far enough to trigger the vision encoder
        try:
            model.generate(**inputs, max_new_tokens=1)
        except Exception:
            # Even if generation fails, the hook should have fired
            pass

    hook_handle.remove()

    if "features" not in vision_output:
        raise RuntimeError("Hook did not capture vision features")

    features = vision_output["features"]
    # Mean pool over patch dimension
    if features.dim() == 3:
        # [batch, n_patches, hidden_dim]
        pooled = features[0].mean(dim=0)
    elif features.dim() == 2:
        # [n_patches, hidden_dim]
        pooled = features.mean(dim=0)
    else:
        pooled = features.flatten()

    return pooled.numpy().astype(np.float32)


def extract_all_embeddings():
    """Extract embeddings for all test images, all models."""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Load test metadata
    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]

    # Save ground truth
    gt = {s["idx"]: s["diameter_mm"] for s in samples}
    np.save(os.path.join(EMBEDDINGS_DIR, "ground_truth.npy"), gt)

    for tag, config in MODELS.items():
        out_path = os.path.join(EMBEDDINGS_DIR, f"{tag}.npy")
        if os.path.exists(out_path):
            print(f"  {tag}: already extracted, skipping")
            continue

        ckpt = config["checkpoint"]
        sft_base = config["sft_base"]

        # Check if checkpoint exists
        if ckpt and not os.path.exists(ckpt):
            print(f"  {tag}: checkpoint {ckpt} not found, skipping")
            continue

        print(f"\n  Extracting embeddings for: {tag}")
        model, processor = load_model_for_probing(ckpt, sft_base)

        embeddings = []
        for i, sample in enumerate(samples):
            image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
            try:
                emb = extract_vision_embeddings(model, processor, image_path)
                embeddings.append(emb)
            except Exception as e:
                print(f"    Image {sample['idx']}: ERROR {e}")
                embeddings.append(np.zeros_like(embeddings[-1]) if embeddings else np.zeros(1024))

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(samples)}")

        embeddings = np.stack(embeddings)
        np.save(out_path, embeddings)
        print(f"  Saved {tag}: {embeddings.shape}")

        # Free GPU memory for next model
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def run_probes():
    """Train linear probes and report results."""
    # Load ground truth
    gt_data = np.load(os.path.join(EMBEDDINGS_DIR, "ground_truth.npy"), allow_pickle=True).item()
    gt = np.array([gt_data[i] for i in sorted(gt_data.keys())])

    # Also load model predictions for analysis 2
    predictions = {}
    for tag in MODELS:
        results_path = None
        if tag == "baseline":
            results_path = "results/baseline/test_results.json"
        elif tag == "sft":
            results_path = "results/sft/test_results.json"
        elif tag == "grpo_answer":
            results_path = "results/ckpt_final/test_results.json"
        elif tag == "sft_then_rl":
            results_path = "results/sft_then_rl/test_results.json"

        if results_path and os.path.exists(results_path):
            with open(results_path) as f:
                data = json.loads(f.read())
            preds = {r["idx"]: r["predicted_mm"] for r in data if r["predicted_mm"]}
            predictions[tag] = np.array([preds.get(i, np.nan) for i in sorted(gt_data.keys())])

    print("=" * 70)
    print("  Analysis 1: Probe on each model's embeddings → predict ground truth")
    print("  (Does the vision encoder encode spatial info better after training?)")
    print("=" * 70)
    print()

    results = {}
    for tag in MODELS:
        emb_path = os.path.join(EMBEDDINGS_DIR, f"{tag}.npy")
        if not os.path.exists(emb_path):
            print(f"  {tag}: no embeddings found, skipping")
            continue

        X = np.load(emb_path)

        # Ridge regression with cross-validation
        probe = Ridge(alpha=1.0)
        scores = cross_val_score(probe, X, gt, cv=5, scoring="neg_mean_absolute_error")
        mae = -scores.mean()
        mae_std = scores.std()

        # Also fit on all data for R²
        probe.fit(X, gt)
        r2 = probe.score(X, gt)

        results[tag] = {"mae": mae, "mae_std": mae_std, "r2": r2}
        print(f"  {tag:20s}: probe MAE = {mae:.2f}mm (±{mae_std:.2f})  R² = {r2:.3f}")

    print()
    if "baseline" in results and "grpo_answer" in results:
        diff = results["baseline"]["mae"] - results["grpo_answer"]["mae"]
        if diff > 0.5:
            print("  → GRPO improved vision encoder representations (probe MAE lower)")
        elif diff < -0.5:
            print("  → GRPO did NOT improve vision encoder (probe MAE higher)")
        else:
            print("  → Vision encoder representations similar (difference < 0.5mm)")
            print("  → RL likely only changed language model's use of existing features")

    print()
    print("=" * 70)
    print("  Analysis 2: Probe on BASE embeddings → predict each model's outputs")
    print("  (Can the base vision encoder explain each model's predictions?)")
    print("=" * 70)
    print()

    base_emb_path = os.path.join(EMBEDDINGS_DIR, "baseline.npy")
    if not os.path.exists(base_emb_path):
        print("  No baseline embeddings, skipping analysis 2")
        return

    X_base = np.load(base_emb_path)

    for tag, preds in predictions.items():
        valid = ~np.isnan(preds)
        if valid.sum() < 50:
            continue

        probe = Ridge(alpha=1.0)
        scores = cross_val_score(probe, X_base[valid], preds[valid], cv=5,
                                 scoring="neg_mean_absolute_error")
        mae = -scores.mean()

        probe.fit(X_base[valid], preds[valid])
        r2 = probe.score(X_base[valid], preds[valid])

        print(f"  base embeddings → {tag:20s} outputs: MAE = {mae:.2f}mm  R² = {r2:.3f}")

    print()
    print("  If R² is high: base embeddings already contain the info, model learned to use it")
    print("  If R² is low: model's outputs depend on changed embeddings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Extract embeddings (GPU)")
    parser.add_argument("--probe", action="store_true", help="Train probes (CPU)")
    parser.add_argument("--all", action="store_true", help="Both extract and probe")
    args = parser.parse_args()

    if args.all or args.extract:
        extract_all_embeddings()
    if args.all or args.probe:
        run_probes()
    if not (args.all or args.extract or args.probe):
        parser.print_help()


if __name__ == "__main__":
    main()
