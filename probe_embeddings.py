"""Vision encoder probing: test whether training changes the vision encoder.

Extracts mean-pooled patch embeddings from the vision encoder's merger
layer for each test image, then trains a linear probe (Ridge regression)
to predict hole diameter. Compares probe accuracy across baseline, SFT,
GRPO, and SFT→RL models.

Analysis 1: Probe each model's embeddings → predict ground truth.
    If GRPO probe ≈ baseline probe: vision encoder unchanged, only LM changed.

Analysis 2: Probe BASE embeddings → predict each model's outputs.
    If R² is high: base vision encoder already contains the information.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

DATASET_DIR: str = "dataset"
EMBEDDINGS_DIR: str = "embeddings"
RESULTS_DIR: str = "results"
MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT: str = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)
USER_PROMPT: str = "What is the diameter of hole H1 in mm?"

MODELS: dict[str, dict[str, str | None]] = {
    "baseline": {"checkpoint": None, "sft_base": None, "results_tag": "baseline"},
    "sft": {"checkpoint": "checkpoints_sft/final", "sft_base": None, "results_tag": "sft"},
    "grpo_answer": {"checkpoint": "checkpoints/final", "sft_base": None, "results_tag": "grpo_answer"},
    "sft_then_rl": {"checkpoint": "checkpoints_sft_rl/final", "sft_base": "checkpoints_sft/final", "results_tag": "sft_then_rl"},
}


def load_model_for_probing(
    checkpoint: str | None = None,
    sft_base: str | None = None,
) -> tuple[Any, Any]:
    """Load model with optional LoRA adapters merged for embedding extraction."""
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


def find_merger_module(model: Any) -> tuple[Any, str]:
    """Auto-discover the vision merger module by searching the model tree."""
    candidates = ["visual.merger", "model.visual.merger", "visual", "model.visual"]

    for path in candidates:
        module = model
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
            print(f"  Found vision merger at: {path}")
            return module, path
        except AttributeError:
            continue

    for name, module in model.named_modules():
        if "merger" in name.lower():
            print(f"  Found merger module at: {name}")
            return module, name

    raise RuntimeError("Could not find vision merger module")


def extract_vision_embeddings(
    model: Any,
    processor: Any,
    image_path: str,
    merger_module: Any,
) -> np.ndarray:
    """Extract mean-pooled patch embeddings from the vision encoder merger layer."""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    vision_output: dict[str, torch.Tensor] = {}

    def hook_fn(module: Any, input: Any, output: Any) -> None:
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

    if "features" not in vision_output:
        raise RuntimeError("Hook did not capture vision features")

    features = vision_output["features"]
    if features.dim() == 3:
        pooled = features[0].mean(dim=0)
    elif features.dim() == 2:
        pooled = features.mean(dim=0)
    else:
        pooled = features.flatten()

    return pooled.numpy()


def extract_all_embeddings() -> None:
    """Extract embeddings for all test images across all model variants."""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]

    gt = {s["idx"]: s["diameter_mm"] for s in samples}
    np.save(os.path.join(EMBEDDINGS_DIR, "ground_truth.npy"), gt)

    for tag, config in MODELS.items():
        out_path = os.path.join(EMBEDDINGS_DIR, f"{tag}.npy")
        if os.path.exists(out_path):
            print(f"  {tag}: already extracted, skipping")
            continue

        ckpt = config["checkpoint"]
        sft_base = config["sft_base"]

        if ckpt and not os.path.exists(ckpt):
            print(f"  {tag}: checkpoint {ckpt} not found, skipping")
            continue

        print(f"\n  Extracting embeddings for: {tag}")
        model, processor = load_model_for_probing(ckpt, sft_base)
        merger_module, _ = find_merger_module(model)

        embeddings: list[np.ndarray] = []
        for i, sample in enumerate(samples):
            image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
            try:
                emb = extract_vision_embeddings(model, processor, image_path, merger_module)
                embeddings.append(emb)
            except Exception as e:
                print(f"    Image {sample['idx']}: ERROR {e}")
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[-1]))
                else:
                    print(f"    First image failed, skipping model {tag}")
                    break

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(samples)}")

        if len(embeddings) == len(samples):
            stacked = np.stack(embeddings)
            np.save(out_path, stacked)
            print(f"  Saved {tag}: {stacked.shape}")

        del model
        torch.cuda.empty_cache()
        gc.collect()


def run_probes() -> None:
    """Train linear probes and print comparison across model variants."""
    gt_data = np.load(os.path.join(EMBEDDINGS_DIR, "ground_truth.npy"), allow_pickle=True).item()
    gt = np.array([gt_data[i] for i in sorted(gt_data.keys())])

    predictions: dict[str, np.ndarray] = {}
    for tag, config in MODELS.items():
        results_tag = config["results_tag"]
        results_path = f"{RESULTS_DIR}/{results_tag}/test_results.json"

        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.loads(f.read())
            preds = {r["idx"]: r["predicted_mm"] for r in data if r["predicted_mm"]}
            predictions[tag] = np.array([preds.get(i, np.nan) for i in sorted(gt_data.keys())])

    print("=" * 70)
    print("  Analysis 1: Probe on each model's embeddings → predict ground truth")
    print("  (Does the vision encoder encode spatial info better after training?)")
    print("=" * 70)
    print()

    probe_results: dict[str, dict[str, float]] = {}
    for tag in MODELS:
        emb_path = os.path.join(EMBEDDINGS_DIR, f"{tag}.npy")
        if not os.path.exists(emb_path):
            print(f"  {tag}: no embeddings found, skipping")
            continue

        X = np.load(emb_path)
        probe = Ridge(alpha=1.0)
        scores = cross_val_score(probe, X, gt, cv=5, scoring="neg_mean_absolute_error")
        mae = -scores.mean()
        mae_std = scores.std()

        probe.fit(X, gt)
        r2 = probe.score(X, gt)

        probe_results[tag] = {"mae": mae, "mae_std": mae_std, "r2": r2}
        print(f"  {tag:20s}: probe MAE = {mae:.2f}mm (±{mae_std:.2f})  R² = {r2:.3f}")

    print()
    if "baseline" in probe_results and "grpo_answer" in probe_results:
        diff = probe_results["baseline"]["mae"] - probe_results["grpo_answer"]["mae"]
        if abs(diff) < 0.5:
            print("  → Vision encoder representations similar (difference < 0.5mm)")
            print("  → RL likely only changed language model's use of existing features")
        elif diff > 0:
            print("  → GRPO improved vision encoder representations (probe MAE lower)")
        else:
            print("  → GRPO did NOT improve vision encoder (probe MAE higher)")

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

    pred_probe_results: dict[str, dict[str, float]] = {}
    for tag, preds in predictions.items():
        valid = ~np.isnan(preds)
        if valid.sum() < 50:
            continue

        probe = Ridge(alpha=1.0)
        scores = cross_val_score(
            probe, X_base[valid], preds[valid], cv=5, scoring="neg_mean_absolute_error"
        )
        mae = -scores.mean()

        probe.fit(X_base[valid], preds[valid])
        r2 = probe.score(X_base[valid], preds[valid])

        pred_probe_results[tag] = {"mae": mae, "r2": r2}
        print(f"  base embeddings → {tag:20s} outputs: MAE = {mae:.2f}mm  R² = {r2:.3f}")

    print()
    print("  If R² is high: base embeddings already contain the info, model learned to use it")
    print("  If R² is low: model's outputs depend on changed vision encoder representations")

    all_results = {"analysis1_gt_probe": probe_results, "analysis2_pred_probe": pred_probe_results}
    with open(os.path.join(EMBEDDINGS_DIR, "probe_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Probe results saved to {EMBEDDINGS_DIR}/probe_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision encoder probing analysis")
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
