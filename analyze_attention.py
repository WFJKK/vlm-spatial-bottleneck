"""Patch-level attention analysis: where does the language model look?

Extracts attention weights from the language model's layers to see
which image patches it attends to. Compares baseline vs trained models.

Hypothesis: after GRPO training, the language model learns to attend
more strongly to scale bar patches and hole patches, explaining how
it closes the 4.31mm (probe) → 2.29mm (model output) gap through
cross-patch attention.

Usage:
    python3 analyze_attention.py --extract    # Extract attention maps (GPU)
    python3 analyze_attention.py --analyze    # Analyze patterns (CPU)
    python3 analyze_attention.py --all        # Both
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASET_DIR = "dataset"
ATTENTION_DIR = "attention_maps"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)
USER_PROMPT = "What is the diameter of hole H1 in mm?"

# Subset of models to analyze (baseline + best)
MODELS = {
    "baseline": {"checkpoint": None, "sft_base": None},
    "grpo_answer": {"checkpoint": "checkpoints/final", "sft_base": None},
    "sft_then_rl": {"checkpoint": "checkpoints_sft_rl/final", "sft_base": "checkpoints_sft/final"},
}

# Only analyze a subset of test images (attention maps are large)
N_IMAGES = 20


def load_model(checkpoint=None, sft_base=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, attn_implementation="eager",  # Need eager for attention weights
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


def extract_attention(model, processor, image_path):
    """Run inference and extract attention weights.
    
    Returns dict with:
      - image_token_positions: which token positions correspond to image patches
      - attention_to_image: for each generated token, attention over image patches
      - attention_summary: mean attention from output tokens to image patches
    """
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"][0]
    
    # Find image token positions (Qwen2.5-VL uses special token IDs)
    # vision_start=151652, image_pad=151655, vision_end=151653
    image_token_mask = (input_ids == 151655)  # image_pad tokens
    image_positions = torch.where(image_token_mask)[0].cpu().numpy()
    
    n_image_tokens = len(image_positions)
    if n_image_tokens == 0:
        return None

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )

    # Extract attention from the last few layers (most task-relevant)
    # Each attention tensor: [batch, n_heads, seq_len, seq_len]
    n_layers = len(outputs.attentions)
    last_layers = outputs.attentions[-4:]  # Last 4 layers
    
    # For each layer: mean over heads, then get attention from 
    # the last input token (generation position) to image tokens
    gen_position = input_ids.shape[0] - 1  # Last token position
    
    attention_to_image = []
    for attn in last_layers:
        # attn shape: [1, n_heads, seq_len, seq_len]
        # Mean over heads
        attn_mean = attn[0].mean(dim=0)  # [seq_len, seq_len]
        # Attention from generation position to image tokens
        attn_to_img = attn_mean[gen_position, image_positions].cpu().float().numpy()
        attention_to_image.append(attn_to_img)
    
    # Average over last 4 layers
    avg_attention = np.mean(attention_to_image, axis=0)
    
    # Clean up
    del outputs
    torch.cuda.empty_cache()
    
    return {
        "n_image_tokens": n_image_tokens,
        "image_positions": image_positions.tolist(),
        "attention_per_layer": [a.tolist() for a in attention_to_image],
        "avg_attention": avg_attention.tolist(),
        "attention_entropy": float(-np.sum(avg_attention * np.log(avg_attention + 1e-10))),
        "max_attention": float(np.max(avg_attention)),
        "top5_positions": np.argsort(avg_attention)[-5:][::-1].tolist(),
    }


def extract_all():
    """Extract attention maps for subset of test images, all models."""
    os.makedirs(ATTENTION_DIR, exist_ok=True)

    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f][:N_IMAGES]

    for tag, config in MODELS.items():
        out_path = os.path.join(ATTENTION_DIR, f"{tag}.json")
        if os.path.exists(out_path):
            print(f"  {tag}: already extracted, skipping")
            continue

        ckpt = config["checkpoint"]
        if ckpt and not os.path.exists(ckpt):
            print(f"  {tag}: checkpoint {ckpt} not found, skipping")
            continue

        print(f"\n  Extracting attention for: {tag}")
        model, processor = load_model(ckpt, config["sft_base"])

        results = []
        for i, sample in enumerate(samples):
            image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
            try:
                attn = extract_attention(model, processor, image_path)
                if attn:
                    attn["idx"] = sample["idx"]
                    attn["diameter_mm"] = sample["diameter_mm"]
                    attn["scale_bar_mm"] = sample["scale_bar_mm"]
                    results.append(attn)
            except Exception as e:
                print(f"    Image {sample['idx']}: ERROR {e}")

            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(samples)}")

        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  Saved {tag}: {len(results)} attention maps")

        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def analyze():
    """Compare attention patterns across models."""
    print("=" * 70)
    print("  Attention Analysis: Where does the language model look?")
    print("=" * 70)
    print()

    all_data = {}
    for tag in MODELS:
        path = os.path.join(ATTENTION_DIR, f"{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_data[tag] = json.load(f)

    if not all_data:
        print("  No attention data found. Run --extract first.")
        return

    # Compare attention statistics
    print(f"  {'Model':<20} {'Entropy':>10} {'Max Attn':>10} {'Spread':>10}")
    print("  " + "-" * 52)
    
    for tag, data in all_data.items():
        entropies = [d["attention_entropy"] for d in data]
        max_attns = [d["max_attention"] for d in data]
        
        # Spread: std of attention over patches (higher = more focused)
        spreads = []
        for d in data:
            attn = np.array(d["avg_attention"])
            spreads.append(np.std(attn))
        
        print(f"  {tag:<20} {np.mean(entropies):>10.3f} {np.mean(max_attns):>10.4f} {np.mean(spreads):>10.5f}")

    print()
    print("  Higher entropy = more diffuse attention (looking everywhere)")
    print("  Higher max attention = more focused on specific patches")
    print("  Higher spread = more selective (attending strongly to some, weakly to others)")
    
    # Analyze if trained models attend to different patches
    if "baseline" in all_data and "grpo_answer" in all_data:
        print()
        print("  Attention shift analysis (GRPO vs baseline):")
        
        for i in range(min(5, len(all_data["baseline"]))):
            base_attn = np.array(all_data["baseline"][i]["avg_attention"])
            grpo_attn = np.array(all_data["grpo_answer"][i]["avg_attention"])
            
            # Correlation between attention patterns
            corr = np.corrcoef(base_attn, grpo_attn)[0, 1]
            
            # KL divergence
            base_norm = base_attn / (base_attn.sum() + 1e-10)
            grpo_norm = grpo_attn / (grpo_attn.sum() + 1e-10)
            kl = np.sum(grpo_norm * np.log((grpo_norm + 1e-10) / (base_norm + 1e-10)))
            
            idx = all_data["baseline"][i]["idx"]
            print(f"    Image {idx}: attention correlation = {corr:.3f}, KL divergence = {kl:.3f}")
        
        # Overall
        all_corrs = []
        for i in range(min(len(all_data["baseline"]), len(all_data["grpo_answer"]))):
            base_attn = np.array(all_data["baseline"][i]["avg_attention"])
            grpo_attn = np.array(all_data["grpo_answer"][i]["avg_attention"])
            all_corrs.append(np.corrcoef(base_attn, grpo_attn)[0, 1])
        
        print(f"    Mean attention correlation: {np.mean(all_corrs):.3f}")
        print()
        if np.mean(all_corrs) > 0.8:
            print("    → Attention patterns are similar: GRPO didn't change WHERE the model looks")
            print("    → Improvement comes from better USE of attended features, not different attention")
        elif np.mean(all_corrs) > 0.5:
            print("    → Attention patterns partially shifted: GRPO changed where the model focuses")
        else:
            print("    → Attention patterns are very different: GRPO fundamentally changed attention")

    # Save analysis
    analysis = {
        "models": {tag: {
            "mean_entropy": float(np.mean([d["attention_entropy"] for d in data])),
            "mean_max_attention": float(np.mean([d["max_attention"] for d in data])),
            "n_images": len(data),
        } for tag, data in all_data.items()}
    }
    with open(os.path.join(ATTENTION_DIR, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to {ATTENTION_DIR}/analysis.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.extract:
        extract_all()
    if args.all or args.analyze:
        analyze()
    if not (args.all or args.extract or args.analyze):
        parser.print_help()


if __name__ == "__main__":
    main()
