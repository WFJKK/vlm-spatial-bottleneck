"""GRPO training for spatial measurement in VLMs.

Task: Given an image of a single hole with a scale bar, predict the
hole diameter in mm. Answer-only format (no chain-of-thought).

Model: Mistral 3 8B Instruct (FP8 -> bf16 auto-dequant on A100)
Training: TRL GRPOTrainer with vLLM colocate mode, LoRA rank 64
Reward: Continuous relative error: -abs(pred - gt) / gt

Usage:
    python3 train_grpo.py                    # Start fresh
    python3 train_grpo.py --resume           # Resume from last checkpoint
    python3 train_grpo.py --test-reward      # Test reward function only (no GPU)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


# Heavy imports deferred to train() function
# import torch
# from datasets import Dataset, Features, Image, Value
# from PIL import Image as PILImage


# ---- Configuration ----

MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"
# For faster iteration on smaller GPU: "mistralai/Ministral-3B-Instruct-2501"

DATASET_DIR = "dataset"
OUTPUT_DIR = "checkpoints"
LOG_FILE = "training_log.jsonl"

LORA_RANK = 64
LORA_ALPHA = 128

# GRPO hyperparameters (sized for ~4hr training on A100 80GB)
NUM_GENERATIONS = 4          # Samples per prompt (4 not 8 to fit time budget)
MAX_NEW_TOKENS = 32          # Answer-only: just a number
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1         # 1 epoch — enough to see if MAE moves
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
SAVE_STEPS = 50
LOGGING_STEPS = 10

# Reward parameters
REWARD_CLIP_MIN = -5.0       # Floor for very bad answers


# ---- Prompt template ----

SYSTEM_PROMPT = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)

USER_PROMPT = "What is the diameter of hole H1 in mm?"


def build_prompt(image_path: str) -> list[dict]:
    """Build chat-formatted prompt with image for the model."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
            ],
        }
    ]


# ---- Reward function ----

def parse_number(text: str) -> float | None:
    """Extract a numeric value from model output.
    
    Handles formats like: "15.3", "15.3mm", "15.3 mm", "The diameter is 15.3mm"
    """
    text = text.strip()
    
    # Try direct float parse first
    try:
        return float(text)
    except ValueError:
        pass
    
    # Extract first number from text
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    
    return None


def compute_reward(completion: str, ground_truth_mm: float) -> float:
    """Compute reward for a single completion.
    
    Reward = -abs(pred - gt) / gt  (continuous relative error)
    
    Perfect answer: 0.0
    Off by 100%: -1.0
    Unparseable: REWARD_CLIP_MIN
    """
    pred = parse_number(completion)
    
    if pred is None:
        return REWARD_CLIP_MIN
    
    if pred <= 0:
        return REWARD_CLIP_MIN
    
    error = abs(pred - ground_truth_mm)
    relative_error = error / ground_truth_mm
    reward = -relative_error
    
    return max(reward, REWARD_CLIP_MIN)


def reward_function(completions: list[str], ground_truth_mm: list[float], **kwargs) -> list[float]:
    """Batch reward function for GRPOTrainer.
    
    Args:
        completions: Model outputs (list of strings)
        ground_truth_mm: Ground truth diameters from dataset
    
    Returns:
        List of reward floats
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth_mm):
        # Extract text from completion if it's a list of dicts
        if isinstance(completion, list):
            text = " ".join(
                c.get("content", "") if isinstance(c, dict) else str(c)
                for c in completion
            )
        elif isinstance(completion, dict):
            text = completion.get("content", str(completion))
        else:
            text = str(completion)
        
        rewards.append(compute_reward(text, gt))
    
    return rewards


# ---- Dataset loading ----

def load_dataset_from_dir(split: str = "train") -> "Dataset":
    """Load generated dataset into HuggingFace Dataset format."""
    from datasets import Dataset
    data_dir = Path(DATASET_DIR) / split
    metadata_path = data_dir / "metadata.jsonl"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run setup.sh first to generate the dataset."
        )
    
    records = []
    with open(metadata_path) as f:
        for line in f:
            sample = json.loads(line)
            image_path = str(data_dir / f"image_{sample['idx']:04d}.png")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            records.append({
                "image_path": image_path,
                "prompt": build_prompt(image_path),
                "ground_truth_mm": sample["diameter_mm"],
                "scale_bar_mm": sample["scale_bar_mm"],
                "hole_px": sample["hole_px"],
                "sb_px": sample["sb_px"],
                "idx": sample["idx"],
            })
    
    return Dataset.from_list(records)


# ---- Test reward function ----

def test_reward():
    """Quick test of the reward function without GPU."""
    print("=== Testing reward function ===\n")
    
    test_cases = [
        ("15.3", 15.0, "Close answer"),
        ("15.0", 15.0, "Perfect answer"),
        ("30.0", 15.0, "100% off"),
        ("7.5", 15.0, "50% off"),
        ("abc", 15.0, "Unparseable"),
        ("The diameter is 15.3mm", 15.0, "Verbose answer"),
        ("-5", 15.0, "Negative"),
        ("0", 15.0, "Zero"),
        ("15.3 mm", 15.0, "With unit"),
    ]
    
    for text, gt, label in test_cases:
        r = compute_reward(text, gt)
        parsed = parse_number(text)
        print(f"  {label:20s} | text='{text:30s}' | parsed={parsed} | gt={gt} | reward={r:+.3f}")
    
    # Test batch function
    print("\n=== Testing batch reward ===")
    completions = ["15.3", "30.0", "abc"]
    gts = [15.0, 15.0, 15.0]
    rewards = reward_function(completions, gts)
    print(f"  Completions: {completions}")
    print(f"  Ground truths: {gts}")
    print(f"  Rewards: {rewards}")
    
    # Test on dataset metadata if available
    meta_path = Path(DATASET_DIR) / "train" / "metadata.jsonl"
    if meta_path.exists():
        with open(meta_path) as f:
            samples = [json.loads(l) for l in f]
        
        diams = [s["diameter_mm"] for s in samples]
        mean_d = np.mean(diams)
        
        print(f"\n=== Dataset sanity check ===")
        print(f"  N samples: {len(samples)}")
        print(f"  Diameter range: {min(diams):.1f} - {max(diams):.1f} mm")
        print(f"  Mean: {mean_d:.1f} mm")
        
        # Reward for always-guess-mean
        mean_rewards = [compute_reward(f"{mean_d:.1f}", d) for d in diams]
        print(f"  Mean-guess avg reward: {np.mean(mean_rewards):.3f}")
        
        # Reward for perfect answers
        perfect_rewards = [compute_reward(f"{d:.2f}", d) for d in diams]
        print(f"  Perfect avg reward: {np.mean(perfect_rewards):.4f}")
    
    print("\n✓ Reward function tests passed")


# ---- Training ----

def train(resume: bool = False):
    """Run GRPO training."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer
    
    print("=== GRPO Training ===\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("✗ No GPU available. Use --test-reward for CPU testing.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = load_dataset_from_dir("train")
    print(f"  Train samples: {len(train_dataset)}")
    
    # Model loading
    print(f"Loading model: {MODEL_ID}")
    
    # Try FP8 first (A100), fall back to bf16
    try:
        from transformers import FineGrainedFP8Config
        quantization_config = FineGrainedFP8Config(
            weights_dtype=torch.float8_e4m3fn,
            activation_dtype=torch.bfloat16,
        )
        print("  Using FineGrainedFP8Config (weights FP8 -> bf16 compute)")
    except ImportError:
        print("  FineGrainedFP8Config not available, using bf16")
        quantization_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    
    # GRPO config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=NUM_GENERATIONS,
        max_new_tokens=MAX_NEW_TOKENS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        bf16=True,
        report_to="none",
        save_total_limit=3,         # Keep last 3 checkpoints
        dataloader_num_workers=2,
        # vLLM colocate mode for efficient generation
        use_vllm=True,
        vllm_device="auto",
        vllm_gpu_memory_utilization=0.4,
    )
    
    # Resume from checkpoint
    resume_from = None
    if resume:
        checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"  Resuming from {resume_from}")
        else:
            print("  No checkpoints found, starting fresh")
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_function,
        peft_config=lora_config,
        processing_class=processor,
    )
    
    # Train
    print("\nStarting GRPO training...")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  Effective batch: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Generations per prompt: {NUM_GENERATIONS}")
    print(f"  Save every {SAVE_STEPS} steps")
    print()
    
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Training complete. Final model saved to {final_path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="GRPO training for spatial measurement")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--test-reward", action="store_true",
                        help="Test reward function only (no GPU needed)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model ID")
    args = parser.parse_args()
    
    if args.model:
        global MODEL_ID
        MODEL_ID = args.model
    
    if args.test_reward:
        test_reward()
    else:
        train(resume=args.resume)


if __name__ == "__main__":
    main()
