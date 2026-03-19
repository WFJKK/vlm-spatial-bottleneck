"""Supervised fine-tuning baseline for spatial measurement (1 epoch).

Trains on (image, question) → ground truth diameter. Same LoRA
configuration as all GRPO variants for controlled comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_DIR: str = "dataset"
OUTPUT_DIR: str = "checkpoints_sft"
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_EPOCHS: int = 1
SAVE_EVERY: int = 100
LOG_EVERY: int = 10

SYSTEM_PROMPT: str = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)
USER_PROMPT: str = "What is the diameter of hole H1 in mm?"


def load_dataset(split: str = "train") -> list[dict[str, Any]]:
    """Load samples from a JSONL metadata file."""
    meta_path = Path(DATASET_DIR) / split / "metadata.jsonl"
    with open(meta_path) as f:
        return [json.loads(l) for l in f]


def train(resume: bool = False) -> None:
    """Run SFT training loop with LoRA on Qwen2.5-VL."""
    print("=== SFT Baseline Training ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    samples = load_dataset("train")
    print(f"Train samples: {len(samples)}")

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "training_log.jsonl")

    start_step = 0
    if resume and os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            start_step = json.loads(lines[-1])["step"]
            print(f"Resuming from step {start_step}")

    global_step = 0
    running_loss: list[float] = []

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        indices = rng.permutation(len(samples))

        for idx in indices:
            global_step += 1
            if global_step <= start_step:
                continue

            sample = samples[idx]
            image_path = str(Path(DATASET_DIR) / "train" / f"image_{sample['idx']:04d}.png")
            gt_mm = sample["diameter_mm"]

            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
                ]},
                {"role": "assistant", "content": f"{gt_mm}"},
            ]

            text = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            model.train()
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss.append(loss.item())

            if global_step % LOG_EVERY == 0:
                avg_loss = np.mean(running_loss[-LOG_EVERY:])
                log_entry = {"step": global_step, "epoch": epoch, "avg_loss": round(avg_loss, 4), "gt_mm": gt_mm}
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                print(f"  Step {global_step}/{len(samples)} | loss={avg_loss:.4f} | gt={gt_mm:.1f}")

            if global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                processor.save_pretrained(ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ SFT training complete. Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)
