"""GRPO with KL penalty against reference policy.

The standard fix for mode collapse. Adds a penalty term that prevents
the policy from diverging too far from the initial model:

    loss = -advantage * log_prob + beta * KL(policy || reference)

This should prevent the round-number collapse seen in vanilla GRPO
and the mode collapse in SFT→RL.

Usage:
    python3 train_grpo_kl.py
    python3 train_grpo_kl.py --resume
"""

from __future__ import annotations

import json, os, re, time, gc, copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_DIR = "dataset"
OUTPUT_DIR = "checkpoints_kl"
LORA_RANK = 64
LORA_ALPHA = 128
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 32
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1
SAVE_EVERY = 100
LOG_EVERY = 5
REWARD_CLIP_MIN = -5.0
KL_BETA = 0.1  # KL penalty weight

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


def compute_reward(text, gt):
    pred = parse_number(text)
    if pred is None or pred <= 0:
        return REWARD_CLIP_MIN
    return max(-abs(pred - gt) / gt, REWARD_CLIP_MIN)


def load_dataset(split="train"):
    meta_path = Path(DATASET_DIR) / split / "metadata.jsonl"
    with open(meta_path) as f:
        return [json.loads(l) for l in f]


def grpo_step_kl(model, ref_model, processor, optimizer, image_path, gt_mm):
    """One GRPO step with KL penalty against reference model."""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    completions = []
    gen_ids_list = []
    model.eval()
    for _ in range(NUM_GENERATIONS):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        gen_ids = outputs[0, input_len:].clone()
        gen_ids_list.append(gen_ids)
        completions.append(processor.decode(gen_ids, skip_special_tokens=True).strip())
        del outputs
    torch.cuda.empty_cache()

    rewards = [compute_reward(c, gt_mm) for c in completions]
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    model.train()
    optimizer.zero_grad()
    total_loss_val = 0.0
    total_kl_val = 0.0

    for gen_ids, advantage in zip(gen_ids_list, advantages):
        full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)

        # Policy forward pass
        out = model(
            input_ids=full_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        logits = out.logits[0, input_len-1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(len(gen_ids)), gen_ids]

        # Reference forward pass (no grad)
        with torch.no_grad():
            ref_out = ref_model(
                input_ids=full_ids,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
            ref_logits = ref_out.logits[0, input_len-1:-1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs[range(len(gen_ids)), gen_ids]

        # KL divergence: sum of (log_policy - log_ref) per token
        kl_per_token = token_log_probs - ref_token_log_probs
        kl = kl_per_token.sum()

        # Combined loss: policy gradient + KL penalty
        pg_loss = -advantage * token_log_probs.sum()
        loss = (pg_loss + KL_BETA * kl) / NUM_GENERATIONS
        loss.backward()

        total_loss_val += pg_loss.item() / NUM_GENERATIONS
        total_kl_val += kl.item() / NUM_GENERATIONS

        del out, ref_out, logits, ref_logits, log_probs, ref_log_probs
        del token_log_probs, ref_token_log_probs, kl_per_token, kl, loss, full_ids
        torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    del inputs, gen_ids_list
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss_val, total_kl_val, rewards, completions


def train(resume=False):
    print("=== GRPO Training with KL Penalty ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"KL beta: {KL_BETA}")

    samples = load_dataset("train")
    print(f"Train samples: {len(samples)}")

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create frozen reference model (base model without LoRA)
    # We use the base model weights which are shared with the LoRA model
    # but we need a separate forward pass that doesn't use LoRA
    print("  Creating reference model (disabling LoRA for reference pass)...")
    # The reference model is just the current model with LoRA disabled
    # We'll use model.disable_adapter_layers() / enable_adapter_layers()

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
    running_reward = []
    running_loss = []
    running_kl = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        indices = rng.permutation(len(samples))

        for i, idx in enumerate(indices):
            global_step += 1
            if global_step <= start_step:
                continue

            sample = samples[idx]
            image_path = str(Path(DATASET_DIR) / "train" / f"image_{sample['idx']:04d}.png")
            gt_mm = sample["diameter_mm"]

            try:
                # Use adapter disable/enable for reference model
                # This avoids loading two copies of the model
                loss_val, kl_val, rewards, completions = grpo_step_kl_efficient(
                    model, processor, optimizer, image_path, gt_mm
                )
                running_reward.extend(rewards)
                running_loss.append(loss_val)
                running_kl.append(kl_val)
            except torch.cuda.OutOfMemoryError:
                print(f"  Step {global_step}: OOM, skipping")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"  Step {global_step}: ERROR {e}")
                continue

            if global_step % LOG_EVERY == 0:
                avg_r = np.mean(running_reward[-LOG_EVERY*NUM_GENERATIONS:])
                avg_l = np.mean(running_loss[-LOG_EVERY:])
                avg_kl = np.mean(running_kl[-LOG_EVERY:])
                best_r = max(running_reward[-LOG_EVERY*NUM_GENERATIONS:])
                elapsed = time.time() - start_time
                actual_steps = global_step - start_step
                spm = actual_steps / (elapsed / 60) if elapsed > 0 else 0
                eta = (len(samples) - global_step) / spm if spm > 0 else 0

                log_entry = {
                    "step": global_step, "epoch": epoch,
                    "avg_reward": round(avg_r, 4),
                    "best_reward": round(best_r, 4),
                    "avg_loss": round(avg_l, 4),
                    "avg_kl": round(avg_kl, 4),
                    "gt_mm": gt_mm,
                    "completions": completions,
                    "rewards": [round(r, 3) for r in rewards],
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(f"  Step {global_step}/{len(samples)} | "
                      f"loss={avg_l:.4f} | kl={avg_kl:.3f} | reward={avg_r:.3f} | "
                      f"best_r={best_r:.3f} | ETA={eta:.0f}m | "
                      f"gt={gt_mm:.1f} pred={completions[0]}")

            if global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                processor.save_pretrained(ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ GRPO+KL training complete. Saved to {final_path}")


def grpo_step_kl_efficient(model, processor, optimizer, image_path, gt_mm):
    """GRPO step using adapter disable/enable for reference model.
    
    This avoids loading two copies of the 7B model. Instead:
    - Generate with LoRA enabled (current policy)
    - Forward pass with LoRA enabled for policy log probs
    - Forward pass with LoRA disabled for reference log probs
    """
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # Generate with LoRA enabled
    completions = []
    gen_ids_list = []
    model.eval()
    for _ in range(NUM_GENERATIONS):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        gen_ids = outputs[0, input_len:].clone()
        gen_ids_list.append(gen_ids)
        completions.append(processor.decode(gen_ids, skip_special_tokens=True).strip())
        del outputs
    torch.cuda.empty_cache()

    rewards = [compute_reward(c, gt_mm) for c in completions]
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    model.train()
    optimizer.zero_grad()
    total_loss_val = 0.0
    total_kl_val = 0.0

    for gen_ids, advantage in zip(gen_ids_list, advantages):
        full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)

        # Policy forward (LoRA enabled)
        out = model(
            input_ids=full_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        logits = out.logits[0, input_len-1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(len(gen_ids)), gen_ids]

        # Reference forward (LoRA disabled)
        with torch.no_grad():
            model.disable_adapter_layers()
            ref_out = model(
                input_ids=full_ids,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
            model.enable_adapter_layers()
            ref_logits = ref_out.logits[0, input_len-1:-1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs[range(len(gen_ids)), gen_ids]

        kl = (token_log_probs - ref_token_log_probs).sum()
        pg_loss = -advantage * token_log_probs.sum()
        loss = (pg_loss + KL_BETA * kl) / NUM_GENERATIONS
        loss.backward()

        total_loss_val += pg_loss.item() / NUM_GENERATIONS
        total_kl_val += kl.item() / NUM_GENERATIONS

        del out, ref_out, logits, ref_logits, log_probs, ref_log_probs
        del token_log_probs, ref_token_log_probs, kl, pg_loss, loss, full_ids
        torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    del inputs, gen_ids_list
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss_val, total_kl_val, rewards, completions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)
