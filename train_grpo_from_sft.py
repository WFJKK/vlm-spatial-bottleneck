"""GRPO starting from SFT checkpoint — the standard SFT→RL pipeline.

Loads base model, merges SFT LoRA adapter, applies fresh LoRA, then
runs answer-only GRPO. This tests whether SFT's output precision
combines with GRPO's spatial reasoning.

Requires: checkpoints_sft/final from a previous SFT run.

Usage:
    python3 train_grpo_from_sft.py
    python3 train_grpo_from_sft.py --resume
"""

import json, os, re, time, gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---- Config ----
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SFT_CHECKPOINT = "checkpoints_sft/final"
DATASET_DIR = "dataset"
OUTPUT_DIR = "checkpoints_sft_rl"
LORA_RANK = 64
LORA_ALPHA = 128
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 32          # Answer-only, same as original GRPO
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1
SAVE_EVERY = 100
LOG_EVERY = 5
REWARD_CLIP_MIN = -5.0

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


def gpu_mem_info():
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return a, r, t


def grpo_step(model, processor, optimizer, image_path, gt_mm):
    """One GRPO step with per-completion backward."""
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

    for gen_ids, advantage in zip(gen_ids_list, advantages):
        full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
        out = model(
            input_ids=full_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        logits = out.logits[0, input_len-1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(len(gen_ids)), gen_ids]

        loss = -advantage * token_log_probs.sum() / NUM_GENERATIONS
        loss.backward()
        total_loss_val += loss.item()

        del out, logits, log_probs, token_log_probs, loss, full_ids
        torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    del inputs, gen_ids_list
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss_val, rewards, completions


def train(resume=False):
    print("=== SFT→RL Pipeline (GRPO from SFT checkpoint) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(SFT_CHECKPOINT):
        print(f"ERROR: SFT checkpoint not found at {SFT_CHECKPOINT}")
        print("Run train_sft.py first.")
        return

    # Memory check
    a, _, t = gpu_mem_info()
    print(f"GPU memory: {a:.1f}GB used / {t:.0f}GB total")
    if a > 5:
        print("WARNING: Stale GPU memory. Restart instance first.")
        return

    samples = load_dataset("train")
    print(f"Train samples: {len(samples)}")

    # Load base model in bf16
    print(f"Loading base model {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Merge SFT adapter into base weights
    print(f"Merging SFT adapter from {SFT_CHECKPOINT}...")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    print("  SFT adapter merged")

    # Apply fresh LoRA for GRPO training
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

    a, r, t = gpu_mem_info()
    free = t - r
    print(f"  GPU: {a:.1f}GB allocated, {free:.0f}GB free")

    # 1-step sanity check
    print("\n  Running 1-step sanity check...")
    test_sample = samples[0]
    test_path = str(Path(DATASET_DIR) / "train" / f"image_{test_sample['idx']:04d}.png")
    try:
        loss_val, rewards, completions = grpo_step(
            model, processor, optimizer, test_path, test_sample["diameter_mm"]
        )
        print(f"  PASSED: loss={loss_val:.4f}, pred='{completions[0]}'")
    except torch.cuda.OutOfMemoryError:
        print("  FAILED: OOM")
        return
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    print("\n  Starting full training...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "training_log.jsonl")

    start_step = 0
    if resume and os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            start_step = json.loads(lines[-1])["step"]
            print(f"  Resuming from step {start_step}")

    global_step = 0
    running_reward = []
    running_loss = []
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
                loss_val, rewards, completions = grpo_step(
                    model, processor, optimizer, image_path, gt_mm
                )
                running_reward.extend(rewards)
                running_loss.append(loss_val)
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
                    "gt_mm": gt_mm,
                    "completions": completions,
                    "rewards": [round(r, 3) for r in rewards],
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(f"  Step {global_step}/{len(samples)} | "
                      f"loss={avg_l:.4f} | reward={avg_r:.3f} | "
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
    print(f"\n✓ SFT→RL training complete. Saved to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)
