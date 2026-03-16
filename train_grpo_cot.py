"""GRPO with chain-of-thought for spatial measurement.

Same as answer-only GRPO but the model can generate up to 256 tokens.
Reward is ONLY based on the final number — the reasoning is unconstrained.

The hypothesis: the model will discover its own spatial reasoning strategy
in the scratchpad (referencing the scale bar, estimating proportions, etc.)
and this will lead to better accuracy than answer-only GRPO.

If the model develops scale bar reasoning language ON ITS OWN — without
being taught it — that's a DeepSeek-R1-style emergent reasoning finding.

Usage:
    python3 train_grpo_cot.py
    python3 train_grpo_cot.py --resume
"""

import json, os, re, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---- Config ----
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_DIR = "dataset"
OUTPUT_DIR = "checkpoints_cot"
LORA_RANK = 64
LORA_ALPHA = 128
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 256        # Room for reasoning
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1
SAVE_EVERY = 100
LOG_EVERY = 5
REWARD_CLIP_MIN = -5.0

# Prompt encourages reasoning but doesn't prescribe a method
SYSTEM_PROMPT = (
    "You are measuring a hole in a technical drawing. "
    "Think step by step about how to determine the diameter using the scale bar. "
    "End your response with the diameter in mm on its own line, like: ANSWER: <number>"
)
USER_PROMPT = "What is the diameter of hole H1 in mm?"


def parse_answer(text):
    """Extract the final ANSWER: <number> from CoT output."""
    text = text.strip()

    # Try ANSWER: pattern first
    match = re.search(r'ANSWER:\s*(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback: last number in the text
    numbers = re.findall(r'(\d+\.?\d*)', text)
    if numbers:
        return float(numbers[-1])

    return None


def compute_reward(text, gt):
    pred = parse_answer(text)
    if pred is None or pred <= 0:
        return REWARD_CLIP_MIN
    return max(-abs(pred - gt) / gt, REWARD_CLIP_MIN)


def load_dataset(split="train"):
    meta_path = Path(DATASET_DIR) / split / "metadata.jsonl"
    with open(meta_path) as f:
        return [json.loads(l) for l in f]


def grpo_loss(model, processor, image_path, gt_mm):
    """Compute GRPO loss for one sample with CoT."""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # Step 1: Generate N completions (no grad)
    completions = []
    gen_ids_list = []

    model.eval()
    for _ in range(NUM_GENERATIONS):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        gen_ids = outputs[0, input_len:]
        gen_ids_list.append(gen_ids)
        completions.append(processor.decode(gen_ids, skip_special_tokens=True).strip())

    # Step 2: Compute rewards and advantages (reward ONLY on final answer)
    rewards = [compute_reward(c, gt_mm) for c in completions]
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    # Step 3: Policy gradient loss
    model.train()
    total_loss = torch.tensor(0.0, device=model.device)

    for gen_ids, advantage in zip(gen_ids_list, advantages):
        full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)

        out = model(input_ids=full_ids, pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"))
        logits = out.logits[0, input_len-1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(len(gen_ids)), gen_ids]

        total_loss += -advantage * token_log_probs.sum()

    total_loss = total_loss / NUM_GENERATIONS
    return total_loss, rewards, completions


def train(resume=False):
    print("=== CoT GRPO Training ===\n")

    device = torch.device("cuda")
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
                loss, rewards, completions = grpo_loss(model, processor, image_path, gt_mm)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_reward.extend(rewards)
                running_loss.append(loss.item())

            except Exception as e:
                print(f"  Step {global_step}: ERROR {e}")
                continue

            if global_step % LOG_EVERY == 0:
                avg_r = np.mean(running_reward[-LOG_EVERY*NUM_GENERATIONS:])
                avg_l = np.mean(running_loss[-LOG_EVERY:])
                best_r = max(running_reward[-LOG_EVERY*NUM_GENERATIONS:])
                elapsed = time.time() - start_time
                steps_per_min = global_step / (elapsed / 60)
                eta_min = (len(samples) - global_step) / steps_per_min if steps_per_min > 0 else 0

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

                # Show first completion (truncated) to see reasoning
                preview = completions[0][:80].replace('\n', ' ')
                print(f"  Step {global_step}/{len(samples)} | "
                      f"loss={avg_l:.4f} | reward={avg_r:.3f} | "
                      f"best_r={best_r:.3f} | ETA={eta_min:.0f}m | "
                      f"gt={gt_mm:.1f} | '{preview}'")

            if global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                processor.save_pretrained(ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ CoT GRPO training complete. Model saved to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)
