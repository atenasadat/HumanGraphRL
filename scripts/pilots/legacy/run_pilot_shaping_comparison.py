"""End-to-End Pilot Comparison: Baseline vs Cognitive-Shaped RL."""
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_root = str(next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'processed_data').is_dir()))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.experiment.thinkaloud_rl.reasoning_shaper import (
    segment_thought_stream,
    classify_reasoning_step,
    compute_trajectory_shaping,
)

def extract_cognitive_shaping(text: str) -> float:
    # If explicit think tags present, take content; else evaluate steps directly
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    thought = m.group(1).strip() if m else text
    units = segment_thought_stream(thought)
    if len(units) < 2:
        return 0.0
    moves = [classify_reasoning_step(u) for u in units]
    rewards = compute_trajectory_shaping(moves, gamma=0.99)
    return float(sum(rewards) / max(1, len(rewards)))

def task_reward(text: str) -> float:
    words = len(text.split())
    if words < 20: return 0.0
    score = min(0.8, words / 150.0)
    if "```python" in text or "def " in text:
        score += 0.2
    return float(min(1.0, score))

def run_pilot(shaping_weight: float, steps: int = 4, group_size: int = 2):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    # Train only top layer for fast test
    for p in model.parameters(): p.requires_grad = False
    for p in model.model.layers[-1].parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

    prompts = [
        "Plan your approach inside <think> tags, then write Python code for 0/1 knapsack.\n<think>\n1.",
        "Plan your approach inside <think> tags, then write Python code for topological sort.\n<think>\n1.",
    ]

    history = []
    print(f"\n--- Running Pilot Condition (Shaping Weight = {shaping_weight}) ---")
    for step in range(1, steps + 1):
        p_text = random.choice(prompts)
        inputs = tok(p_text, return_tensors="pt").to(device)
        rollouts = []
        with torch.no_grad():
            for _ in range(group_size):
                out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.8, pad_token_id=tok.pad_token_id)
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                rollouts.append(gen)

        tasks = [task_reward(r) for r in rollouts]
        shapes = [extract_cognitive_shaping(r) for r in rollouts]
        totals = [t + shaping_weight * s for t, s in zip(tasks, shapes)]

        mean_tot = sum(totals) / len(totals)
        mean_shape = sum(shapes) / len(shapes)
        std_tot = max(1e-6, (sum((t - mean_tot)**2 for t in totals)/len(totals))**0.5)
        advs = [(t - mean_tot)/std_tot for t in totals]

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        for gen, adv in zip(rollouts, advs):
            enc = tok(p_text + gen, return_tensors="pt").to(device)
            labels = enc["input_ids"].clone()
            labels[:, :inputs["input_ids"].shape[1]] = -100
            loss = loss - adv * model(**enc, labels=labels).loss
        loss = loss / group_size
        loss.backward()
        optimizer.step()

        print(f"Step {step:2d} | Task R: {sum(tasks)/len(tasks):.2f} | Shape Signal: {mean_shape:+.4f} | Loss: {loss.item():+.4f}")
        history.append({"step": step, "task_reward": sum(tasks)/len(tasks), "shape_reward": mean_shape, "loss": loss.item()})
    return history

if __name__ == "__main__":
    h_baseline = run_pilot(shaping_weight=0.0, steps=3, group_size=2)
    h_shaped = run_pilot(shaping_weight=1.0, steps=3, group_size=2)
    out_path = "pilots/results/legacy_runs/thinkaloud_grpo/pilot_comparison_result.json"
    os.makedirs("pilots/results/legacy_runs/thinkaloud_grpo", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline": h_baseline, "shaped": h_shaped}, f, indent=2)
    print(f"\nComparison saved to: {out_path}")
