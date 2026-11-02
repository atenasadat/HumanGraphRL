"""Full Local Training & Evaluation Experiment:
Comparing Baseline Outcome-only RL vs Pure Cognitive Process Shaping RL.

Runs full training iterations on local Apple Silicon (MPS) / PyTorch:
1. Condition A (Baseline): Standard GRPO (shaping weight = 0.0)
2. Condition B (Cognitively Shaped): GRPO + Pure Cognitive Shaping (weight = 0.5)
3. Evaluates held-out tasks on both trained checkpoints to measure:
   - Move frequencies (explore, evaluate, backtrack, commit)
   - Reduction in evaluation rumination loops
   - Final task rewards and length efficiency
"""
import json
import math
import os
import random
import re
import sys
import time
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/athenasaghi/VSProjects/TopGRPO")

from scripts.experiment.thinkaloud_rl.reasoning_shaper import (
    segment_thought_stream,
    classify_reasoning_step,
    compute_trajectory_shaping,
    PURE_MOVES,
)

TRAIN_PROMPTS = [
    "Plan your approach inside <think> tags, then write Python code for 0/1 knapsack with dynamic programming.\n<think>\n1.",
    "Plan your approach inside <think> tags, then write Python code for finding cycles in a directed graph.\n<think>\n1.",
    "Plan your approach inside <think> tags, then write Python code for an LRU Cache with O(1) operations.\n<think>\n1.",
    "Plan your approach inside <think> tags, then write Python code for finding the longest palindromic substring.\n<think>\n1.",
    "Plan your approach inside <think> tags, then write Python code for Dijkstra's shortest path algorithm.\n<think>\n1.",
]

EVAL_PROMPTS = [
    "Plan your approach inside <think> tags, then write Python code for merge sort on linked lists.\n<think>\n1.",
    "Plan your approach inside <think> tags, then write Python code for word break dynamic programming.\n<think>\n1.",
    "Plan your approach inside <think> tags, then write Python code for binary tree lowest common ancestor.\n<think>\n1.",
]


def extract_cognitive_shaping(text: str) -> float:
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    thought = m.group(1).strip() if m else text
    units = segment_thought_stream(thought)
    if len(units) < 2:
        return 0.0
    moves = [classify_reasoning_step(u) for u in units]
    rewards = compute_trajectory_shaping(moves, gamma=0.99)
    return float(sum(rewards) / max(1, len(rewards)))


def task_reward(text: str) -> float:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    words = len(clean.split())
    if words < 15:
        return 0.0
    score = min(0.7, words / 120.0)
    if "```python" in clean or "def " in clean or "class " in clean:
        score += 0.3
    return float(min(1.0, score))


def train_condition(condition_name: str, shaping_weight: float, steps: int = 15, group_size: int = 4):
    print(f"\n{'='*70}")
    print(f"STARTING {condition_name.upper()} (Shaping Weight = {shaping_weight}, Steps = {steps}, G = {group_size})")
    print(f"{'='*70}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    # Train top transformer layers
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model.layers[-1].parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

    history = []
    t_start = time.time()

    for step in range(1, steps + 1):
        t0 = time.time()
        p_text = TRAIN_PROMPTS[(step - 1) % len(TRAIN_PROMPTS)]
        inputs = tok(p_text, return_tensors="pt").to(device)

        # Rollouts
        rollouts = []
        with torch.no_grad():
            for _ in range(group_size):
                out = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tok.pad_token_id,
                )
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                rollouts.append(gen)

        tasks = [task_reward(r) for r in rollouts]
        shapes = [extract_cognitive_shaping(r) for r in rollouts]
        totals = [t + shaping_weight * s for t, s in zip(tasks, shapes)]

        mean_tot = sum(totals) / len(totals)
        mean_task = sum(tasks) / len(tasks)
        mean_shape = sum(shapes) / len(shapes)

        std_tot = max(1e-6, (sum((t - mean_tot) ** 2 for t in totals) / len(totals)) ** 0.5)
        advs = [(t - mean_tot) / std_tot for t in totals]

        # Optimization
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        for gen, adv in zip(rollouts, advs):
            enc = tok(p_text + gen, return_tensors="pt").to(device)
            labels = enc["input_ids"].clone()
            labels[:, :inputs["input_ids"].shape[1]] = -100
            out = model(**enc, labels=labels)
            loss = loss - adv * out.loss

        loss = loss / group_size
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0
        print(f"[{condition_name}] Step {step:2d}/{steps} | Task R: {mean_task:.2f} | Shape: {mean_shape:+.4f} | Loss: {loss.item():+.4f} | {elapsed:.1f}s")
        history.append({
            "step": step,
            "task_reward": mean_task,
            "shape_reward": mean_shape,
            "loss": loss.item(),
            "time_sec": elapsed,
        })

    total_time = time.time() - t_start
    print(f"--> {condition_name} finished in {total_time:.1f}s")

    # Evaluate trained model on held-out tasks
    print(f"Evaluating {condition_name} on held-out tasks...")
    eval_moves = Counter()
    eval_tasks = []
    with torch.no_grad():
        for ep in EVAL_PROMPTS:
            inputs = tok(ep, return_tensors="pt").to(device)
            for _ in range(3):
                out = model.generate(**inputs, max_new_tokens=180, do_sample=True, temperature=0.7, pad_token_id=tok.pad_token_id)
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                eval_tasks.append(task_reward(gen))
                m = re.search(r"<think>(.*?)</think>", gen, flags=re.DOTALL)
                thought = m.group(1).strip() if m else gen
                units = segment_thought_stream(thought)
                for u in units:
                    eval_moves[classify_reasoning_step(u)] += 1

    tot_m = max(1, sum(eval_moves[m] for m in PURE_MOVES))
    eval_dist = {m: round(eval_moves[m] / tot_m, 4) for m in PURE_MOVES}
    print(f"Eval Cognitive Distribution: {eval_dist}")
    print(f"Eval Mean Task Reward: {sum(eval_tasks)/len(eval_tasks):.4f}")

    return {
        "condition": condition_name,
        "shaping_weight": shaping_weight,
        "training_time_sec": total_time,
        "training_history": history,
        "eval_task_reward": sum(eval_tasks)/len(eval_tasks),
        "eval_move_distribution": eval_dist,
    }


def main():
    out_dir = "pilots/results/legacy_runs/thinkaloud_grpo"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "full_local_experiment_results.json")

    # Run Condition 1: Baseline (Outcome-only RL, 15 steps)
    res_baseline = train_condition("Baseline_Outcome_RL", shaping_weight=0.0, steps=15, group_size=4)

    # Run Condition 2: Shaped (Cognitive Process Shaping, 15 steps)
    res_shaped = train_condition("Cognitive_Shaped_RL", shaping_weight=0.5, steps=15, group_size=4)

    final_results = {
        "metadata": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "steps_per_condition": 15,
            "group_size": 4,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baseline": res_baseline,
        "shaped": res_shaped,
    }

    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*70}")
    print("ALL RUNS COMPLETE! Results written to:", out_file)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

