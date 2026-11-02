import json
import math
import os
import glob
from collections import Counter

from scripts.experiment.thinkaloud_rl.reasoning_shaper import (
    classify_reasoning_step,
    compute_move_potential,
    compute_trajectory_shaping,
    HUMAN_PHASE_DISTRIBUTIONS,
    HUMAN_TRANSITION_MATRIX,
    PURE_MOVES,
    segment_thought_stream,
)

print("=" * 70)
print("VERIFICATION CHECK 1: The Human Ground Truth (580 Segments from XLSX)")
print("=" * 70)
human_file = "processed_data/cognitive_priors_and_audits_from_transcripts/verbalizations_classified_gemini_flash.json"
with open(human_file) as f:
    human_data = json.load(f)

counts = Counter(r["move"] for r in human_data)
tot = len(human_data)
print(f"Total classified human verbalizations: {tot}")
for m, c in counts.most_common():
    print(f"  {m:12s}: {c:3d} ({c/tot*100:5.1f}%)")

active_tot = sum(counts[m] for m in PURE_MOVES)
print("\nNormalized over Active Cognitive Moves (pure reasoning):")
for m in PURE_MOVES:
    c = counts[m]
    print(f"  {m:12s}: {c:3d} ({c/active_tot*100:5.1f}%)")

print("\n" + "=" * 70)
print("VERIFICATION CHECK 2: Model Audit (LLMs Stuck in 95-98% Evaluate Loop)")
print("=" * 70)
audit_file = "processed_data/cognitive_priors_and_audits_from_transcripts/model_cognitive_audit.json"
with open(audit_file) as f:
    audit_data = json.load(f)

print(f"{'Model':20s} | {'Traces':6s} | {'Nodes':6s} | {'evaluate':10s} | {'commit':8s} | {'backtrack':10s} | {'Mean KL':8s}")
print("-" * 75)
for model_name, stats in audit_data.items():
    d = stats["overall_distribution"]
    print(
        f"{model_name:20s} | {stats['num_traces']:6d} | {stats['total_nodes']:6d} | "
        f"{d['evaluate']*100:8.1f}% | {d['commit']*100:6.1f}% | {d['backtrack']*100:8.1f}% | "
        f"{stats['mean_kl']:7.4f}"
    )

print("\n" + "=" * 70)
print("VERIFICATION CHECK 3: Mathematical Invariance Proof (Telescoping Sum)")
print("=" * 70)
sample_trajectory = ["evaluate", "commit", "evaluate", "backtrack", "evaluate", "commit"]
gamma = 0.99
shaping_rewards = compute_trajectory_shaping(sample_trajectory, gamma=gamma)
discounted_sum = sum(gamma**t * r for t, r in enumerate(shaping_rewards))
print(f"Sample Trajectory Moves: {' -> '.join(sample_trajectory)}")
print(f"Computed Step Shaping Rewards F_t: {[round(r, 4) for r in shaping_rewards]}")
print(f"Discounted Sum (sum_t gamma^t * F_t): {discounted_sum:.10f}")
print("Exact Zero Policy Invariance Verified:", math.isclose(discounted_sum, 0.0, abs_tol=1e-7))

print("\n" + "=" * 70)
print("VERIFICATION CHECK 4: Local Pilot Comparison (Baseline vs Shaped RL)")
print("=" * 70)
pilot_file = "pilots/results/legacy_runs/thinkaloud_grpo/pilot_comparison_result.json"
with open(pilot_file) as f:
    pilot_data = json.load(f)

print("Baseline Condition (Shaping Weight = 0.0):")
for row in pilot_data["baseline"]:
    print(f"  Step {row['step']} | Task Reward: {row['task_reward']:.4f} | Loss: {row['loss']:+.4f}")

print("\nCognitively-Shaped Condition (Shaping Weight = 1.0):")
for row in pilot_data["shaped"]:
    print(f"  Step {row['step']} | Task Reward: {row['task_reward']:.4f} | Shape Return: {row['shape_reward']:+.4f} | Loss: {row['loss']:+.4f}")
print("=" * 70)
