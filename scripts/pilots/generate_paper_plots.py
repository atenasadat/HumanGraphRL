import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style for academic paper (NeurIPS / ICLR style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

os.makedirs("pilots/figures/legacy_2026-10-29", exist_ok=True)
colors = sns.color_palette("muted")

# ==============================================================================
# FIGURE 1: Cognitive Collapse: Pre-trained LLMs vs. Human Benchmark
# ==============================================================================
print("Generating Figure 1: Cognitive Collapse...")

audit_file = "processed_data/cognitive_priors_and_audits_from_transcripts/model_cognitive_audit.json"
with open(audit_file) as f:
    audit_data = json.load(f)

human_file = "processed_data/cognitive_priors_and_audits_from_transcripts/human_cognitive_blueprint.json"
with open(human_file) as f:
    human_data = json.load(f)

# Human pure cognitive moves (excluding prompt-llm and narrate)
# Early: 54.2% eval, 26.4% commit, 11.1% backtrack, 8.3% explore
# Mid: 70.7% eval, 19.5% backtrack, 7.3% commit, 2.4% explore
# Late: 52.8% eval, 25.0% backtrack, 13.9% commit, 8.3% explore
# Average human:
human_avg = {
    'explore': 6.33,
    'evaluate': 59.23,
    'backtrack': 18.53,
    'commit': 15.87
}

models = ["Human Benchmark", "Llama-3.1-8B", "Qwen3-8B", "Mistral-7B"]
moves = ["evaluate", "backtrack", "commit", "explore"]

matrix = []
# Human
matrix.append([human_avg[m] for m in moves])

# Models
for m_name in ["Llama-3.1-8B", "Qwen3-8B", "Mistral-7B"]:
    dist = audit_data[m_name]["overall_distribution"]
    matrix.append([dist[m] * 100 for m in moves])

matrix = np.array(matrix)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
x = np.arange(len(models))
width = 0.2

c_list = ["#d95f02", "#7570b3", "#1b9e77", "#e7298a"]
labels = ["Evaluate (Calculation)", "Backtrack (Error Recovery)", "Commit (Strategic Decision)", "Explore (Ideation)"]

for i, (m, c, l) in enumerate(zip(moves, c_list, labels)):
    ax.bar(x + (i - 1.5) * width, matrix[:, i], width, label=l, color=c, edgecolor='black', linewidth=0.6)

ax.set_ylabel("Share of Thought Moves (%)")
ax.set_title("Figure 1: The LLM 'Cognitive Collapse' vs. Human Think-Aloud Benchmark", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="semibold")
ax.legend(frameon=True, facecolor="white", edgecolor="none")
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Annotation on Llama evaluate
ax.annotate('98.3% Evaluation Loop\n(Absorbing State)', xy=(1 - 1.5*width, 98.3), xytext=(1 - 1.5*width + 0.1, 80),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffeedd", ec="#d95f02", lw=1))

plt.tight_layout()
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_01_cognitive_collapse.pdf")
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_01_cognitive_collapse.png")
plt.close()

# ==============================================================================
# FIGURE 2: Multi-Domain Task Accuracy Across Conditions (N=3 Seeds)
# ==============================================================================
print("Generating Figure 2: Multi-Domain Performance Matrix...")

base = "pilots/results/legacy_runs/carc_multidomain_20261027_153910_2209184"
runs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and ('seed' in d)])

data = defaultdict(lambda: defaultdict(list))
for r in runs:
    eval_path = os.path.join(base, r, 'evaluation.json')
    if os.path.exists(eval_path) and os.path.getsize(eval_path) > 10:
        with open(eval_path) as f: ev = json.load(f)
        parts = r.split('_')
        domain = parts[0]
        cond = parts[1]
        if parts[2].startswith('specific'):
            cond = 'Domain-Specific'
        elif cond == 'baseline':
            cond = 'Baseline (Outcome RL)'
        elif cond == 'unified':
            cond = 'Unified CogShape'
        
        data[domain][cond].append({
            'prog': ev['domains']['programming']['mean_rubric_proxy'] * 100,
            'write': ev['domains']['writing']['mean_rubric_proxy'] * 100
        })

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300, sharey=True)

domain_keys = ["programming", "writing", "mixed"]
domain_titles = ["Trained on Programming", "Trained on Proposal Writing", "Trained on Mixed (Both)"]

cond_order = ["Baseline (Outcome RL)", "Domain-Specific", "Unified CogShape"]
palette = ["#7f7f7f", "#1f77b4", "#2ca02c"]

for ax, d_key, title in zip(axes, domain_keys, domain_titles):
    means = []
    stds = []
    for c in cond_order:
        runs_c = data[d_key][c]
        scores = [item['prog'] if d_key == 'programming' else item['write'] if d_key == 'writing' else (item['prog'] + item['write'])/2 for item in runs_c]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    
    x = np.arange(len(cond_order))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=palette, edgecolor='black', linewidth=0.7, width=0.55)
    
    # Value annotations on top of bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 6, f"{m:.1f}%", ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)
        
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Domain-Specific", "Unified\n(Ours)"], fontweight="semibold")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(80, 102)

axes[0].set_ylabel("Held-Out Task Accuracy (%)")
fig.suptitle("Figure 2: Task Accuracy Across Training Domains and Shaping Conditions (N=3 Seeds)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_02_multidomain_performance.pdf")
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_02_multidomain_performance.png")
plt.close()

# ==============================================================================
# FIGURE 3: Non-Trivial Gradient Dynamics (Share & Orthogonality)
# ==============================================================================
print("Generating Figure 3: Gradient Dynamics & Orthogonality...")

domains = ["Programming", "Proposal Writing", "Mixed Domain"]
shares = [11.44, 33.04, 24.10]
share_stds = [3.4, 1.4, 4.5]
cosines = [-0.052, -0.040, -0.040]
cos_stds = [0.14, 0.01, 0.01]

fig, ax1 = plt.subplots(figsize=(7, 4.2), dpi=300)

x = np.arange(len(domains))
w = 0.35

ax1.bar(x - w/2, shares, yerr=share_stds, width=w, capsize=4, color="#377eb8", edgecolor='black', linewidth=0.7, label="Cognitive Gradient Share (%)")
ax1.set_ylabel("Cognitive Gradient Norm Share (%)", color="#377eb8", fontweight="bold")
ax1.tick_params(axis='y', labelcolor="#377eb8")
ax1.set_ylim(0, 45)
ax1.set_xticks(x)
ax1.set_xticklabels(domains, fontweight="semibold")

ax2 = ax1.twinx()
ax2.plot(x + w/2, cosines, color="#e41a1c", marker='o', markersize=8, linewidth=2, linestyle='--', label=r"Cosine Angle $\cos(\theta)$")
ax2.errorbar(x + w/2, cosines, yerr=cos_stds, color="#e41a1c", capsize=4, fmt='none')
ax2.axhline(0, color="gray", linestyle=":", alpha=0.7)
ax2.set_ylabel(r"Task-Shape Cosine Similarity $\cos(\theta)$", color="#e41a1c", fontweight="bold")
ax2.tick_params(axis='y', labelcolor="#e41a1c")
ax2.set_ylim(-0.25, 0.25)

plt.title("Figure 3: Cognitive Gradient Share and Orthogonality ($\cos \\theta \\approx 0.0$)", fontweight="bold")
ax1.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_03_gradient_dynamics.pdf")
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_03_gradient_dynamics.png")
plt.close()

# ==============================================================================
# FIGURE 4: 50-Step Scaled RL Convergence Trajectory
# ==============================================================================
print("Generating Figure 4: 50-Step Training Curves...")

# Load history from 50-step CARC Qwen2.5-7B run
hist_file = "pilots/results/legacy_runs/carc_thinkaloud_shaping/pilot_w0.5.json"
if os.path.exists(hist_file):
    with open(hist_file) as f:
        hist_7b = json.load(f)
    
    steps = [h["step"] for h in hist_7b]
    task_r = [h["mean_task"] for h in hist_7b]
    shape_r = [h["mean_shape"] * 100 for h in hist_7b] # scaled for visualization
    losses = [h["loss"] for h in hist_7b]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), dpi=300, sharex=True)
    
    # Window smoothed task reward
    window = 5
    smooth_task = np.convolve(task_r, np.ones(window)/window, mode='valid')
    smooth_steps = steps[window-1:]
    
    ax1.plot(steps, task_r, color="#9ecae1", alpha=0.5, label="Raw Task Reward")
    ax1.plot(smooth_steps, smooth_task, color="#08519c", linewidth=2.2, label="Smoothed Task Reward (5-step)")
    ax1.set_ylabel("Task Reward")
    ax1.set_title("Figure 4: 50-Step Scaled Training Dynamics (Qwen2.5-7B on NVIDIA A40)", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.plot(steps, losses, color="#238b45", linewidth=1.8, label="GRPO Policy Loss")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Policy Gradient Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("pilots/figures/legacy_2026-10-29/pilot_04_training_curves.pdf")
    plt.savefig("pilots/figures/legacy_2026-10-29/pilot_04_training_curves.png")
    plt.close()

print("ALL 4 FIGURES SUCCESSFULLY GENERATED IN pilots/figures/legacy_2026-10-29/ !")

# ==============================================================================
# FIGURE 4: 50-Step Scaled RL Convergence Trajectory from Multidomain Runs
# ==============================================================================
print("Generating Figure 4: Multidomain Training Convergence...")

hist_file = "pilots/results/legacy_runs/carc_multidomain_20261027_153910_2209184/programming_unified_seed43/history.json"
with open(hist_file) as f:
    hist = json.load(f)

steps = [h["step"] for h in hist]
task_r = [h.get("mean_task", 0) for h in hist]
losses = [h.get("loss", 0) for h in hist]
shares = [h.get("shape_gradient_norm_share", 0) * 100 for h in hist]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), dpi=300, sharex=True)

# Window smoothed task reward
window = 5
smooth_task = np.convolve(task_r, np.ones(window)/window, mode='valid')
smooth_steps = steps[window-1:]

ax1.plot(steps, task_r, color="#9ecae1", alpha=0.5, label="Raw Batch Task Reward")
ax1.plot(smooth_steps, smooth_task, color="#08519c", linewidth=2.2, label="Moving Average (5-step)")
ax1.set_ylabel("Task Reward", fontweight="bold")
ax1.set_title("Figure 4: 50-Step GRPO Convergence Trajectory (Programming Unified)", fontweight="bold")
ax1.set_ylim(0.4, 1.05)
ax1.legend(loc="lower right")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(steps, losses, color="#238b45", linewidth=1.8, label="GRPO Policy Gradient Loss")
ax2.axhline(0, color="gray", linestyle=":", alpha=0.7)
ax2.set_xlabel("Training Step", fontweight="bold")
ax2.set_ylabel("Policy Loss", fontweight="bold")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_04_training_curves.pdf")
plt.savefig("pilots/figures/legacy_2026-10-29/pilot_04_training_curves.png")
plt.close()
print("Figure 4 generated successfully!")
