import json
import glob
import math
import sys
import os
from collections import Counter, defaultdict

# Ensure scripts/experiment package resolves
sys.path.insert(0, "/Users/athenasaghi/VSProjects/TopGRPO")

from scripts.experiment.thinkaloud_rl.reasoning_shaper import (
    classify_reasoning_step,
    compute_move_potential,
    HUMAN_PHASE_DISTRIBUTIONS,
    HUMAN_TRANSITION_MATRIX,
    PURE_MOVES,
    get_session_phase,
)

def run_model_audit():
    model_dirs = {
        "Llama-3.1-8B": "/Users/athenasaghi/VSProjects/TopGRPO/pilots/results/legacy_runs/llm_graphs/llama3.1_8b/*.json",
        "Qwen3-8B": "/Users/athenasaghi/VSProjects/TopGRPO/pilots/results/legacy_runs/llm_graphs/qwen3_8b/*.json",
        "Mistral-7B": "/Users/athenasaghi/VSProjects/TopGRPO/pilots/results/legacy_runs/llm_graphs/mistral_7b/*.json",
        "DeepSeek-R1-1.5B": "/Users/athenasaghi/VSProjects/TopGRPO/pilots/results/legacy_runs/llm_graphs/deepseek-r1_1.5b/*.json",
    }
    
    results = {}
    
    for model_name, pattern in model_dirs.items():
        files = sorted(glob.glob(pattern))
        if not files:
            continue
            
        phase_counts = defaultdict(Counter)
        overall_counts = Counter()
        transitions = defaultdict(Counter)
        potentials_all = []
        valid_files = 0
        
        for fpath in files:
            with open(fpath) as f:
                d = json.load(f)
            if not isinstance(d, dict):
                continue
            graph = d.get("graph")
            if not isinstance(graph, dict):
                continue
            nodes = graph.get("nodes", [])
            utterances = [n.get("utterance", "") for n in nodes if isinstance(n, dict) and n.get("utterance")]
            if not utterances:
                continue
                
            valid_files += 1
            classified_moves = [classify_reasoning_step(u) for u in utterances]
            n_moves = len(classified_moves)
            
            # Record potentials
            for i in range(1, n_moves + 1):
                potentials_all.append(compute_move_potential(classified_moves[:i], n_moves))
                
            # Record transitions
            for a, b in zip(classified_moves, classified_moves[1:]):
                transitions[a][b] += 1
                
            # Record phases
            for idx, m in enumerate(classified_moves):
                ph = get_session_phase(idx, n_moves)
                phase_counts[ph][m] += 1
                overall_counts[m] += 1
                
        tot_moves = sum(overall_counts.values())
        if tot_moves == 0:
            continue
            
        # Overall distribution
        dist = {m: round(overall_counts[m] / tot_moves, 4) for m in PURE_MOVES}
        
        # Phase distributions & KL divergence
        phase_dists = {}
        kl_divs = {}
        for ph in ["early", "mid", "late"]:
            tot_ph = sum(phase_counts[ph].values())
            p_ph = {m: round(phase_counts[ph][m] / tot_ph, 4) if tot_ph > 0 else 0.0 for m in PURE_MOVES}
            phase_dists[ph] = p_ph
            
            # KL divergence from human phase prior
            target_pi = HUMAN_PHASE_DISTRIBUTIONS[ph]
            kl = 0.0
            for m in PURE_MOVES:
                q = p_ph[m]
                p = target_pi[m]
                if q > 0:
                    kl += q * math.log(q / p)
            kl_divs[ph] = round(kl, 4)
            
        results[model_name] = {
            "num_traces": valid_files,
            "total_nodes": tot_moves,
            "overall_distribution": dist,
            "phase_distributions": phase_dists,
            "kl_divergence_from_human": kl_divs,
            "mean_kl": round(sum(kl_divs.values()) / len(kl_divs), 4),
            "mean_potential": round(sum(potentials_all) / len(potentials_all), 4) if potentials_all else 0.0,
            "transitions": {
                m1: {m2: round(transitions[m1][m2] / sum(transitions[m1].values()), 4) if sum(transitions[m1].values()) > 0 else 0.0
                     for m2 in PURE_MOVES}
                for m1 in PURE_MOVES
            }
        }
        
    out_file = "/Users/athenasaghi/.gemini/antigravity/brain/7446f207-ac8f-41c0-a47c-125d1ee94a11/scratch/model_cognitive_audit.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Audit completed across {len(results)} models. Results saved to {out_file}.")
    return results

if __name__ == "__main__":
    run_model_audit()
