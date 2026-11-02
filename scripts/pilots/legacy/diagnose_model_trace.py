"""Diagnostic demonstration of pure reasoning shaping on sample reasoning traces."""
import json
from scripts.experiment.thinkaloud_rl.reasoning_shaper import (
    segment_thought_stream,
    classify_reasoning_step,
    compute_move_potential,
    compute_trajectory_shaping
)

# 1. Healthy trace demonstrating early commit, continuous evaluation, and error-recovery backtracking
HEALTHY_TRACE = """
Let's consider how to implement the sorting function with external memory constraints.
What if we use a k-way external merge sort with a min-heap?
Therefore, we choose a 4-way merge sort using a priority queue.
Let's verify the memory bounds: each chunk is at most 10MB.
Now let's check the buffer allocation: 4 input buffers of 2MB and 1 output buffer of 2MB.
Wait, that is wrong because 4 * 2MB + 2MB = 10MB, which leaves zero memory for the min-heap overhead.
So we must revert the chunk buffer size to 1.5MB each.
Let's calculate again: 4 * 1.5MB + 1.5MB = 7.5MB, leaving 2.5MB for heap pointers and runtime overhead.
This satisfies all constraints.
Therefore, we implement the 1.5MB partition scheme.
"""

# 2. Pathological trace stuck in repetitive rumination / local verification loop without committing
PATHOLOGICAL_TRACE = """
Let's examine the memory constraints.
Let's check the memory limit again.
Let's verify if 10MB is enough.
Let's inspect the buffer size.
Let's check the buffer size once more.
Let's inspect the memory bounds again.
Let's verify the numbers again.
Let's re-verify the allocation.
Let's check the 10MB limit again.
Let's verify if the buffer fits.
"""

def evaluate_sample(name: str, text: str):
    units = segment_thought_stream(text)
    moves = [classify_reasoning_step(u) for u in units]
    n = len(moves)
    potentials = [compute_move_potential(moves[:i], n) for i in range(1, n + 1)]
    shaping = compute_trajectory_shaping(moves, gamma=0.99)
    
    print(f"=== {name} (Total Steps: {n}) ===")
    print(f"Move Sequence: {' -> '.join(moves)}")
    print(f"Mean Prefix Potential Phi: {sum(potentials)/len(potentials):.4f}")
    print(f"Step Shaping Rewards (First 3): {[round(r, 4) for r in shaping[:3]]}")
    print(f"Discounted Return of Shaping: {sum(0.99**t * r for t, r in enumerate(shaping)):.6f} (exact zero invariant)")
    print()

if __name__ == "__main__":
    evaluate_sample("Healthy Human-Like Reasoner", HEALTHY_TRACE)
    evaluate_sample("Pathological Rumination Reasoner", PATHOLOGICAL_TRACE)
