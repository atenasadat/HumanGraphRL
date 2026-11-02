"""Pure cognitive reasoning verifier and potential-based reward shaper.

Uses the empirical human thinking blueprint (derived from 580 think-aloud segments)
to evaluate and shape LLM reasoning traces over the 4 intrinsic cognitive moves:
    explore, evaluate, backtrack, commit
"""
import math
import json
import re
from typing import List, Dict, Tuple, Optional

# The 4 pure internal cognitive reasoning moves
PURE_MOVES = ("explore", "evaluate", "backtrack", "commit")

# Empirical human distributions derived from 580 think-aloud segments
HUMAN_PHASE_DISTRIBUTIONS = {
    "early": {
        "evaluate": 0.5417,
        "commit": 0.2639,
        "backtrack": 0.1111,
        "explore": 0.0833,
    },
    "mid": {
        "evaluate": 0.7073,
        "backtrack": 0.1951,
        "commit": 0.0732,
        "explore": 0.0244,
    },
    "late": {
        "evaluate": 0.5278,
        "backtrack": 0.2500,
        "commit": 0.1389,
        "explore": 0.0833,
    },
}

# Empirical human transition matrix: P(m_{t+1} | m_t)
HUMAN_TRANSITION_MATRIX = {
    "explore": {
        "evaluate": 0.6923,
        "backtrack": 0.1538,
        "commit": 0.1538,
        "explore": 0.0000,
    },
    "evaluate": {
        "evaluate": 0.7029,
        "backtrack": 0.1620,
        "commit": 0.0775,
        "explore": 0.0577,
    },
    "backtrack": {
        "evaluate": 0.5250,
        "backtrack": 0.2750,
        "commit": 0.1500,
        "explore": 0.0500,
    },
    "commit": {
        "commit": 0.3529,
        "evaluate": 0.3235,
        "backtrack": 0.2647,
        "explore": 0.0588,
    },
}


def segment_thought_stream(text: str) -> List[str]:
    """Segment a continuous thought stream into granular thought units."""
    if not text or not text.strip():
        return []
    
    # Strip <think> and </think> tags if present
    cleaned = re.sub(r"</?think>", " ", text).strip()
    
    # Split by explicit step markers, newlines, or sentence boundaries
    raw_units = re.split(r"(?:\n\s*\n|\n(?=[0-9]+[\.\)]|\*|-|\bStep\b)|\.\s+(?=[A-Z]))", cleaned)
    
    units = []
    for u in raw_units:
        u_str = u.strip()
        if len(u_str) >= 10:  # Skip trivial fragments
            units.append(u_str)
            
    # Fallback to lines if no units detected
    if not units and cleaned:
        units = [line.strip() for line in cleaned.splitlines() if len(line.strip()) >= 10]
        
    return units if units else [cleaned]


def classify_reasoning_step(step_text: str) -> str:
    """Classify an individual reasoning step into the 4 pure cognitive moves.
    
    Uses identical category definitions and pattern hierarchies established
    for the human verbalization data.
    """
    t = step_text.lower().strip().replace('’', "'")
    
    # 1. Backtrack: explicit failures, contradictions, dead-ends, realizing errors
    bt_patterns = [
        r"\b(?:delete|remove|scrap|retract|abandon) (?:this|that|the paragraph|the section|the idea|my claim)\b",
        r"\b(?:need to rethink|overstated|too broad|instead of claiming|revise my (?:claim|approach))\b",
        r"\b(didn't work|doesn't work|won't work|not working|not right)\b",
        r"\b(that's not what|not what we want|not what i want)\b",
        r"\b(that's wrong|that was wrong|my mistake|i was wrong|error|wait, no)\b",
        r"\b(dead[- ]end|broke|broken|cannot do this|can't get this|incorrect)\b",
        r"\b(revert|undo|start over|switch approach|try another way|alternatively no)\b",
        r"\b(contradiction|fails for|counterexample|violates)\b",
    ]
    for p in bt_patterns:
        if re.search(p, t):
            # Exclude hypothetical checks ("if this is wrong")
            if not re.search(r"\b(what if|check if|verify whether|inspect)\b", t):
                return "backtrack"

    # 2. Commit: locking in a concrete decision, choosing a formula, finalizing step
    commit_patterns = [
        r"\b(?:finalize|finalise|settled on|my thesis is|the final argument is|i'm done)\b",
        r"\b(i'll use|i will use|we will use|let's use|going to use|let's go with)\b",
        r"\b(decided to|let's implement|so i'll|so we'll|moving forward with|implement it)\b",
        r"\b(therefore, we set|thus we have|so the answer is|this gives the final)\b",
        r"\b(that's settled|this is the plan|we conclude that|let's choose|choose option|we choose)\b",
    ]
    for p in commit_patterns:
        if re.search(p, t):
            return "commit"

    # 3. Explore: open-ended hypothesis generation, considering alternative angles
    explore_patterns = [
        r"\b(what if|maybe we can|wondering if|wonder if|could try|perhaps)\b",
        r"\b(thinking about|considering|another option|alternative|how should we)\b",
        r"\b(brainstorm|hypothesize|possible approaches|could be either)\b",
    ]
    for p in explore_patterns:
        if re.search(p, t):
            return "explore"

    # 4. Evaluate: default analytical workhorse (checking, inspecting, calculation)
    # Most reasoning steps in math/programming are active verification and calculation
    return "evaluate"


def get_session_phase(step_index: int, total_steps: int) -> str:
    """Determine temporal phase (early, mid, late) for a given step index."""
    if total_steps <= 1:
        return "mid"
    frac = step_index / max(total_steps - 1, 1)
    if frac < 0.33:
        return "early"
    elif frac < 0.66:
        return "mid"
    return "late"


def compute_move_potential(
    move_prefix: List[str],
    total_steps: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1e-6,
    blueprint: Optional[Dict] = None,
    scale: float = 1.0,
    phase_override: Optional[str] = None,
) -> float:
    """Compute shaping potential Phi(s_t) for a prefix of classified moves.
    
    Phi(s_t) measures alignment with human phase move distribution and transition priors.
    Boundary conditions: Phi(empty) = 0.0, and terminal potential handles final state.
    """
    if not move_prefix:
        return 0.0
    
    t = len(move_prefix) - 1
    current_phase = phase_override or get_session_phase(t, total_steps)
    target_pi = (blueprint['phase_distributions'] if blueprint else HUMAN_PHASE_DISTRIBUTIONS)[current_phase]
    
    # 1. Empirical distribution over prefix (with Laplace smoothing)
    counts = {m: eps for m in PURE_MOVES}
    for m in move_prefix:
        if m in counts:
            counts[m] += 1.0
    total_counts = sum(counts.values())
    q_empirical = {m: counts[m] / total_counts for m in PURE_MOVES}
    
    # 2. Phase-divergence penalty: D_KL(q_empirical || target_pi)
    kl = 0.0
    for m in PURE_MOVES:
        q_m = q_empirical[m]
        p_m = max(target_pi.get(m, eps), eps)
        if q_m > 0:
            kl += q_m * math.log(q_m / p_m)
            
    # 3. Transition likelihood bonus
    trans_log_prob = 0.0
    if len(move_prefix) >= 2:
        prev_m, curr_m = move_prefix[-2], move_prefix[-1]
        matrix = blueprint['transition_matrix'] if blueprint else HUMAN_TRANSITION_MATRIX
        p_trans = matrix.get(prev_m, {}).get(curr_m, eps)
        trans_log_prob = math.log(max(p_trans, eps))
    else:
        trans_log_prob = 0.0

    potential = -alpha * kl + beta * trans_log_prob
    return float(scale * potential)


def compute_trajectory_shaping(
    moves: List[str],
    gamma: float = 0.99,
    alpha: float = 0.5,
    beta: float = 0.5,
    blueprint: Optional[Dict] = None,
    scale: float = 1.0,
) -> List[float]:
    """Compute step-level potential-based shaping rewards F_t = gamma * Phi_{t+1} - Phi_t.
    
    Ensures exact policy invariance by enforcing zero terminal potential.
    """
    if not 0 < gamma <= 1 or not math.isfinite(scale):
        raise ValueError('gamma must be in (0, 1] and scale must be finite')
    n = len(moves)
    if n == 0:
        return []
    
    # Potentials for each prefix t = 1 ... n
    potentials = [compute_move_potential(moves[:i], n, alpha=alpha, beta=beta, blueprint=blueprint, scale=scale) for i in range(1, n + 1)]
    
    # Terminal potential forced to 0 for finite trajectory invariance
    potentials[-1] = 0.0
    
    # Initial state potential Phi(s_0) = 0.0
    all_potentials = [0.0] + potentials
    
    shaping_rewards = [
        gamma * all_potentials[t + 1] - all_potentials[t]
        for t in range(n)
    ]
    return shaping_rewards


def load_blueprint(path):
    """Load the explicit four-move schema, rejecting malformed probability rows."""
    with open(path) as handle:
        data = json.load(handle)
    for key, rows in [('phase_distributions', ('early', 'mid', 'late')),
                      ('transition_matrix', PURE_MOVES)]:
        for row in rows:
            values = data[key][row]
            if set(values) != set(PURE_MOVES) or any(not math.isfinite(v) or v < 0 for v in values.values()) or not math.isclose(sum(values.values()), 1, abs_tol=1e-6):
                raise ValueError(f'Invalid blueprint probability row: {key}/{row}')
    return data
