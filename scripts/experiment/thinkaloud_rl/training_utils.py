"""Causal potential control variates and single-update on-policy GRPO utilities."""
import math
import re
import torch
from scripts.experiment.thinkaloud_rl.reasoning_shaper import (
    classify_reasoning_step, segment_thought_stream, compute_move_potential, get_session_phase,
)


def causal_potentials(tokenizer, token_ids, blueprint, horizon):
    """Phi before each action; only its observed prefix and fixed budget are used.

    The final element is Phi(terminal)=0 even at a generation-budget truncation.
    Realized response length is never used to choose the phase of a prefix.
    """
    values = [0.]
    for t in range(1, len(token_ids)):
        prefix = tokenizer.decode(token_ids[:t], skip_special_tokens=True)
        match = re.search(r'<think>(.*?)(?:</think>|$)', prefix, re.S)
        moves = [classify_reasoning_step(u) for u in segment_thought_stream(match[1])] if match else []
        values.append(compute_move_potential(moves, horizon, blueprint=blueprint,
                     phase_override=get_session_phase(t, horizon)))
    return values+[0.] if token_ids else [0.]


def potential_rewards(potentials, gamma=1.):
    if not 0 < gamma <= 1: raise ValueError('gamma must be in (0,1]')
    return [gamma*b-a for a,b in zip(potentials,potentials[1:])]


def discounted_sum(rewards, gamma=1.):
    return math.fsum(gamma**t*r for t,r in enumerate(rewards))


def normalized_advantages(rewards):
    values = torch.tensor(rewards,dtype=torch.float32)
    return (values-values.mean())/values.std(unbiased=False).clamp_min(1e-8)


def policy_loss(log_probs, advantages, horizon, clip=.2):
    # One on-policy update per fresh group. Ratio initially equals one, but has
    # the correct gradient. Additional optimization epochs would need saved old logprobs.
    ratio = (log_probs-log_probs.detach()).exp()
    objective = torch.minimum(ratio*advantages, ratio.clamp(1-clip,1+clip)*advantages)
    # Fixed divisor: dividing by realized length would bias the control variate.
    return -objective.sum()/horizon


def grad_norm(grads):
    return math.sqrt(sum(float(g.float().square().sum()) for g in grads))


def calibrated_scale(task_norm, shape_norm, target=.35, cap=10.):
    """For the NEXT group only; never select a potential from future outcomes."""
    if not 0 < target < 1: raise ValueError('target must lie in (0,1)')
    if task_norm <= 1e-12 or shape_norm <= 1e-12: return 0.
    return min(cap, target/(1-target)*task_norm/shape_norm)
