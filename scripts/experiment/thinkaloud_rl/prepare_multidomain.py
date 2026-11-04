"""Shared move blueprints and divergence helpers for the human cognitive priors.

Labels are deterministic heuristic estimates, NOT validated human annotations.
Human text is loaded by paper/human_sources.py from the reviewed programming
cells and the manually cleaned writing transcripts (P*.full.txt).
"""
from collections import Counter
import math

from scripts.experiment.thinkaloud_rl.reasoning_shaper import PURE_MOVES

PHASES = ('early', 'mid', 'late')


def phase(index, count):
    fraction = index / max(count - 1, 1)
    return 'early' if fraction < .33 else 'mid' if fraction < .66 else 'late'


def normalize(counts):
    n = sum(counts.values())
    return {m: counts.get(m, 0) / n if n else .25 for m in PURE_MOVES}


def blueprint(sessions, description):
    pc = {p: Counter({m: 0 for m in PURE_MOVES}) for p in PHASES}
    tc = {m: Counter({n: 0 for n in PURE_MOVES}) for m in PURE_MOVES}
    counts = Counter({m: 0 for m in PURE_MOVES})
    for session in sessions:
        for i, event in enumerate(session):
            m = event['move']; counts[m] += 1
            pc[phase(i, len(session))][m] += 1
        for a, b in zip(session, session[1:]):
            tc[a['move']][b['move']] += 1
    return {'metadata': {'description': description, 'label_method': 'shared_regex_v1_unvalidated',
            'sessions': len(sessions), 'events': sum(counts.values()),
            'phase_basis': 'within-session thought index: <.33, <.66, otherwise late',
            'zero_row_policy': 'uniform fallback; inspect raw counts', 'moves': list(PURE_MOVES)},
            'phase_counts': pc, 'transition_counts': tc, 'move_counts': counts,
            'phase_distributions': {p: normalize(c) for p,c in pc.items()},
            'transition_matrix': {m: normalize(c) for m,c in tc.items()},
            'move_distribution': normalize(counts)}


def js(p, q):
    middle = {m: (p[m]+q[m])/2 for m in PURE_MOVES}
    return sum(.5*x[m]*math.log2(x[m]/middle[m]) for x in (p,q) for m in PURE_MOVES if x[m])


def compare(a, b):
    rows = {m: js(a['transition_matrix'][m], b['transition_matrix'][m]) for m in PURE_MOVES}
    return {'transition_js_bits_macro': sum(rows.values())/4, 'transition_js_bits_per_row': rows,
            'definition': 'equal-weight mean of row-wise Jensen-Shannon divergence, base 2; not JS distance',
            'writing_minus_programming_move_probability': {m: b['move_distribution'][m]-a['move_distribution'][m] for m in PURE_MOVES}}
