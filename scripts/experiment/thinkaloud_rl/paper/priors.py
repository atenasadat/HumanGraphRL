"""Build controls from session sequences; never shuffle across participants."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random

from .common import ROOT, atomic_json, digest, read_jsonl
from ..prepare_multidomain import blueprint
from ..reasoning_shaper import PURE_MOVES, classify_reasoning_step, segment_thought_stream

VARIANTS = ('human', 'shuffled', 'uniform', 'frequency', 'transition', 'phase_free', 'synthetic')


def sessions():
    from .human_sources import session_texts
    return {domain:{pid:[classify_reasoning_step(s) for text in rows for s in segment_thought_stream(text)]
                    for pid,rows in people.items()} for domain,people in session_texts().items()}


def make_prior(groups, variant, seed=2026, synthetic=None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    groups = copy.deepcopy(groups)
    if variant == 'synthetic':
        if not synthetic:
            raise ValueError('Synthetic control requires independently generated traces')
        groups = synthetic
    if variant == 'shuffled':
        rng = random.Random(seed)
        for moves in groups:
            rng.shuffle(moves)
    bp = blueprint([[{'move': m} for m in seq] for seq in groups], variant)
    # Identical Dirichlet pseudocounts for every data-derived condition.
    for counts_name, probs_name in [('phase_counts', 'phase_distributions'),
                                    ('transition_counts', 'transition_matrix')]:
        for row, counts in bp[counts_name].items():
            total = sum(counts.values()) + .5 * len(PURE_MOVES)
            bp[probs_name][row] = {m: (counts[m]+.5)/total for m in PURE_MOVES}
    marginal = {m: (bp['move_counts'][m]+.5)/(sum(bp['move_counts'].values())+2) for m in PURE_MOVES}
    if variant == 'uniform':
        for name in ('phase_distributions', 'transition_matrix'):
            bp[name] = {row: {m: .25 for m in PURE_MOVES} for row in bp[name]}
    if variant in ('frequency', 'phase_free'):
        bp['phase_distributions'] = {p: marginal.copy() for p in ('early','mid','late')}
    bp['potential_alpha'] = 0. if variant == 'transition' else .5
    bp['potential_beta'] = 0. if variant == 'frequency' else .5
    bp['metadata'].update(variant=variant, control_seed=seed, pseudocount=.5,
                          sequence_hash=digest(groups), label_status='unvalidated_regex')
    return bp


def build(out, held_out=(), synthetic_path=None, seed=2026):
    data = sessions()
    for domain in data:
        data[domain] = {pid: seq for pid, seq in data[domain].items() if pid not in held_out}
        if not data[domain]:
            raise ValueError('No remaining human sessions for ' + domain)
    pool = {d: list(g.values()) for d,g in data.items()}
    pool['unified'] = pool['programming'] + pool['writing']
    synth = defaultdict(list)
    if synthetic_path:
        for r in read_jsonl(synthetic_path):
            if r.get('split') != 'train':
                raise ValueError('Synthetic priors must use only training prompts')
            moves = [classify_reasoning_step(u) for u in segment_thought_stream(r['text'])]
            synth[r['domain']].append(moves)
        synth['unified'] = synth['programming'] + synth['writing']
    manifest = {'held_out_participants': list(held_out), 'participants': {d: list(g) for d,g in data.items()},
                'label_status': 'unvalidated_regex', 'priors': {}}
    for domain, groups in pool.items():
        for variant in VARIANTS:
            if variant == 'synthetic' and not synth[domain]:
                continue
            bp = make_prior(groups, variant, seed, synth[domain])
            name = f'{domain}_{variant}.json'
            atomic_json(Path(out)/name, bp)
            manifest['priors'][name] = digest(bp)
    atomic_json(Path(out)/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hold-out', nargs='*', default=[])
    p.add_argument('--synthetic', type=Path)
    a = p.parse_args()
    build(a.output, a.hold_out, a.synthetic)
