"""Reproducible, auditable writing ingestion and event-pooled cognitive priors.

Labels are deterministic heuristic estimates, NOT validated human annotations.
Only source transcript arrays are ingested (never duplicate full/part exports).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from scripts.experiment.thinkaloud_rl.reasoning_shaper import PURE_MOVES, classify_reasoning_step, segment_thought_stream

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'processed_data').is_dir())
HERE = ROOT/'processed_data/cognitive_priors_and_audits_from_transcripts'
PHASES = ('early', 'mid', 'late')
# Reviewed against raw turns; boundaries are exclusive. Explicit exceptions to
# the start phrase accommodate ASR and actual facilitator wording, not guesses.
REVIEWED = {
 'P4': (76, 451, 'you can just start; we can start the interview now'),
 'P5': (12, 300, "you’re gonna start now; now we’re gonna move to the interview"),
 'P6': (32, 397, 'you can basically just start; participant I think I’m done'),
 'P7': (29, 341, 'you can start; participant I think I’m done'),
 'P9': (8, 649, 'can you start; we can move to the interview'),
 'P10': (9, 445, 'after that you can start; let me move the interview part; turn 418 is prospective'),
 'P11': (23, 508, 'you can restart (ASR); press done, we can move to the interview; turn 490 is conditional'),
 'P12': (23, 542, 'I’ll start, yeah (ASR), followed by task reading; turn 478 is prospective'),
 'P13': (23, 334, 'you can start; we can move to the interview part now'),
 'P14': (11, 319, 'you’re gonna start; first I’m done (with the draft): strict requested cutoff, revision excluded'),
 'P15': (16, 448, 'you can start; gonna move to the interview part now'),
 'P16': (4, 309, 'you can start now; I guess I’m done'),
 'P17': (12, 356, 'can start; now we’re gonna move to the interview'),
}
START = re.compile(r"\b(?:you\s+can\s+(?:(?:basically|just)\s+)*start(?:\s+now)?|you['’]re\s+gonna\s+start(?:\s+now)?)\b", re.I)
END = re.compile(r"\b(?:let['’]s\s+move\s+to\s+the\s+interview|i\s*['’]?\s*m\s+done)\b", re.I)
CHAT = re.compile(r'^(?:okay|ok|yeah|yes|no|right|alright|thank you|thanks|sorry|oh|hmm|um|uh|great|got it|sounds good|yep|sure)[\s,.!?…]*$', re.I)
TECH = re.compile(r'\b(?:can you (?:see|hear) (?:me|my screen)|share (?:my|your) screen|subject id|zoom chat|vpn|wi-?fi|microphone permission)\b', re.I)


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


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def ingest(rows, start, end, participant_speaker):
    kept, dropped = [], []
    for pos, row in enumerate(rows):
        reason = None
        if not start < pos < end: reason = 'outside_task_boundaries'
        elif row['speaker'] != participant_speaker: reason = 'instructor'
        elif CHAT.fullmatch(row['text'].strip()) or TECH.search(row['text']): reason = 'chitchat_or_setup'
        if reason:
            dropped.append({'source_turn': pos, 'reason': reason}); continue
        for thought in segment_thought_stream(row['text']):
            if len(thought.split()) < 3: continue
            kept.append({'turn_index': len(kept), 'source_turn': pos, 'source_id': row.get('id'),
                         'start_ms': row['startTime'], 'end_ms': row['endTime'],
                         'text': thought, 'move': classify_reasoning_step(thought),
                         'label_method': 'shared_regex_v1_unvalidated'})
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=ROOT/'data/think-aloud/writing_transcripts')
    ap.add_argument('--output', type=Path, default=ROOT/'processed_data/writing_task_segments_extracted_from_transcripts')
    args = ap.parse_args()
    manifest = {m['participant']: m for m in json.loads((args.source/'manifest.json').read_text())}
    writing, audits = [], []
    for source in sorted(args.source.glob('*.transcript.json'), key=lambda p: int(p.name.split('.')[0][1:])):
        pid = source.name.split('.')[0]
        rows = json.loads(source.read_text())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != json.loads((HERE/'proposal_source_hashes.json').read_text())[source.name]:
            raise ValueError(f'{pid}: source changed; re-review boundary indices')
        aliases = manifest[pid]['speaker_aliases']
        speakers = [name for name, alias in aliases.items() if alias == 'S2']
        if pid not in REVIEWED or len(speakers) != 1:
            audit = {'participant': pid, 'status': 'excluded', 'reason': 'unreliable speaker diarization or unreviewed boundaries', 'sha256': digest}
            audits.append(audit)
            dump(args.output/f'{pid}_proposal_verbalizations.json', {'metadata': audit, 'segments': []})
            continue
        start, end, note = REVIEWED[pid]
        segments, dropped = ingest(rows, start, end, speakers[0])
        audit = {'participant': pid, 'status': 'included_provisional', 'sha256': digest,
                 'start_turn_exclusive': start, 'end_turn_exclusive': end,
                 'start_text': rows[start]['text'], 'end_text': rows[end]['text'], 'boundary_review_note': note,
                 'segments': len(segments), 'excluded_turn_counts': dict(Counter(d['reason'] for d in dropped)),
                 'limitations': 'ASR, heuristic labels and chit-chat filtering require human validation; timestamps retained at source-turn resolution.'}
        audits.append(audit); writing.append(segments)
        dump(args.output/f'{pid}_proposal_verbalizations.json', {'metadata': audit, 'segments': segments, 'excluded_turns': dropped})
    # Shared classifier avoids conflating domain differences with Gemini-vs-regex labels.
    groups = defaultdict(list)
    for r in json.loads((HERE/'verbalizations_extracted_segments.json').read_text()): groups[r['participant']].append(r)
    programming = []
    for pid, rows in sorted(groups.items()):
        session = []
        for row in sorted(rows, key=lambda r:r['seq']):
            for thought in segment_thought_stream(row['text']):
                session.append({'source_id': row['id'], 'text': thought, 'move': classify_reasoning_step(thought)})
        programming.append(session)
    pb = blueprint(programming, 'Programming source segments reclassified with shared writing classifier')
    wb = blueprint(writing, 'Boundary-filtered proposal-writing transcripts; P8 excluded')
    ub = blueprint(programming+writing, 'Event-count pooled programming and writing; transitions never cross sessions/domains')
    for name,obj in [('programming_cognitive_blueprint.json',pb),('proposal_cognitive_blueprint.json',wb),('unified_cognitive_blueprint.json',ub)]: dump(HERE/name,obj)
    old = json.loads((HERE/'human_cognitive_blueprint.json').read_text())
    legacy = {'transition_matrix': {m: normalize({n: old['transition_matrix']['active_moves_only'][m].get(n,0) for n in PURE_MOVES}) for m in PURE_MOVES}, 'move_distribution': pb['move_distribution']}
    report = {'status': 'descriptive_estimates_not_evidence_of_generalization',
              'shared_classifier_comparison': compare(pb,wb),
              'requested_legacy_transition_comparison': {k:v for k,v in compare(legacy,wb).items() if k != 'writing_minus_programming_move_probability'},
              'legacy_caveat': 'Legacy 5-move rows projected onto four moves and renormalized; label methods differ. Original human blueprint preserved.',
              'programming_events': pb['metadata']['events'], 'writing_events': wb['metadata']['events'],
              'limitations': ['No inter-rater validation', 'Programming source includes narration and tool interaction; shared regex defaults to evaluate', 'Unequal domain event counts; pooled prior is event-weighted', 'P14 truncated at first I’m done, as requested', 'No causal or task-success conclusions from these counts'],
              'sessions': audits}
    dump(args.output/'ingestion_audit.json', report)
    dump(HERE/'cross_domain_divergence.json', {k:v for k,v in report.items() if k != 'sessions'})
    print(json.dumps({k:report[k] for k in ('programming_events','writing_events','shared_classifier_comparison')},indent=2))

if __name__ == '__main__': main()
