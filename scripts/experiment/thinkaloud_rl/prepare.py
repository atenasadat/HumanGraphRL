"""Source-linked preparation of writing-study data. No inferred reward labels.

python3 -m scripts.experiment.thinkaloud_rl.prepare build --output pilots/results/legacy_runs/thinkaloud_training_v1
python3 -m scripts.experiment.thinkaloud_rl.prepare check --bundle pilots/results/legacy_runs/thinkaloud_training_v1

The check command exits 2 when scientifically required supervision is missing.
Build is useful even when check refuses training. Standard library only.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from .core import SnapshotDecoder
from .inventory import EVENT, PHASES
from .paths import ROOT, cloud_only

VERSION = 1
DOC_EVENTS = {'writing.document_snapshot', 'writing.final_response_snapshot'}


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def emit(f, row):
    f.write(canonical(row) + '\n')


def record(kind, participant, **fields):
    r = dict(schema_version=VERSION, kind=kind, participant=participant, **fields)
    r['record_id'] = digest(canonical(r))
    return r


def verify_record(r):
    body = {k: v for k, v in r.items() if k != 'record_id'}
    if r.get('record_id') != digest(canonical(body)):
        raise ValueError('Record content hash mismatch')


def checked_source(path, expected=None):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed during read: {path}')
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f'Source differs from audited inventory: {path}')
    return raw, dict(path=str(path.resolve()), sha256=sha, bytes=len(raw))


def phase_for(start, end, phases):
    """Require one entire interval in exactly one annotated task window.

    An overlap with any other phase is ambiguous, including partial overlap.
    No clipping of speech or inferred phase boundaries.
    """
    if not finite(start) or not finite(end) or start < 0 or end <= start:
        return None
    overlaps = [p for p in phases if min(end, p['endTime']) > max(start, p['startTime'])]
    if len(overlaps) != 1:
        return None
    p = overlaps[0]
    return p['codeName'] if p['startTime'] <= start and end <= p['endTime'] else None


def annotation_records(pid, annotation, source):
    phases = [a for a in annotation['annotations']
              if a['tierId'] == 'feature' and a['codeName'] in PHASES]
    for p in phases:
        if not finite(p['startTime']) or not finite(p['endTime']) or p['endTime'] <= p['startTime']:
            raise ValueError(f'Invalid phase: {pid}')
    accepted, excluded, moves = [], [], []
    seen = set()
    for i, t in enumerate(annotation['transcript']):
        if t['index'] in seen:
            raise ValueError(f'Duplicate transcript index: {pid}')
        seen.add(t['index'])
        phase = phase_for(t['startTime'], t['endTime'], phases)
        r = record('transcript_segment', pid, source=source, source_pointer=f'/transcript/{i}',
                   segment_index=t['index'], start_ms=t['startTime'], end_ms=t['endTime'],
                   clock='annotation_media', text=t['text'], speaker_track=t.get('speaker'),
                   speaker_role=None, utterance_origin=None, phase=phase,
                   exclusion=None if phase else 'outside_task_or_crosses_or_overlaps_phase',
                   quality_label=None, training_eligible=False)
        (accepted if phase else excluded).append(r)
    for i, a in enumerate(annotation['annotations']):
        if a['tierId'] != 'reasoning':
            continue
        moves.append(record('existing_process_annotation', pid, source=source,
                            source_pointer=f'/annotations/{i}', annotation=a,
                            phase=phase_for(a['startTime'], a['endTime'], phases),
                            meaning='descriptive activity; not step quality',
                            training_eligible=False))
    return accepted, excluded, moves, phases


def alignment_report(pid, annotation, paths):
    """Audit recorded nudge anchors; never declare independent synchronization."""
    if not paths:
        return dict(participant=pid, status='missing_raw_anchor_log', matched=0,
                    independently_validated=False), []
    # Caller chooses a single local-preferred log; do not merge duplicate copies.
    path = paths[0]
    raw, src = checked_source(path)
    rows, issues, anchors = defaultdict(list), [], []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            r = json.loads(line)
            p, c = r['payload'], r['clock']
            rows[p['nudge_id']].append((number, p, c))
        except (ValueError, KeyError, TypeError):
            issues.append(dict(line=number, reason='invalid_anchor_record'))
    for a in annotation.get('nudgeCodings', []):
        candidates = [(n, p, c) for n, p, c in rows.get(a.get('nudge_id'), [])
                      if finite(c.get('session_time_ms')) and finite(a.get('raw_time_ms'))
                      and abs(c['session_time_ms'] - a['raw_time_ms']) <= .01
                      and p.get('delivered_text') == a.get('delivered_text')]
        if len(candidates) != 1 or not finite(a.get('aligned_time_ms')):
            issues.append(dict(nudge_id=a.get('nudge_id'), reason='missing_or_nonunique_exact_anchor'))
            continue
        n, p, c = candidates[0]
        if not finite(c.get('epoch_ms')):
            issues.append(dict(line=n, reason='missing_epoch'))
            continue
        anchors.append(dict(source_line=n, nudge_id=a['nudge_id'], epoch_ms=c['epoch_ms'],
                            session_ms=c['session_time_ms'], media_ms=a['aligned_time_ms'],
                            session_offset_ms=a['aligned_time_ms'] - c['session_time_ms'],
                            epoch_offset_ms=a['aligned_time_ms'] - c['epoch_ms']))
    offsets = [a['epoch_offset_ms'] for a in anchors]
    spread = max(offsets) - min(offsets) if offsets else None
    # 0.01 ms is solely a float serialization tolerance, not a validated sync error.
    consistent = len(anchors) >= 2 and spread <= .01 and not issues
    return dict(participant=pid, source=src, anchors=anchors, issues=issues,
                matched=len(anchors), annotation_anchors=len(annotation.get('nudgeCodings', [])),
                epoch_offset_spread_ms=spread,
                status='metadata_consistent' if consistent else 'needs_alignment_review',
                independently_validated=False,
                limitation='Repeated UI offsets are not independent synchronization measurements; no automatic join.'), [src]


def snapshot_clock(r):
    c = r.get('clock') or {}
    if finite(c.get('epoch_ms')):
        return 'epoch_ms', c['epoch_ms']
    if finite(r.get('t_s')):
        return 'event_session_ms', r['t_s'] * 1000
    return None, None


def scan_changes(path, pid, output, issue_file):
    """Keep observed adjacent changes, including A -> B -> A, within each log.

    No joins across logs, clock regressions, decode errors or missing bodies.
    The latest repeated observation becomes the next input, avoiding stale context.
    Snapshot intervals are not asserted to be atomic human reasoning actions.
    """
    path = Path(path)
    if cloud_only(path):
        raise ValueError(f'Large cloud log must be retrieved explicitly: {path}')
    before = path.stat()
    decoder, previous = SnapshotDecoder(), None
    counts, raw_hash = Counter(), hashlib.sha256()
    opener = gzip.open if path.suffix == '.gz' else open
    last_notice = time.monotonic()
    with opener(path, 'rb') as f:
        for number, line in enumerate(f, 1):
            raw_hash.update(line)
            counts['lines'] += 1
            match = EVENT.search(line[:4096])
            if match is None:
                counts['unclassified_lines'] += 1
                emit(issue_file, dict(participant=pid, path=str(path), line=number, reason='unclassified_line'))
                previous, decoder = None, SnapshotDecoder()
                continue
            kind = match.group(1).decode()
            # The nudges stream repeats the start as an audit notification,
            # sometimes after the first writing snapshot. It is not a reset.
            if kind == 'session_task_started':
                previous, decoder = None, SnapshotDecoder()
                counts['task_start_resets'] += 1
            if kind not in DOC_EVENTS:
                continue
            counts['document_events'] += 1
            try:
                r = json.loads(line)
                p = r.get('payload') or {}
                if p.get('text_mode'):
                    text = decoder.accept(p)
                else:
                    text = p.get('text')
                    if not isinstance(text, str):
                        raise ValueError('missing_document_body')
                    if p.get('character_count') != len(text):
                        raise ValueError('character_count_mismatch')
                clock, timestamp = snapshot_clock(r)
                if clock is None or timestamp < 0:
                    raise ValueError('missing_or_invalid_capture_clock')
            except (ValueError, TypeError, KeyError) as e:
                counts['rejected_document_events'] += 1
                emit(issue_file, dict(participant=pid, path=str(path), line=number, reason=str(e)))
                previous, decoder = None, SnapshotDecoder()
                continue
            counts['decoded_document_events'] += 1
            current = dict(source_line=number, text=text, text_sha256=digest(text), clock=clock,
                           time_ms=timestamp, recorded_phase=r.get('study_phase', r.get('phase')),
                           ai_insertion_metadata={k: p[k] for k in ('ai_insertions', 'ai_insertion_count') if k in p})
            if previous is not None:
                if (previous['clock'] == clock and timestamp == previous['time_ms']
                        and previous['text_sha256'] == current['text_sha256']):
                    counts['duplicate_capture_events'] += 1
                    continue
                if previous['clock'] != clock or timestamp <= previous['time_ms']:
                    counts['clock_breaks'] += 1
                    emit(issue_file, dict(participant=pid, path=str(path), line=number,
                                          reason='clock_changed_nonincreasing_or_reset'))
                    previous = None
                elif previous['text_sha256'] != current['text_sha256']:
                    emit(output, record('observed_document_change', pid, source_path=str(path),
                                        before=previous, after=current,
                                        action_semantics='document replacement between captures; actor and atomicity unknown',
                                        transcript_context=None, quality_label=None, training_eligible=False,
                                        required_reviews=['capture_to_media_alignment', 'task_phase',
                                                          'speaker_and_text_origin', 'step_quality_rubric']))
                    counts['candidate_changes'] += 1
                else:
                    counts['unchanged_observations'] += 1
            previous = current
            if time.monotonic() - last_notice > 20:
                print(f'{pid}: {counts["lines"]} lines scanned; {counts["candidate_changes"]} changes', flush=True)
                last_notice = time.monotonic()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed during scan: {path}')
    src = dict(path=str(path.resolve()), bytes=before.st_size,
               sha256=file_hash(path) if path.suffix == '.gz' else raw_hash.hexdigest())
    return dict(participant=pid, source=src, counts=dict(counts),
                uncompressed_sha256=raw_hash.hexdigest())


def participant_folds(participants):
    if len(set(participants)) != len(participants) or len(participants) < 3:
        raise ValueError('Need at least three distinct participants')
    return [dict(fold=f'heldout_{test}', test=[test], validation=[participants[(i + 1) % len(participants)]],
                 train=[p for p in participants if p not in {test, participants[(i + 1) % len(participants)]}])
            for i, test in enumerate(participants)]


def verify_folds(folds, participants):
    tests = []
    for fold in folds:
        groups = [set(fold[k]) for k in ('train', 'validation', 'test')]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('Participant leakage between splits')
        if set.union(*groups) != set(participants) or not all(groups):
            raise ValueError('Incomplete fold')
        tests.extend(fold['test'])
    if sorted(tests) != sorted(participants):
        raise ValueError('Every participant must be test exactly once')


def duplicate_audit(rows):
    groups = defaultdict(set)
    for r in rows:
        text = r.get('text', r.get('after', {}).get('text', ''))
        if text.strip():
            groups[digest(' '.join(text.split()))].add(r['participant'])
    return [dict(whitespace_normalized_sha256=h, participants=sorted(p))
            for h, p in sorted(groups.items()) if len(p) > 1]


def load_rows(path):
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            verify_record(r)
            yield r


def build(inventory_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite a prepared bundle; choose a new version directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.thinkaloud-prepare-', dir=output.parent))
    try:
        raw, inventory_source = checked_source(inventory_path)
        inventory = json.loads(raw)
        sources, counts, alignments, scans, participant_summary = [inventory_source], Counter(), [], [], []
        participants = [r['participant'] for r in inventory['participants']]
        folds = participant_folds(participants)
        verify_folds(folds, participants)
        filenames = ['task_segments.jsonl', 'excluded_segments.jsonl', 'process_annotations.jsonl',
                     'unsequenced_document_states.jsonl', 'candidate_changes.jsonl', 'issues.jsonl']
        handles = {n: (staging / n).open('w') for n in filenames}
        try:
            for r in inventory['participants']:
                pid = r['participant']
                print(f'Preparing {pid}', flush=True)
                raw, src = checked_source(r['annotation']['path'], r['annotation']['sha256'])
                sources.append(src)
                a = json.loads(raw)
                task, excluded, moves, phases = annotation_records(pid, a, src)
                for name, rows in [('task_segments', task), ('excluded_segments', excluded), ('process_annotations', moves)]:
                    for row in rows:
                        emit(handles[name + '.jsonl'], row)
                    counts[name] += len(rows)
                state = r['physical_action_history']
                raw, src = checked_source(state['path'], state['sha256'])
                sources.append(src)
                state_count = 0
                for i, node in enumerate(json.loads(raw).get('nodes', {}).get('physical_action', [])):
                    if not isinstance(node.get('full_code_snapshot'), str):
                        continue
                    emit(handles['unsequenced_document_states.jsonl'], record('unsequenced_graph_document', pid,
                         source=src, source_pointer=f'/nodes/physical_action/{i}', node_id=node.get('id'),
                         recorded_node_timestamp=node.get('timestamp'), text=node['full_code_snapshot'],
                         capture_time=None, training_eligible=False,
                         exclusion='graph nodes can be updated without updating their timestamp'))
                    state_count += 1
                counts['unsequenced_document_states'] += state_count
                anchor_paths = [Path(folder) / 'raised_nudges.jsonl' for folder in r['folders']
                                if (Path(folder) / 'raised_nudges.jsonl').exists()]
                report, extra_sources = alignment_report(pid, a, anchor_paths)
                alignments.append(report)
                sources.extend(extra_sources)
                pc = 0
                for log in r['logs']:
                    if log['status'] != 'fully_scanned':
                        continue
                    result = scan_changes(log['path'], pid, handles['candidate_changes.jsonl'], handles['issues.jsonl'])
                    scans.append(result)
                    sources.append(result['source'])
                    pc += result['counts'].get('candidate_changes', 0)
                participant_summary.append(dict(participant=pid, task_segments=len(task),
                     excluded_segments=len(excluded), process_spans=len(moves), graph_documents=state_count,
                     captured_change_candidates=pc, alignment=report['status'], trainable_reward_examples=0))
                counts['candidate_changes'] += pc
        finally:
            for f in handles.values():
                f.close()
        all_candidates = list(load_rows(staging / 'candidate_changes.jsonl'))
        duplicates = dict(task_segments=duplicate_audit(load_rows(staging / 'task_segments.jsonl')),
                          document_changes=duplicate_audit(all_candidates),
                          limitation='Exact normalized duplicates only; shared tasks and semantic overlap still require review.')
        # Templates are explicitly incomplete. No default numeric label or approval.
        with (staging / 'review_template.jsonl').open('w') as f:
            for r in all_candidates:
                emit(f, dict(candidate_record_id=r['record_id'], participant=r['participant'],
                             alignment_review=None, attribution_review=None, quality_review=None,
                             use='review template only; never training supervision'))
        report = dict(schema_version=VERSION, counts=dict(counts), participants=participant_summary,
                      alignment=alignments, event_scans=scans, training_ready=False,
                      blockers=['No independently validated capture-to-media alignment',
                                'No verified participant/AI/researcher attribution for training examples',
                                'No validated step-quality or final-quality reward labels',
                                'Descriptive process labels are sparse and not a standardized move vocabulary',
                                'No validated reward model for generated policy actions'],
                      operational_choices={'splits': '14 leave-one-participant-out folds; next ID validation; remaining train',
                           'split_status': 'exploratory; all participants previously inspected',
                           'trajectory_unit': 'adjacent captured document states; not atomic cognition',
                           'context_policy': 'no transcript-document joining until alignment validated',
                           'reward_policy': 'missing labels remain null; no fabricated negatives',
                           'graph_policy': 'unsequenced source material only',
                           'source_policy': 'read-only originals; no wholesale cloud log hydration'})
        for name, data in [('report.json', report), ('folds.json', folds), ('duplicate_audit.json', duplicates)]:
            (staging / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        (staging / 'REPORT.md').write_text(report_markdown(report))
        for code in sorted(Path(__file__).parent.glob('*.py')):
            sources.append(checked_source(code)[1])
        manifest = dict(schema_version=VERSION, sources=sources,
                        artifacts={p.name: file_hash(p) for p in sorted(staging.iterdir()) if p.is_file()})
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(output)
        return report
    except BaseException:
        shutil.rmtree(staging)
        raise


def report_markdown(r):
    lines = ['# Writing think-aloud training preparation', '',
             '**Preparation complete; PRM/RL training blocked by missing validated supervision.**', '',
             '| Writer | Task segments | Process spans | Graph documents | Captured changes |',
             '|---|---:|---:|---:|---:|']
    for p in r['participants']:
        lines.append(f'| {p["participant"]} | {p["task_segments"]} | {p["process_spans"]} | {p["graph_documents"]} | {p["captured_change_candidates"]} |')
    lines += ['', 'Captured changes are review candidates, not verified human actions or reward labels.',
              'Document-change candidates retain recorded phase metadata but are not filtered by independently aligned task boundaries.',
              'Task segments exclude introductions, interviews and ambiguous phase boundaries. They still contain researcher speech and quotations.',
              'Graph snapshots are retained without reconstructing a trajectory from node timestamps.', '', '## Blocking evidence gaps', '']
    lines += ['- ' + x for x in r['blockers']]
    lines += ['', '## Operational rules (choices, not empirical findings)', '']
    lines += [f'- **{k}:** {v}' for k, v in r['operational_choices'].items()]
    lines += ['', 'The machine-readable report includes source hashes, anchor mismatches, malformed lines, and decoding exclusions.',
              'Repeated clock offsets are metadata consistency, not independent proof of alignment.',
              'No model training was run and no claims about improved reasoning or writing quality were tested.', '']
    return '\n'.join(lines)


def check(bundle, verify_sources=False):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['schema_version'] != VERSION:
        raise ValueError('Unsupported bundle version')
    actual = {p.name for p in bundle.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if actual != set(manifest['artifacts']):
        raise ValueError('Unexpected or missing bundle artifact')
    for name, sha in manifest['artifacts'].items():
        if Path(name).name != name or file_hash(bundle / name) != sha:
            raise ValueError(f'Bundle artifact changed: {name}')
    if verify_sources:
        for src in manifest['sources']:
            if file_hash(src['path']) != src['sha256']:
                raise ValueError(f'Source changed: {src["path"]}')
    report = json.loads((bundle / 'report.json').read_text())
    verify_folds(json.loads((bundle / 'folds.json').read_text()), [p['participant'] for p in report['participants']])
    for name in ['task_segments', 'excluded_segments', 'process_annotations', 'unsequenced_document_states', 'candidate_changes']:
        rows = list(load_rows(bundle / (name + '.jsonl')))
        if len(rows) != report['counts'][name] or len({r['record_id'] for r in rows}) != len(rows):
            raise ValueError(f'Count or uniqueness mismatch: {name}')
        if any(r.get('training_eligible') is not False for r in rows):
            raise ValueError('Unreviewed data cannot be training eligible')
    return dict(integrity='passed', training_ready=False, trainable_reward_examples=0,
                blockers=report['blockers'])


def load_training_examples(bundle):
    """Mandatory gate for a training entry point consuming this bundle.

    Version 1 is explicitly a preparation/review format. It cannot be promoted
    by toggling a report flag or by filling template cells with guessed scores.
    A reviewed-supervision import with a versioned rubric must precede training.
    """
    status = check(bundle)
    if not status['training_ready']:
        raise ValueError('Training blocked: ' + '; '.join(status['blockers']))
    raise ValueError('No supported reviewed-supervision format in this bundle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--inventory', type=Path, default=ROOT / 'pilots/results/legacy_runs/thinkaloud_rl_inventory/inventory.json')
    b.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('check')
    c.add_argument('--bundle', type=Path, required=True)
    c.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.inventory, args.output)
        print(canonical(dict(output=str(args.output), counts=result['counts'], training_ready=False)))
    else:
        result = check(args.bundle, args.verify_sources)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['training_ready'] else 2)


if __name__ == '__main__':
    main()
