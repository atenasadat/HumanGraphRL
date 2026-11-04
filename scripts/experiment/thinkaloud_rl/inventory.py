"""Inventory all 14 main-study writers; never hydrate cloud-only files.

Scans available local event logs once, including legacy full-text snapshots.
Does not infer writing quality, expert status or validated transcript alignment.
"""
import argparse
from collections import Counter
import difflib
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .core import SnapshotDecoder, union_duration
from .paths import ANNOTATIONS, CLOUD, LOCAL, ROOT, TRANSCRIPTS, cloud_only, relocated

SESSIONS = {
    'P4':'P4_Task_2026-08-28_17-16-07', 'P5':'P5_Task_2026-08-28_21-12-44',
    'P6':'P6_Task_2026-08-28_23-23-47', 'P7':'P7_Task_2026-08-29_15-29-06',
    'P8':'P8_Task_2026-08-29_21-19-38', 'P9':'P9_Task_2026-08-31_16-20-20',
    'P10':'p10_Task_2026-08-31_20-18-58', 'P11':'P11_Task_2026-08-31_22-38-01',
    'P12':'P12_Task_2026-09-01_17-27-27', 'P13':'P13_Task_2026-09-01_19-36-27',
    'P14':'p14_Task_2026-09-01_21-42-18', 'P15':'P15_Task_2026-09-02_17-23-31',
    'P16':'p16_Task_2026-09-02_23-44-19', 'P17':'P17_Task_2026-09-03_23-48-41',
}
EVENT = re.compile(rb'"event_type"\s*:\s*"([^"\\]+)"')
PHASES = {'Brainstorm','Draft','Revise'}
RESPONSES = Path.home()/'VSProjects/Jaam/CHI2027 Analysis/responses'


def metadata(path):
    s = path.stat()
    return dict(path=str(path), bytes=s.st_size, allocated_bytes=s.st_blocks*512,
                cloud_only=cloud_only(path), mtime_ns=s.st_mtime_ns)


def clip(a,b,c,d):
    return (max(a,c),min(b,d)) if min(b,d)>max(a,c) else None


def normalize_whitespace(text):
    """Comparison only: retain wording; normalize line endings/trailing spaces."""
    return '\n'.join(line.rstrip() for line in text.replace('\r\n','\n').replace('\r','\n').split('\n')).strip()


def state_snapshots(data, pid, source):
    nodes=data.get('nodes',{})
    actions=nodes.get('physical_action',[]) if isinstance(nodes,dict) else []
    rows=[]
    for index,node in enumerate(actions):
        text=node.get('full_code_snapshot')
        if not isinstance(text,str):continue
        timestamp=node.get('timestamp')
        valid_time=isinstance(timestamp,(int,float)) and math.isfinite(timestamp)
        rows.append(dict(participant=pid,source=str(source),node_index=index,node_id=node.get('id'),
                         timestamp=timestamp,timestamp_valid=valid_time,text=text,
                         text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         quality_label=None,alignment='not validated',
                         provenance='stored physical-action full_code_snapshot; not a generated reasoning label',
                         timing_caveat='Node timestamp may describe action creation, not exact text-capture time'))
    return rows


def enrich(rows,out):
    """Inspect compact session states and curated responses for all writers."""
    dest=out/'physical_action_states';dest.mkdir(exist_ok=True)
    endpoint_differences=[]
    for row in rows:
        pid=row['participant'];session=row['session_id']
        candidates=[p for folder in row['folders'] for p in sorted(Path(folder).glob('*session_state.json'))]
        cache=out/'selective_state_cache'/f'{session}.session_state.json'
        local=[p for p in candidates if not cloud_only(p)]
        # Prefer the relocated local study copy, then the explicitly retrieved cache.
        primary=next((p for p in local if str(p).startswith(str(LOCAL))),None)
        if primary is None and cache.exists():primary=cache
        if primary is None and local:primary=local[0]
        if primary is None:
            row['physical_action_history']=dict(status='not_read',candidates=[str(p) for p in candidates])
            continue
        raw=primary.read_bytes();data=json.loads(raw);states=state_snapshots(data,pid,primary)
        state_path=dest/f'{pid}.snapshots.jsonl'
        state_path.write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in states))
        valid=[s['timestamp'] for s in states if s['timestamp_valid']]
        normalized={normalize_whitespace(s['text']) for s in states}
        item=dict(status='read',path=str(primary),sha256=hashlib.sha256(raw).hexdigest(),
                  original_cloud_source=str(CLOUD/session/'session_state.json') if primary==cache else None,
                  bytes=len(raw),snapshots=len(states),distinct_texts=len({s['text_sha256'] for s in states}),
                  empty_snapshots=sum(not s['text'].strip() for s in states),
                  timestamps_valid=len(valid),duplicate_timestamps=len(valid)-len(set(valid)),
                  timestamp_reversals=sum(a>b for a,b in zip(valid,valid[1:])),
                  node_id_duplicates=len(states)-len({s['node_id'] for s in states}),
                  epoch_range=[min(valid),max(valid)] if valid else None,
                  output=str(state_path),complete_edit_history=False,
                  limitation='Retained graph nodes are sampled/merged action records. Do not assume all edits '
                  'survive, node order is temporal, or node timestamp equals text capture.')
        file=RESPONSES/f'{pid}_response.txt'
        if file.exists():
            response=file.read_text();nr=normalize_whitespace(response)
            item['curated_response']=dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                  characters=len(response),words=len(response.split()),exact_match_any_snapshot=any(s['text']==response for s in states),
                  whitespace_normalized_match_any_snapshot=nr in normalized,
                  word_sequence_match_any_snapshot=any(s['text'].split()==response.split() for s in states),
                  whitespace_normalized_match_last_listed=bool(states and nr==normalize_whitespace(states[-1]['text'])),
                  provenance='See responses/MANIFEST_responses.md; P12 file is present but not documented in that manifest.')
            if states and not item['curated_response']['whitespace_normalized_match_any_snapshot']:
                before=normalize_whitespace(states[-1]['text'])
                edits=[dict(operation=op,snapshot_span=[i,j],curated_span=[k,l],snapshot_text=before[i:j],curated_text=nr[k:l])
                       for op,i,j,k,l in difflib.SequenceMatcher(None,before,nr,autojunk=False).get_opcodes() if op!='equal']
                endpoint_differences.append(dict(participant=pid,last_listed_node=states[-1]['node_id'],
                    snapshot_characters=len(before),curated_characters=len(nr),edits=edits))
        # Compare against independent stored editor bodies when available.
        matches=[]
        for f in row['small_files']:
            if f['status']!='read':continue
            path=Path(f['path']);text=None
            if path.name in {'response.txt','latest_editor_content.txt'}:text=path.read_text()
            elif path.name=='reload_state.json':text=json.loads(path.read_text()).get('editor_text')
            if isinstance(text,str):matches.append(dict(path=str(path),exact_match_any_snapshot=any(s['text']==text for s in states)))
        item['saved_editor_checks']=matches
        row['physical_action_history']=item
        if len(states)>1:row['readiness']='stored_document_states_available; timing_and_quality_unvalidated'
        print(pid,'physical-action snapshots',len(states),'distinct',item['distinct_texts'],flush=True)
    (out/'endpoint_differences.json').write_text(json.dumps(endpoint_differences,ensure_ascii=False,indent=2)+'\n')


def annotation_audit(pid, original):
    path = ANNOTATIONS/f'{pid}.json'
    raw = path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if digest != original['source_sha256']: raise ValueError(f'Annotation source changed: {pid}')
    x = json.loads(raw); transcript=x['transcript']; anns=x['annotations']
    phases=[a for a in anns if a['tierId']=='feature' and a['codeName'] in PHASES]
    moves=[a for a in anns if a['tierId']=='reasoning']
    phase_rows={}
    for phase in sorted(PHASES):
        windows=[(a['startTime'],a['endTime']) for a in phases if a['codeName']==phase]
        covered=[]; ids=set()
        for a in moves:
            for c,d in windows:
                overlap=clip(a['startTime'],a['endTime'],c,d)
                if overlap: covered.append(overlap);ids.add(a['id'])
        duration=union_duration(windows)
        phase_rows[phase]=dict(windows_ms=windows,process_spans=len(ids),
                               covered_fraction=union_duration(covered)/duration if duration else None)
    # Boundary-crossing segments are not silently treated as fully task-only.
    windows=[(a['startTime'],a['endTime']) for a in phases]
    contained=[];boundary=[]
    for r in transcript:
        overlaps=[v for a,b in windows if (v:=clip(r['startTime'],r['endTime'],a,b))]
        if not overlaps:continue
        span=r['endTime']-r['startTime']
        (contained if abs(union_duration(overlaps)-span)<1e-6 else boundary).append(r['index'])
    return dict(path=str(path),sha256=digest,source_unchanged=True,
                total_transcript_segments=len(transcript),task_contained_segments=len(contained),
                boundary_crossing_segments=len(boundary),task_segment_indexes=contained,
                boundary_segment_indexes=boundary,process_annotations=len(moves),
                process_labels=sorted({a['codeName'] for a in moves}),phases=phase_rows,
                speaker_tracks=len({r.get('speaker') for r in transcript}),
                ai_usage_annotations=sum(a['tierId']=='ai-usage' for a in anns),
                nudge_coding_records=len(x.get('nudgeCodings',[])),
                quality_supervision='Goal Analysis codes describe activities; not step-quality labels',
                transcript_alignment='video clock available; event-clock alignment not validated')


def log_audit(path, pid, out):
    info=metadata(path)
    if info['cloud_only']:return dict(**info,status='cloud_only_not_read',reason='Avoid hydration of large event logs')
    counts=Counter(); modes=Counter();decoder=SnapshotDecoder();state_hashes=set();last_hash=None
    bad=0;bytes_read=0;timing=[];sample_clocks=[];start=time.monotonic();last_notice=start
    state_file=out/f'{pid}.{path.name}.states.jsonl'
    opener=gzip.open if path.suffix=='.gz' else open
    # Every line is scanned; only relevant events incur full JSON decoding.
    with opener(path,'rb') as stream,state_file.open('w') as dst:
        for number,line in enumerate(stream,1):
            bytes_read+=len(line)
            match=EVENT.search(line[:4096])
            if not match:
                counts['unclassified_lines']+=1;continue
            kind=match.group(1).decode('utf-8');counts['event:'+kind]+=1
            selected=(kind in {'writing.document_snapshot','writing.final_response_snapshot','session.sync_marker',
                               'session.phase_change','study_phase.changed','session_task_started','session.task_started',
                               'raw_text','ai.prompt','ai.response','raw_chat','speech.raw_transcript'})
            if selected:
                try:r=json.loads(line)
                except (ValueError,UnicodeError):bad+=1;continue
                p=r.get('payload');p=p if isinstance(p,dict) else {}
                if p:counts['payload:'+kind]+=1
                if kind in {'session.sync_marker','session.phase_change','study_phase.changed','session_task_started','session.task_started'}:
                    timing.append({k:r[k] for k in ['event_type','clock','t_s','study_phase','phase','server_received_at','payload','summary'] if k in r})
                if kind=='speech.raw_transcript' and isinstance(p.get('text'),str):counts['speech_events_with_text']+=1
                if kind in {'ai.prompt','ai.response','raw_chat'}:
                    fields=[k for k,v in p.items() if isinstance(v,str) and ('text' in k or 'content' in k or 'prompt' in k or 'response' in k)]
                    if fields:counts['chat_events_with_text_fields']+=1
                if kind in {'writing.document_snapshot','writing.final_response_snapshot','raw_text'}:
                    text=None; method=None
                    if kind=='raw_text' and isinstance(p.get('current'),str):
                        text=p['current'];method='raw_text_current'
                    elif kind=='raw_text':
                        # Raw-text edits have an independent text_snapshot_seq schema.
                        # Never feed them into the document_snapshot decoder/state chain.
                        counts['raw_text_without_full_current_not_decoded']+=1
                    elif p.get('text_mode'):
                        modes[p['text_mode']]+=1
                        try:text=decoder.accept(p);method='decoded_'+p['text_mode']
                        except ValueError:counts['snapshot_decode_rejected']+=1
                    elif isinstance(p.get('text'),str):
                        text=p['text'];method='legacy_full_text'
                        if p.get('character_count') is not None and len(text)!=p['character_count']:
                            counts['legacy_character_count_mismatch']+=1
                    else:counts['writing_events_without_body']+=1
                    if text is not None:
                        counts['recoverable_text_events']+=1
                        digest=hashlib.sha256(text.encode()).hexdigest();last_hash=digest
                        if digest not in state_hashes:
                            state_hashes.add(digest)
                            dst.write(json.dumps(dict(participant=pid,source=str(path),line=number,
                                event_type=kind,clock=r.get('clock'),t_s=r.get('t_s'),
                                phase=r.get('study_phase',r.get('phase')),method=method,text=text,
                                text_sha256=digest,quality_label=None,alignment='not validated'),ensure_ascii=False)+'\n')
                        if len(sample_clocks)<3:sample_clocks.append(dict(clock=r.get('clock'),t_s=r.get('t_s')))
            now=time.monotonic()
            if now-last_notice>20:
                print(f'{pid}: scanned {bytes_read/1e9:.2f} GB; {len(state_hashes)} distinct document texts',flush=True)
                last_notice=now
    after=path.stat()
    if after.st_size!=info['bytes'] or after.st_mtime_ns!=info['mtime_ns']:raise ValueError('Source changed during scan')
    return dict(**info,status='fully_scanned',counts=dict(counts),modes=dict(modes),
                malformed_selected_lines=bad,uncompressed_bytes_scanned=bytes_read,distinct_texts=len(state_hashes),
                last_text_sha256=last_hash,states_file=str(state_file),timing_events=timing,
                sample_clocks=sample_clocks,elapsed_seconds=time.monotonic()-start,
                state_export='Distinct text values only; first occurrence retained. Not a complete transition sequence.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'pilots/results/legacy_runs/thinkaloud_rl_inventory')
    ap.add_argument('--refresh-state-sources',action='store_true',help='Reuse completed event scans; inspect compact session states and saved responses')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.refresh_state_sources:
        result=json.loads((out/'inventory.json').read_text())
        enrich(result['participants'],out)
        result['scope']='All 14 writing-study annotation exports and compact session states inspected. '
        result['scope']+='Four cloud session states selectively retrieved (9.35 MB); large cloud logs not read. '
        result['scope']+='Local event scans reused; no new quality labels or validated transcript-clock alignment.'
        (out/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
        report(result['participants'],out);return
    manifest={r['participant']:r for r in json.loads((TRANSCRIPTS/'manifest.json').read_text())}
    results=[]
    for pid,session in SESSIONS.items():
        print('Auditing',pid,flush=True)
        folders=list(dict.fromkeys(p for p in [LOCAL/session,LOCAL/pid,CLOUD/session] if p.is_dir()))
        row=dict(participant=pid,session_id=session,annotation=annotation_audit(pid,manifest[pid]),
                 folders=[str(p) for p in folders],artifacts=[],small_files=[],logs=[])
        for folder in folders:
            for p in sorted(folder.iterdir()):
                if p.name.startswith('.'):continue
                if p.is_dir():
                    files=[f for f in p.rglob('*') if f.is_file() and not f.name.startswith('.')]
                    if any(w in p.name for w in ['reasoning_graph','physical_action_graph','audio','screen']):
                        row['artifacts'].append(dict(path=str(p),kind=p.name,files=len(files),
                            cloud_only_files=sum(cloud_only(f) for f in files),bytes=sum(f.stat().st_size for f in files),
                            content_validation='metadata only; media not played; graphs not treated as truth'))
                    continue
                if p.name.startswith('real_time_events') and p.suffix in {'.jsonl','.gz'}:
                    # Use a local copy when it has the same session filename; list cloud duplicate without reading.
                    if folder.parent==CLOUD and any(Path(v['path']).name==p.name and v['status']=='fully_scanned' for v in row['logs']):
                        row['logs'].append(dict(**metadata(p),status='cloud_duplicate_not_read'));continue
                    tag='cloud' if folder.parent==CLOUD else 'local'
                    run_dir=out/tag;run_dir.mkdir(exist_ok=True)
                    row['logs'].append(log_audit(p,pid,run_dir));continue
                wanted=p.name in {'response.txt','latest_editor_content.txt','reload_state.json'} or p.name.endswith('manifest.json')
                if not wanted:continue
                entry=metadata(p)
                if entry['cloud_only']:
                    entry['status']='cloud_only_not_read';row['small_files'].append(entry);continue
                raw=p.read_bytes();entry['sha256']=hashlib.sha256(raw).hexdigest();entry['status']='read'
                if p.suffix=='.txt':
                    text=raw.decode('utf-8');entry.update(text_characters=len(text),text_words=len(text.split()),role='saved document; finality not assumed')
                else:
                    x=json.loads(raw)
                    if p.name=='reload_state.json':
                        text=x.get('editor_text');entry.update(saved_at=x.get('saved_at'),study_phase=x.get('study_phase'),
                            editor_characters=len(text) if isinstance(text,str) else None,
                            editor_sha256=hashlib.sha256(text.encode()).hexdigest() if isinstance(text,str) else None,
                            messages=len(x.get('messages',[])),role='single recovery state; not a document history')
                    else:
                        entry.update(session_id=x.get('session_id'),task_description_present=bool(x.get('task_description')),
                                     declared_files=x.get('files',{}),role='manifest declarations do not prove files exist')
                row['small_files'].append(entry)
        histories=[v for v in row['logs'] if v.get('distinct_texts',0)>1]
        local_saved=[v for v in row['small_files'] if v['status']=='read' and (v.get('text_characters',0)>0 or v.get('editor_characters',0)>0)]
        for log in histories:
            log['last_text_matches_saved_files']=[v['path'] for v in local_saved if log['last_text_sha256'] in {v.get('editor_sha256'),v.get('sha256')}]
        row['readiness']=('document_history_available' if histories else 'saved_document_only' if local_saved
                          else 'history_unverified_cloud' if any(v['status']=='cloud_only_not_read' for v in row['logs'])
                          else 'transcript_and_graphs_only_in_inspected_folders')
        row['quality_labels']='No validated step-quality labels identified in inspected process annotations or document artifacts'
        results.append(row)
        (out/'inventory.json').write_text(json.dumps(dict(participants=results,complete=len(results)==14,
            scope='Main study P4–P17. Source files read only. Cloud-only contents not hydrated. '
                  'Transcripts verified against previous copies; no claim of a new full manual reading. '
                  'No new quality labels, clock alignment or training performed.'),indent=2)+'\n')
        print(pid,row['readiness'],flush=True)
    enrich(results,out)
    (out/'inventory.json').write_text(json.dumps(dict(participants=results,complete=True,
        scope='All 14 writers; local event logs and compact session states, including any selectively retrieved caches. '
              'Large cloud event logs not hydrated. No new quality labels or clock alignment.'),indent=2)+'\n')
    report(results,out)


def report(rows,out):
    lines=['# All-writer training-data audit','', 'Audit date: 2026-10-26.', '',
           'Main-study writers P4–P17. No source data was edited. Four compact cloud session states '
           'were selectively retrieved (9.35 MB total); large cloud event logs were not downloaded. '
           'Legacy paths are resolved to the Desktop consolidation. This checks data availability and structure; '
           'it does not validate quality labels or establish expert demonstrations.','',
           '**Finding: all 14 writers have stored document snapshots inside physical-action nodes. '
           'Restricting the sample to P16/P17 was unnecessary. These are retained action snapshots, '
           'not proven complete edit histories. All 14 also have curated proposal files.**','',
           '| Writer | Transcript segments (all / wholly in task) | Process spans | Action snapshots / distinct texts | Curated proposal agrees after whitespace normalization |',
           '|---|---:|---:|---:|---|']
    for r in rows:
        a=r['annotation'];h=r.get('physical_action_history',{});c=h.get('curated_response',{})
        agreement='yes' if c.get('whitespace_normalized_match_any_snapshot') else 'same words; layout differs' if c.get('word_sequence_match_any_snapshot') else 'wording differs—retain both versions'
        lines.append(f"| {r['participant']} | {a['total_transcript_segments']} / {a['task_contained_segments']} | {a['process_annotations']} | {h.get('snapshots','?')} / {h.get('distinct_texts','?')} | {agreement} |")
    total=sum(r.get('physical_action_history',{}).get('snapshots',0) for r in rows)
    distinct=sum(r.get('physical_action_history',{}).get('distinct_texts',0) for r in rows)
    lines+=['',f'{total} stored action snapshots; {distinct} distinct texts summed within writers. '
            'Repeated/near-identical snapshots are not independent demonstrations. Original text is preserved; '
            'normalization is used only for comparison. A matching endpoint does not validate intermediate timestamps.',
            '', 'Endpoint discrepancies: P10’s curated proposal contains an additional 77 characters relative to '
            'the last listed action snapshot after trailing-whitespace normalization; the earlier recovery notes '
            'also document a truncated graph endpoint. P12 has identical words but nine additional newline characters '
            'in the curated copy. Both versions are retained. See [endpoint differences](endpoint_differences.json).',
            '', '## Additional event-log coverage', '',
            '| Writer | Event-log status | Distinct text values extracted |', '|---|---|---:|']
    for r in rows:
        scanned=[v for v in r['logs'] if v['status']=='fully_scanned']
        desc='local logs fully scanned' if scanned else 'large cloud log not read' if r['logs'] else 'no event log in inspected folders'
        lines.append(f"| {r['participant']} | {desc} | {sum(v.get('distinct_texts',0) for v in scanned)} |")
    lines+=['', 'The event scan includes raw-text current states and final snapshots as well as document snapshots; '
            'its distinct-text counts can exceed the earlier document-snapshot-only audit. '
            'Metadata-only events do not supply text bodies. Full event-log payloads were available for P4; '
            'P16/P17 document snapshot chains decoded without failures after separating the raw-text event schema.',
            '', '## Phase coverage and timing', '',
            '| Writer | Brainstorm coverage | Draft coverage | Revision coverage | Boundary-crossing transcript segments | Speaker tracks |',
            '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a=r['annotation'];p=a['phases']
        vals=[f"{p[k]['covered_fraction']:.1%}" if p[k]['covered_fraction'] is not None else 'missing' for k in ['Brainstorm','Draft','Revise']]
        lines.append('| '+' | '.join([r['participant']]+vals+[str(a['boundary_crossing_segments']),str(a['speaker_tracks'])])+' |')
    reversed_people=sum(r.get('physical_action_history',{}).get('timestamp_reversals',0)>0 for r in rows)
    duplicate_people=sum(r.get('physical_action_history',{}).get('duplicate_timestamps',0)>0 for r in rows)
    lines+=['', f'Action snapshots carry numeric timestamps, but {reversed_people}/14 writers have reversals in '
            f'the saved node order and {duplicate_people}/14 have duplicate timestamps. Do not assume list order '
            'is chronological. More importantly, node creation time may precede a later merged snapshot; simply '
            'sorting the nodes does not prove a leak-free time alignment. Capture times need corroborating event '
            'or graph-version records before pairing text with earlier speech.', '',
            'Task-window transcript counts include researcher speech and AI quotations; they are not counts '
            'of participant-only reasoning steps. Coverage means time covered by existing Goal Analysis annotations, not reasoning frequency. '
            'Overlapping spans are counted once within each phase. Boundary-crossing speech is flagged rather than silently '
            'assigned to a task. P8 has known diarization problems. Phase order was imposed by the study.', '',
            '## What can be used now', '',
            '- All 14 writers: transcript prefixes and existing descriptive process spans can support source-linked '
            'candidate examples. Their quality and actor attribution still need assessment.',
            '- All writers with stored action snapshots: candidate document-before/action/document-after examples, '
            'after event-to-transcript clock alignment and human/AI attribution. Text availability alone is not a PRM label.',
            '- A single saved or recovery document supports an endpoint, not a temporal quality-delta sequence.',
            '- Large cloud event logs remain uninspected, but their compact session states provide usable text candidates.',
            '- Stored full_code_snapshot text is distinct from generated graph labels. Generated move types, confidence, '
            'Toulmin explanations and graph changes are not independently validated reward labels.',
            '', '## Next step for the RL ideas', '',
            'Align stored action snapshots for all writers with the existing speech and process spans. '
            'Create candidate episodes without assigning good/bad labels automatically. '
            'Expand the sample using selectively retrieved cloud logs only when storage and retrieval are available. '
            'Keep writer/task splits separate; do not train a PRM to imitate frequent actions as if they were high-quality actions.', '',
            'Full paths, availability flags, media/graph counts, snapshot decode counts, timing events and saved-text checks '
            'are in [inventory.json](inventory.json). All document exports are local audit derivatives, not moved source folders.','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
