"""Read-only audit of data prerequisites for rl_ideas_thinkaloud.md."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from scripts.experiment.thinkaloud_rl.core import SnapshotDecoder,union_duration
from scripts.experiment.thinkaloud_rl.paths import LOCAL,TRANSCRIPTS,relocated

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'processed_data').is_dir())
OUT=ROOT/'pilots/results/legacy_runs/thinkaloud_rl_audit'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((TRANSCRIPTS/'manifest.json').read_text())
    people=[];spans=[]
    for m in manifest:
        source=relocated(m['source'])
        raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=m['source_sha256']:raise ValueError('Annotation source changed')
        x=json.loads(raw)
        phases=[a for a in x['annotations'] if a['tierId']=='feature' and a['codeName'] in ['Brainstorm','Draft','Revise']]
        anns=[a for a in x['annotations'] if a['tierId']=='reasoning']
        phase_stats={}
        for name in ['Brainstorm','Draft','Revise']:
            windows=[(a['startTime'],a['endTime']) for a in phases if a['codeName']==name]
            intervals=[];included=[]
            for a in anns:
                clips=[(max(a['startTime'],s),min(a['endTime'],e)) for s,e in windows if min(a['endTime'],e)>max(a['startTime'],s)]
                if clips:included.append(a['id']);intervals.extend(clips)
            total=union_duration(windows)
            phase_stats[name]=dict(annotations=len(included),covered_fraction=union_duration(intervals)/total if total else None)
        people.append(dict(participant=m['participant'],source=str(source),original_source=m['source'],source_sha256=m['source_sha256'],
                           process_annotations=len(anns),distinct_labels=len({a['codeName'] for a in anns}),phases=phase_stats))
        for a in anns:
            overlaps=[dict(phase=p['codeName'],start_ms=max(p['startTime'],a['startTime']),end_ms=min(p['endTime'],a['endTime']))
                      for p in phases if min(p['endTime'],a['endTime'])>max(p['startTime'],a['startTime'])]
            if not overlaps:continue
            # Exact labels are retained as descriptive metadata, not numeric rewards.
            spans.append(dict(participant=m['participant'],annotation_id=a['id'],label=a['codeName'],
                              start_ms=a['startTime'],end_ms=a['endTime'],task_overlaps=overlaps,
                              source=str(source),source_segment_indexes=a.get('transcriptSegmentIndexes',[]),
                              quality_label=None,quality_label_status='not supplied in this annotation',
                              role='descriptive process-span candidate; not expert step or reward'))
    snapshots=[]
    folders={'P14':'p14_Task_2026-09-01_21-42-18','P16':'p16_Task_2026-09-02_23-44-19','P17':'P17_Task_2026-09-03_23-48-41'}
    for pid,folder in folders.items():
        base=LOCAL/folder
        path=base/'real_time_events.jsonl.gz'
        decoder=SnapshotDecoder();counts=Counter();states=[];unique=set();errors=[]
        with gzip.open(path,'rt') as f:
            for line_no,line in enumerate(f,1):
                r=json.loads(line)
                if r.get('event_type')!='writing.document_snapshot':continue
                counts['events']+=1
                p=r.get('payload')
                if not p:counts['metadata_only']+=1;continue
                counts['mode_'+str(p.get('text_mode'))]+=1
                try:text=decoder.accept(p)
                except ValueError:
                    counts['rejected']+=1;errors.append(dict(line=line_no,seq=p.get('snapshot_seq')));continue
                counts['decoded']+=1
                digest=hashlib.sha256(text.encode()).hexdigest();unique.add(digest)
                states.append(dict(participant=pid,source=str(path),line=line_no,t_s=r.get('t_s'),
                                   event_phase=r.get('phase'),snapshot_seq=p['snapshot_seq'],text=text,
                                   text_sha256=digest,logged_identifier=p['text_sha256_16'],
                                   has_quality_label=False,transcript_time_alignment='not yet validated'))
        final=base/'latest_editor_content.txt'
        matched=bool(states and final.exists() and states[-1]['text']==final.read_text())
        snapshots.append(dict(participant=pid,source=str(path),counts=dict(counts),unique_texts=len(unique),
                              last_matches_saved_final=matched,rejected=errors,
                              source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with (OUT/f'{pid}.document_states.jsonl').open('w') as f:
            for s in states:f.write(json.dumps(s,ensure_ascii=False)+'\n')
    (OUT/'process_spans.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in spans))
    result=dict(participants=people,total_process_annotations=sum(p['process_annotations'] for p in people),
                task_overlapping_candidates=len(spans),snapshots=snapshots,
                scope='All 14 annotation exports; snapshot reconstruction only for the three listed compressed local logs. '
                      'Other event logs not fully audited; no absence claim about uninspected files. '
                      'No invented quality labels, expert labels, or utterance-to-document synchronization.')
    (OUT/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Process annotations:',result['total_process_annotations'],'across',len(people),'participants')
    for s in snapshots:print(s['participant'],s['counts'],'distinct',s['unique_texts'],'final_match',s['last_matches_saved_final'])


if __name__=='__main__':main()
