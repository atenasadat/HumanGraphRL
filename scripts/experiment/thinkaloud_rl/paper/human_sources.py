"""Task-boundary-reviewed human text, with raw-source hash verification."""
import json
from pathlib import Path
from .common import ROOT,file_hash


def session_texts():
    result={'programming':{},'writing':{}}
    reviewed=ROOT/'processed_data/programming_transcript_cells_and_task_boundaries'
    manifest=json.loads((reviewed/'task_manifest.json').read_text())
    for entry in manifest:
        pid=entry['participant']
        raw=ROOT/'data/think-aloud/programming_transcripts'/Path(entry['source']).name
        if not raw.exists() or file_hash(raw)!=entry['source_sha256']:
            raise ValueError('Missing or changed raw programming source: '+str(raw))
        rows=json.loads((reviewed/(pid+'.task_cells.json')).read_text())
        result['programming'][pid]=[r['text'] for r in sorted(rows,key=lambda r:r['excel_row']) if r['text'].strip()]
    for p in sorted((ROOT/'processed_data/writing_task_segments_extracted_from_transcripts').glob('*_proposal_verbalizations.json')):
        obj=json.loads(p.read_text());rows=obj['segments']
        if not rows:continue
        pid=p.name.split('_')[0]
        raw=ROOT/'data/think-aloud/writing_transcripts'/(pid+'.transcript.json')
        if not raw.exists() or file_hash(raw)!=obj['metadata']['sha256']:
            raise ValueError('Missing or changed raw writing source: '+str(raw))
        result['writing'][pid]=[r['text'] for r in rows if r['text'].strip()]
    return result
