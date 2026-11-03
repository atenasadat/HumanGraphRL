"""Blinded independent human review; missing or invalid judgments remain pending."""
from collections import Counter
import json
from pathlib import Path
import random

from .common import atomic_json,digest,read_jsonl,iter_jsonl,write_jsonl


def packet(root,kind,items):
    folder=Path(root)/'human_review'/kind;folder.mkdir(parents=True,exist_ok=True)
    # Stable IDs preserve existing human work when new model samples become available.
    key=[];public=[]
    for item in items:
        identifier=digest(item)
        key.append({'id':identifier,**item})
        public.append({'id':identifier,**{k:v for k,v in item.items() if k not in ('run','source','participant','automated')},
                       'score':None,'risk':None,'notes':''})
    random.Random(2026).shuffle(public)
    write_jsonl(folder/'private_key.jsonl',key)
    for reviewer in ('a','b'):
        path=folder/f'reviewer_{reviewer}.jsonl'
        existing={r['id']:r for r in read_jsonl(path)} if path.exists() else {}
        write_jsonl(path,[existing.get(r['id'],r) for r in public])
    atomic_json(folder/'instructions.json',{'kind':kind,'score_scale':'0–4 integer',
        'risk':'Boolean: serious unsupported claim/nonsense for output; materially false semantic grouping/recurrence for graph.',
        'instruction':'Two reviewers work independently. Keep model, method, seed and automatic scores hidden. '
            'Use unknown in notes when evidence is insufficient; leave score/risk null rather than inventing certainty. '
            'Graph: assess whether grouped steps have a recurring semantic function and whether transitions are faithful. '
            'Output: assess relevance, coherence and factual grounding against supplied evidence. '
            'A fluent or human-sounding trace is not proof of faithful internal reasoning.'})


def output_packet(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());items=[]
    limit=plan['config'].get('review_samples_per_run',8)
    tasks={t['id']:t for t in iter_jsonl(Path(plan['data_root'])/'tasks.jsonl') if t['domain']=='writing'}
    for run in sorted((root/'runs').glob('*')):
        samples=sorted((run/'test/samples').glob('*.json'),key=lambda p:digest(p.name))
        writing=[json.loads(p.read_text()) for p in samples if json.loads(p.read_text())['domain']=='writing'][:limit]
        for s in writing:
            items.append({'run':run.name,'source':'model','prompt':s['prompt'],'answer':s['final_answer'],
                'evidence':tasks[s['task_id']].get('evidence',''),'criteria':tasks[s['task_id']].get('criteria',[]),
                'task_id':s['task_id'],'sample':s['sample'],'automated':s.get('output_assessment')})
    preflight=root/'quality_preflight.json'
    if preflight.exists():
        for r in json.loads(preflight.read_text())['fixtures']:
            if r['phase']=='evaluation':items.append({'source':'fixture','run':'preflight',
                'prompt':r['prompt'],'answer':r['answer'],'evidence':r['evidence'],'automated':r['assessment']})
    packet(root,'output',items)


def graph_packet(root,include_model=False):
    root=Path(root);data=json.loads((root/'structure/references.json').read_text());items=[]
    config=json.loads((root/'plan.json').read_text())['config']
    grouped={}
    for r in data['human_windows']:
        if config.get('human_eval_partition')=='diagnostic' and r['split']=='held_out':continue
        key=(r['domain'],r['split']);grouped.setdefault(key,[]).append(r)
    for key,rows in grouped.items():
        for r in sorted(rows,key=lambda r:digest(r['text_steps']))[:8]:
            items.append({'source':'human_'+r['domain'],'participant':r['participant'],
                          'text_steps':r['text_steps'],'graph':r['graph']})
    if include_model:
        from .structure import steps
        for run in sorted((root/'runs').glob('*')):
            valid=[]
            for p in sorted((run/'test/samples').glob('*.json'),key=lambda p:digest(p.name)):
                r=json.loads(p.read_text());g=r.get('structure',{})
                if g.get('valid'):valid.append({'source':'model','run':run.name,'prompt':r['prompt'],
                    'final_answer':r['final_answer'],'text_steps':steps(r['text']),'graph':g['graph']})
                if len(valid)>=8:break
            items.extend(valid)
    packet(root,'graph',items)


def review_status(root):
    result={}
    for kind in ('graph','output'):
        folder=Path(root)/'human_review'/kind;key=folder/'private_key.jsonl'
        expected={r['id'] for r in read_jsonl(key)} if key.exists() else set()
        reviewers=[]
        for name in ('a','b'):
            p=folder/f'reviewer_{name}.jsonl';rows=read_jsonl(p) if p.exists() else []
            ids=[r.get('id') for r in rows]
            if len(ids)!=len(set(ids)):raise ValueError('Duplicate human review ID')
            if set(ids)-expected:raise ValueError('Unknown human review ID')
            reviewers.append({r['id']:r for r in rows if type(r.get('score')) is int and
                              0<=r['score']<=4 and type(r.get('risk')) is bool})
        paired=set(reviewers[0])&set(reviewers[1]);n=len(paired)
        result[kind]={'expected':len(expected),'paired_complete':n,
            'complete':bool(expected) and paired==expected,
            'mean_absolute_score_disagreement':sum(abs(reviewers[0][k]['score']-reviewers[1][k]['score']) for k in paired)/n if n else None,
            'risk_agreement':sum(reviewers[0][k]['risk']==reviewers[1][k]['risk'] for k in paired)/n if n else None,
            'jointly_flagged_risks':sum(reviewers[0][k]['risk'] and reviewers[1][k]['risk'] for k in paired),
            'either_reviewer_risk_fraction':sum(reviewers[0][k]['risk'] or reviewers[1][k]['risk'] for k in paired)/n if n else None,
            'mean_human_score':sum(reviewers[i][k]['score'] for i in (0,1) for k in paired)/(2*n) if n else None}
        if kind=='output':
            private={r['id']:r for r in read_jsonl(key)} if key.exists() else {}
            consensus=[k for k in paired if reviewers[0][k]['risk']==reviewers[1][k]['risk'] and private[k].get('automated')]
            tp=fp=fn=tn=0;errors=[]
            for k in consensus:
                gold=reviewers[0][k]['risk'];pred=private[k]['automated']['risk_detected']
                tp+=bool(gold and pred);fp+=bool(not gold and pred);fn+=bool(gold and not pred);tn+=bool(not gold and not pred)
                value=private[k]['automated'].get('quality_score')
                if value is not None:errors.append(abs(value-(reviewers[0][k]['score']+reviewers[1][k]['score'])/8))
            result[kind]['judge_calibration']={'consensus_items':len(consensus),'true_positive':tp,'false_positive':fp,
                'false_negative':fn,'true_negative':tn,'risk_precision':tp/(tp+fp) if tp+fp else None,
                'risk_recall':tp/(tp+fn) if tp+fn else None,
                'score_mae':sum(errors)/len(errors) if errors else None,
                'limitation':'Consensus subset and included fixtures are not a representative hallucination-rate estimate.'}
    return result
