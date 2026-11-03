"""Resumable evaluation, full generations and correct task-level Pass@k."""
import json
from pathlib import Path
import time
from collections import defaultdict
import torch
from transformers import set_seed

from .common import atomic_json, digest, final_answer, pass_at_k, stable_seed
from .model import sample
from .scoring import score
from ..multidomain_tasks import cognitive_metrics


def summarize(records,expected_samples):
    domains=defaultdict(list)
    for r in records:domains[r['domain']].append(r)
    result={}
    for domain,rows in domains.items():
        tasks=defaultdict(list)
        for r in rows:tasks[r['task_id']].append(r)
        if any(len(x)!=expected_samples for x in tasks.values()):raise ValueError('Incomplete task samples')
        verified={t:rs for t,rs in tasks.items() if all(type(r['success']) is bool for r in rs)}
        metrics={f'pass@{k}':sum(pass_at_k(len(rs),sum(r['success'] for r in rs),k) for rs in verified.values())/len(verified)
                 for k in (1,4,8) if k<=expected_samples and verified}
        metrics.update(tasks=len(tasks),samples=len(rows),verified_tasks=len(verified),
            unscored_tasks=len(tasks)-len(verified),truncated_fraction=sum(r['truncated'] for r in rows)/len(rows),
            mean_generated_tokens=sum(r['tokens'] for r in rows)/len(rows),
            cognitive_metrics=cognitive_metrics([r['text'] for r in rows]))
        proxy=[r['reward'] for r in rows if r['status']=='heuristic_writing_proxy']
        metrics['mean_writing_proxy']=sum(proxy)/len(proxy) if proxy else None
        for key in ('reasoning_truncated','answer_truncated'):
            values=[r[key] for r in rows if r.get(key) is not None]
            metrics[key+'_fraction']=sum(values)/len(values) if values else None
        for key,extract in {
            'mean_output_quality':lambda r:r.get('output_assessment',{}).get('quality_score'),
            'output_risk_fraction':lambda r:r.get('output_assessment',{}).get('risk_detected'),
            'mean_human_train_similarity':lambda r:r.get('structure',{}).get('human_train_similarity'),
            'mean_human_held_out_similarity':lambda r:r.get('structure',{}).get('human_held_out_similarity'),
            'valid_structure_fraction':lambda r:r.get('structure',{}).get('valid'),
        }.items():
            values=[extract(r) for r in rows if extract(r) is not None]
            metrics[key]=sum(values)/len(values) if values else None
        metrics['unverified_factuality_samples']=sum(r.get('output_assessment',{}).get('factuality_status')!='executable_or_symbolic_verifier'
            for r in rows) if any('output_assessment' in r for r in rows) else None
        cue=[abs(r['structure']['cue_removed_similarity']-r['structure']['human_train_similarity'])
             for r in rows if r.get('structure',{}).get('cue_removed_similarity') is not None]
        sensitivity=[max(r['structure']['threshold_sensitivity'].values())-min(r['structure']['threshold_sensitivity'].values())
                     for r in rows if r.get('structure',{}).get('threshold_sensitivity')]
        metrics['mean_cue_removal_score_shift']=sum(cue)/len(cue) if cue else None
        metrics['mean_threshold_score_range']=sum(sensitivity)/len(sensitivity) if sensitivity else None
        metrics['structure_by_length_bin']={}
        for low,high in ((0,256),(256,512),(512,1024),(1024,1000000)):
            subset=[r for r in rows if low<=r['tokens']<high and 'structure' in r]
            if subset:metrics['structure_by_length_bin'][f'{low}:{high}']={
                'samples':len(subset),'held_out_similarity':sum(r['structure'].get('human_held_out_similarity',0.) for r in subset)/len(subset)}
        result[domain]=metrics
    return result


def evaluate(model,tok,tasks,config,out,lcb_root=None,samples=None,train_state=None,runtime=None):
    if lcb_root is not None:
        # The robust reward runtime reads its evaluator path from config.
        # Honor this public argument there as well as in legacy score().
        config={**config,'lcb_root':str(Path(lcb_root).resolve())}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    n=samples or config.get('eval_samples',4)
    signature=digest({'config':config,'task_ids':[t['id'] for t in tasks],'samples':n,'train_state':train_state})
    signature_file=out/'signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text())!=signature:
        raise ValueError('Evaluation settings changed: '+str(out))
    atomic_json(signature_file,signature)
    if config.get('robust_pilot') and runtime is None:
        from .runtime import RewardRuntime
        runtime=RewardRuntime(config,'evaluation')
    generation_config={**config,'telemetry_role':'validation_policy' if tasks[0]['split']=='val' else 'evaluation_policy'}
    started=time.monotonic()
    records=[]
    for task in tasks:
        for j in range(n):
            from . import train as trainer
            if trainer.STOP:raise InterruptedError
            path=out/'samples'/f'{digest([task["id"],j])}.json'
            if path.exists():row=json.loads(path.read_text())
            else:
                set_seed(stable_seed('evaluation',config['seed'],task['id'],j))
                start=time.monotonic()
                r=sample(model,tok,task,generation_config,greedy=n==1)
                scored=runtime.assess(r['text'],task) if runtime else score(r['text'],task,lcb_root=lcb_root,timeout=config.get('code_timeout',6))
                row={'task_id':task['id'],'sample':j,'domain':task['domain'],'source':task['source'],
                    'prompt':task['prompt'],'text':r['text'],'final_answer':final_answer(r['text']),
                    'token_ids':r['ids'],'tokens':len(r['ids']),'prompt_tokens':r['prefix'],'truncated':r['truncated'],
                    'generation_protocol':r.get('generation_protocol','tagged_single_pass'),
                    'reasoning_tokens':r.get('reasoning_tokens'),
                    'reasoning_truncated':r.get('reasoning_truncated'),
                    'answer_truncated':r.get('answer_truncated'),
                    'seconds':time.monotonic()-start,**scored}
                atomic_json(path,row)
            records.append(row)
    summary={'signature':signature,'config':config,'training_state':train_state,
        'decoding':'greedy' if n==1 else f"temperature={config.get('sampling_temperature',1.)}, top_p=1, top_k=0",
        'domains':summarize(records,n)}
    summary['evaluation_usage']={'wall_seconds':time.monotonic()-started,
        'generated_tokens':sum(r['tokens'] for r in records),'prompt_tokens':sum(r.get('prompt_tokens',0) for r in records),
        'reward_runtime':runtime.usage() if runtime else None}
    atomic_json(out/'summary.json',summary)
    return summary
