"""One GPU stage per process, freeing model memory between sequential runs."""
import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import sys

from .common import atomic_json,digest,read_jsonl,iter_jsonl,stable_seed,write_jsonl
from .data import select,select_evaluation,task_rows


def prior_for(plan,job,variant):
    root=Path(plan['root'])/'priors'
    if variant=='synthetic':root=Path(plan['root'])/'synthetic'/job['model'].replace('/','_')/'priors'
    manifest=json.loads((root/'manifest.json').read_text())
    output={}
    for domain in ('programming','writing','math'):
        requested=plan['config'].get('prior_source','unified')
        if requested not in ('domain_specific','unified','programming','writing'):raise ValueError('Unknown prior_source')
        source=domain if requested=='domain_specific' and domain!='math' else 'unified' if requested=='domain_specific' else requested
        name=f'{source}_{variant}.json'
        bp=json.loads((root/name).read_text())
        if digest(bp)!=manifest['priors'][name]:raise ValueError('Prior artifact changed: '+name)
        output[domain]=bp
    return output


def execute(plan,job):
    import torch
    from transformers import set_seed
    from . import train as trainer
    from . import model as policy
    from .evaluate import evaluate
    from .priors import build
    from .algorithms import COGNITIVE
    signal.signal(signal.SIGUSR1,trainer.request_stop);signal.signal(signal.SIGTERM,trainer.request_stop)
    root=Path(plan['root']);data_root=Path(plan['data_root'])
    prepared=json.loads((root/'prepared.json').read_text())
    config={**plan['config'],**job,'model_revision':prepared['models'].get(job.get('model'))}
    if config.get('robust_pilot'):
        from .runtime import attach
        attach(config,plan,prepared)
    if 'dra' in config['methods']:
        config['embedding_revision']=prepared['models'][config['embedding_model']]
    # Read only relevant splits; multi-gigabyte test payloads stay off the heap
    # during preflight, calibration, audits and training.
    def rows(split=None):return task_rows(data_root,split)
    lcb_root=data_root/'upstream/LiveCodeBench'
    out=root/'runs'/job['id']
    if job['action']=='structure_calibration':
        from .structure import calibrate_llm
        tasks=select(rows('train'),'train',['programming','writing'],config['validation_limit_per_domain'])
        calibrate_llm(config,tasks,config['llm_reference_path'])
        return
    if job['action']=='quality_preflight':
        from .preflight import run
        run(config,root)
        return
    if job['action']=='review_packets':
        from .reviews import output_packet,graph_packet
        output_packet(root);graph_packet(root,include_model=True)
        return
    if job['action']=='existing':
        adapter=Path(job['adapter'])
        model,tok=policy.load(config,adapter)
        tasks=select_evaluation(rows(),config)
        evaluate(model,tok,tasks,{**config,'max_new_tokens':config['eval_max_new_tokens']},out/'test',lcb_root=lcb_root,
            train_state={'tokens':None,'seconds':None,'provenance':'Existing legacy adapter; see original training config',
                         'original_config':str(adapter.parent/'config.json')})
        return
    if job['action']=='external':
        # External reproduction consumes the SAME frozen tasks and outputs a named adapter.
        # No shell evaluation. The external environment and source commit must be explicit.
        env=__import__('os').environ.copy()
        env.update(TOPGRPO_PLAN=str(root/'plan.json'),TOPGRPO_DATA=str(data_root/'tasks.jsonl'),TOPGRPO_OUTPUT=str(out))
        subprocess.run(job['argv'],env=env,check=True)
        expected=out/'external_result.json'
        if not expected.exists():raise ValueError('External baseline must write external_result.json; see README schema')
        result=json.loads(expected.read_text())
        if not all(k in result for k in ('adapter','model','seed','training_tokens','training_seconds','train_task_ids')):
            raise ValueError('Incomplete external baseline provenance')
        training_ids={r['id'] for r in rows('train')}
        if set(result['train_task_ids'])-training_ids:
            raise ValueError('External baseline used non-training tasks')
        config.update(model=result['model'],seed=result['seed'],model_revision=prepared['models'][result['model']])
        adapter=Path(result['adapter']);model,tok=policy.load(config,adapter)
        test=select_evaluation(rows(),config)
        evaluate(model,tok,test,{**config,'max_new_tokens':config['eval_max_new_tokens']},out/'test',lcb_root=lcb_root,
                 train_state={'tokens':result['training_tokens'],'seconds':result['training_seconds'],
                              'source_commit':job['source_commit'],'source_url':job['source_url']})
        return
    if job['action']=='judge':
        from .judge import packet,judge_local,import_scores
        review=root/'writing_review'
        packet(root,review)
        judge_local(review/'packet.jsonl',review/'scores',job['model'],config['model_revision'],config['load_in_4bit'])
        import_scores(review/'packet.jsonl',review/'private_key.jsonl',review/'scores',root/'writing_scores.jsonl')
        return
    if job['action']=='synthetic':
        out=root/'synthetic'/job['model'].replace('/','_')
        out.mkdir(parents=True,exist_ok=True)
        selected=select(rows('train'),'train',['programming','writing'],config['synthetic_tasks_per_domain'])
        model,tok=policy.load(config)
        records=[]
        for task in selected:
            path=out/'samples'/(digest(task['id'])+'.json')
            if path.exists():record=json.loads(path.read_text())
            else:
                set_seed(stable_seed('synthetic',task['id'],config['seed']))
                r=policy.sample(model,tok,task,config)
                match=re.search(r'<think>(.*?)</think>',r['text'],re.S)
                record={'id':task['id'],'domain':task['domain'],'split':'train',
                    'text':match[1] if match else '', 'full_text':r['text'],'complete_trace':bool(match),
                    'model':job['model'],'model_revision':config['model_revision']}
                atomic_json(path,record)
            records.append(record)
            if trainer.STOP:raise InterruptedError
        usable=[r for r in records if r['complete_trace']]
        for d in ('programming','writing'):
            if not any(r['domain']==d for r in usable):raise ValueError('No complete synthetic traces in '+d)
        write_jsonl(out/'traces.jsonl',usable)
        build(out/'priors',config.get('held_out_participants',[]),out/'traces.jsonl')
        atomic_json(out/'coverage.json',{'total':len(records),'complete':len(usable)})
        return
    if job['action']=='audit':
        from .audit import gradient_audit
        domain=job.get('audit_domain','programming')
        tasks=select(rows('val'),'val',[domain],config['audit_tasks'])
        priors={v:prior_for(plan,job,v)[domain] for v in ('human','shuffled','uniform')}
        adapter=None
        if job.get('checkpoint_run'):
            latest=json.loads((root/'runs'/job['checkpoint_run']/'latest.json').read_text())
            adapter=Path(latest['checkpoint'])/'adapter'
        gradient_audit(config,tasks,priors,root/'audits'/job['id'],lcb_root,adapter=adapter)
        return
    if job['action'] in ('train_eval','learning_preflight'):
        probe=job['action']=='learning_preflight'
        domains=['programming','writing'] if job['train_domain']=='mixed' else [job['train_domain']]
        tasks=select(rows('train'),'train',domains)
        if probe:
            out=root/'learning_preflight'/job['id']
            config.update(steps=config['learning_preflight_steps'],checkpoint_every=1,token_budget=0)
        else:tasks+=select(rows('val'),'val',domains,config['validation_limit_per_domain'])
        variant=job['method'] if job['method'] in COGNITIVE else 'human'
        priors=prior_for(plan,job,variant)
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'prior_snapshot.json',priors)
        config['prior_hash']=digest(priors)
        config['training_task_hash']=digest(tasks)
        if not trainer.train(config,tasks,priors,out,lcb_root):raise InterruptedError
        if probe:
            state=json.loads((out/'complete.json').read_text())
            from .diagnostics import training_report
            health=training_report(out)['health']
            truncation=health['mean_truncation_fraction'];coverage=health['valid_structure_fraction']
            checks={'nonzero_optimizer_updates':state.get('optimizer_updates',0)>0,
                    'finite_metrics':health['finite_logged_metrics'],
                    'truncation_within_limit':truncation is not None and truncation<=config.get('max_truncation_fraction',.25),
                    'valid_graph_coverage':coverage is not None and coverage>=config.get('min_valid_graph_fraction',.7)}
            passed=all(checks.values())
            atomic_json(out/'learning_gate.json',{'passed':passed,'checks':checks,
                        'optimizer_updates':state.get('optimizer_updates',0),'health':health,
                        'batches':state['step'],'limitation':'A nonzero update is a viability check, not evidence of improvement.'})
            if not passed:
                reason='no optimizer updates' if not checks['nonzero_optimizer_updates'] else ', '.join(k for k,v in checks.items() if not v)
                raise ValueError('Learning preflight failed: '+reason+'. Inspect rewards, reasoning coverage and generation budgets before the main pilot: '+str(out))
            return
        # train() has returned: release its cached GPU buffers before a fresh eval load.
        import gc
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if trainer.STOP:raise InterruptedError
        evaluate_only(plan,job)
        return
    if job['action']=='base':
        model,tok=policy.load(config)
        tasks=select_evaluation(rows(),config)
        evaluate(model,tok,tasks,{**config,'max_new_tokens':config['eval_max_new_tokens']},out/'test',lcb_root=lcb_root)


def evaluate_only(plan,job):
    from . import train as trainer
    from .model import load
    from .evaluate import evaluate
    signal.signal(signal.SIGUSR1,trainer.request_stop);signal.signal(signal.SIGTERM,trainer.request_stop)
    root=Path(plan['root']);out=root/'runs'/job['id']
    config=json.loads((out/'config.json').read_text())
    latest=json.loads((out/'latest.json').read_text())
    model,tok=load(config,Path(latest['checkpoint'])/'adapter')
    rows=iter_jsonl(Path(plan['data_root'])/'tasks.jsonl')
    tasks=select_evaluation(rows,config)
    state=json.loads((out/'complete.json').read_text())
    evaluate(model,tok,tasks,{**config,'max_new_tokens':config['eval_max_new_tokens']},out/'test',
        lcb_root=Path(plan['data_root'])/'upstream/LiveCodeBench',train_state=state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    job=next(j for j in plan['jobs'] if j['id']==args.job)
    try:
        if args.evaluate_only:evaluate_only(plan,job)
        else:execute(plan,job)
    except InterruptedError:
        raise SystemExit(75)
