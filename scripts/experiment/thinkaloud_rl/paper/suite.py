"""One allocation, sequential subprocesses, immutable manifests, resumable stages."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import ROOT, atomic_json, digest, file_hash, read_jsonl, iter_jsonl, source_fingerprint, write_jsonl
from .data import select,verify
from .priors import build


def defaults(profile):
    cfg={'profile':profile,'models':['Qwen/Qwen2.5-1.5B-Instruct'], 'seeds':[17],
         'methods':['grpo','human','shuffled','uniform'],'train_domains':['programming'],
         'prior_source':'unified','held_out_participants':[],
         'steps':30,'prompts_per_batch':2,'group_size':4,'max_new_tokens':1024,
         'eval_max_new_tokens':2048,'max_prompt_tokens':8192,'eval_samples':1,'eval_limit_per_domain':24,
         'validation_limit_per_domain':8,'lr':2e-5,'lora_rank':16,'load_in_4bit':True,'device':'cuda',
         'gradient_checkpointing':True,'potential_scale':.1,'controller':'fixed','scale_cap':10.,
         'grad_clip':1.,'entropy_coefficient':.01,'t2t_alpha':.2,'checkpoint_every':5,'eval_every':10,
         'token_budget':0,'code_timeout':6,'audit_groups':8,'audit_tasks':2,'run_gradient_audit':True,
         'synthetic_tasks_per_domain':32,'offline':True,'external_baselines':[],
         'existing_adapters':[],'max_dynamic_groups':32,'overlong_buffer':256,
         'audit_trained_methods':[],'judge_model':None,'audit_domains':['programming']}
    if profile=='pilot':
        cfg.update(train_domains=['programming','writing','mixed'],steps=150,
                   prompts_per_batch=4,eval_every=15,validation_limit_per_domain=24,
                   eval_limit_per_domain=50,eval_samples=4,audit_groups=16,audit_tasks=4,
                   audit_domains=['programming','writing'],seeds=[17],minimum_training_seeds=1,
                   methods=['grpo','human_domain_topology','human_pooled_topology',
                            'shuffled_topology','llm_topology','small_world_topology','human'],
                   robust_pilot=True,held_out_participants=['P4','P7','P10','P11'],
                   human_diagnostic_participants=['P4','P7'],human_eval_partition='diagnostic',
                   evaluation_partition='pilot_diagnostic',
                   structure_embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                   structure_threshold=.7,structure_sensitivity_thresholds=[.6,.7,.8],
                   human_graph_window=32,min_graph_steps=4,max_graph_steps=128,
                   structure_reward_weight=.2,outcome_reward_weight=1.,
                   reward_judge_model='AtlaAI/Selene-1-Mini-Llama-3.1-8B',
                   reward_judge_backend='selene_absolute',
                   reward_judge_revision='427792f1c3e2073cb7da216924fd884b1ba496e0',
                   # Larger, different-family judge for evaluation (~25 GB in 4-bit).
                   evaluation_judge_model='prometheus-eval/prometheus-8x7b-v2.0',
                   evaluation_judge_backend='prometheus_absolute',
                   evaluation_judge_revision='2db013b60e3e91f7a06113e436410899768e8228',
                   judge_max_new_tokens=1536,judge_retry_max_new_tokens=3072,judge_max_prompt_tokens=16384,
                   audit_trained_methods=['grpo','human'],mixed_sampling='balanced',
                   validation_samples=2,validation_tail_points=4,
                   plateau_tolerance=.01,review_samples_per_run=8)
        # 1024 reasoning + 1024 answer tokens: measured writing reasoning reaches
        # ~1.5k tokens and answers exceeded 768 (docs/reports/2026-11-03_judge_and_budget_fixes.md).
        cfg.update(generation_protocol='two_stage_v1',reasoning_fraction=.5,max_new_tokens=2048,
                   sampling_temperature=.7,learning_preflight_steps=2,attention_implementation='sdpa',
                   training_stop_rule='fixed_batches',primary_methods=['grpo','human_domain_topology'])
        cfg.update(review_min_score=3,review_max_disagreement=1,review_max_risk_fraction=.1)
        cfg.update(judge_min_risk_recall=.8,judge_max_score_mae=.25,max_truncation_fraction=.25,
                   min_valid_graph_fraction=.7,max_cue_score_shift=.1,max_threshold_score_range=.2,
                   max_zero_gradient_steps=10)
    cfg['embedding_model']='jinaai/jina-embeddings-v2-small-en'
    if profile=='main':
        cfg.update(models=['Qwen/Qwen2.5-7B-Instruct','mistralai/Mistral-7B-Instruct-v0.3'],
            seeds=[17,29,43],methods=['grpo','grpo_sequence','dr_grpo','dapo_one_update','rloo','shrinkage','t2t','dra','human','shuffled','synthetic'],
            train_domains=['programming','writing','mixed'],steps=250,prompts_per_batch=4,
            max_new_tokens=2048,eval_max_new_tokens=4096,eval_samples=4,eval_limit_per_domain=0,
            validation_limit_per_domain=24,eval_every=50,checkpoint_every=5,audit_groups=32,audit_tasks=4,
            token_budget=0,audit_trained_methods=['grpo','human'])
    elif profile=='ablations':
        cfg.update(models=['Qwen/Qwen2.5-7B-Instruct'],seeds=[17,29,43],
            methods=['grpo','human','shuffled','uniform','frequency','transition','phase_free','synthetic','entropy','learned_prefix'],
            steps=250,prompts_per_batch=4,eval_samples=4,eval_limit_per_domain=100,
            token_budget=0,eval_every=25,audit_groups=16)
    elif profile=='existing':
        cfg.update(models=['Qwen/Qwen2.5-7B-Instruct'],seeds=[17,29,43],methods=[],train_domains=[],
                   run_gradient_audit=False,eval_samples=4,eval_limit_per_domain=50,eval_max_new_tokens=2048)
        for seed in cfg['seeds']:
            for method,old in [('grpo','baseline'),('human','unified')]:
                cfg['existing_adapters'].append({'id':f'existing_mixed_{method}_seed{seed}',
                    'model':cfg['models'][0],'seed':seed,'train_domain':'mixed','method':method,
                    'adapter':str(ROOT/f'pilots/results/legacy_runs/carc_multidomain_20261027_153910_2209184/mixed_{old}_seed{seed}/adapter')})
    elif profile!='pilot':raise ValueError(profile)
    return cfg


def create_plan(root,config,data_root):
    from .algorithms import METHODS
    # Importing algorithms imports torch, so the plan process exits before GPU work.
    root=Path(root).resolve();data_root=Path(data_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if any(m not in METHODS for m in config['methods']):raise ValueError('Unknown method')
    if not config['seeds'] or not config['models']:raise ValueError('Empty seed/model list')
    if config['prompts_per_batch']<2 or config['group_size']<2:raise ValueError('Use at least two prompts and samples per group')
    from .common import validate_training_schedule
    validate_training_schedule(config)
    primary=config.get('primary_methods',[])
    if any(m not in METHODS for m in primary):raise ValueError('Unknown primary method')
    if config.get('robust_pilot'):
        from .specialist_judges import validate_config
        validate_config(config)
        import math
        if config.get('generation_protocol','tagged_single_pass')=='two_stage_v1':
            if not 0<config.get('reasoning_fraction',.25)<1:raise ValueError('Invalid reasoning budget fraction')
        if not math.isfinite(config.get('sampling_temperature',1.)) or config.get('sampling_temperature',1.)<=0:
            raise ValueError('Invalid sampling temperature')
        if config.get('learning_preflight_steps',0)<0:raise ValueError('Negative learning preflight budget')
        if config.get('judge_retry_max_new_tokens',config['judge_max_new_tokens']*2)<config['judge_max_new_tokens']:
            raise ValueError('Judge retry token budget must not shrink')
        if any(not math.isfinite(config[k]) or config[k]<0 for k in ('structure_reward_weight','outcome_reward_weight')):
            raise ValueError('Reward weights must be finite and nonnegative')
        if config['outcome_reward_weight']<=0:raise ValueError('Keep a final-output reward in the joint objective')
        if not 0<config['structure_threshold']<1:raise ValueError('Invalid graph similarity threshold')
        if not 2<=config['min_graph_steps']<=config['human_graph_window']<=config['max_graph_steps']:
            raise ValueError('Invalid graph window bounds')
        if len(set(config['seeds']))<config.get('minimum_training_seeds',3):
            raise ValueError('Fewer distinct seeds than required by the configured protocol')
        if config['reward_judge_model']==config['evaluation_judge_model']:raise ValueError('Use separate training and evaluation judges')
        if any(m in config['models'] for m in (config['reward_judge_model'],config['evaluation_judge_model'])):
            raise ValueError('Policy must not judge its own final outputs')
        if not config['held_out_participants']:raise ValueError('Hold out human participants')
        diagnostic=set(config['human_diagnostic_participants']);held=set(config['held_out_participants'])
        if not diagnostic<=held or len(diagnostic)<2 or len(held-diagnostic)<2:
            raise ValueError('Use disjoint diagnostic/final human holdouts with at least two participants each')
        if config['mixed_sampling']=='balanced' and config['prompts_per_batch']%2:raise ValueError('Balanced batches require an even prompt count')
    jobs=[];omissions=[]
    if config.get('robust_pilot'):jobs.append({'id':'quality_preflight','action':'quality_preflight','seed':2026})
    for index,model in enumerate(config['models']):
        tag=model.replace('/','_')
        if config.get('robust_pilot'):
            jobs.append({'id':tag+'_structure_calibration','action':'structure_calibration','model':model,'seed':2026})
            if config.get('learning_preflight_steps',0):
                for domain in config['train_domains']:
                    jobs.append({'id':tag+'_learning_preflight_'+domain,'action':'learning_preflight',
                                 'model':model,'seed':config['seeds'][0],'train_domain':domain,
                                 'method':'human_domain_topology'})
        if 'synthetic' in config['methods']:
            jobs.append({'id':f'{tag}_synthetic','action':'synthetic','model':model,'seed':2026})
        if config['run_gradient_audit']:
            for domain in config.get('audit_domains',['programming']):
                jobs.append({'id':f'{tag}_gradient_audit_{domain}','action':'audit',
                             'model':model,'seed':2026,'audit_domain':domain})
        for seed in config['seeds']:
            jobs.append({'id':f'{tag}_base_seed{seed}','action':'base','model':model,'seed':seed})
            for domain in config['train_domains']:
                for method in config['methods']:
                    if method=='t2t' and domain!='programming':
                        omissions.append({'model':model,'seed':seed,'domain':domain,'method':method,
                                          'reason':'T2T requires verified binary rewards; writing training uses a proxy'})
                        continue
                    jobs.append({'id':f'{tag}_{domain}_{method}_seed{seed}','action':'train_eval',
                                 'model':model,'seed':seed,'train_domain':domain,'method':method})
                    if method in config.get('audit_trained_methods',[]):
                        parent=jobs[-1]['id']
                        for audited in (['programming','writing'] if domain=='mixed' else [domain]):
                            jobs.append({'id':parent+'_audit_'+audited,'action':'audit','model':model,
                                         'seed':2026,'checkpoint_run':parent,'audit_domain':audited})
    if primary:
        # Stable ordering keeps prerequisites first and each trained audit after
        # its parent. Finish the primary comparison across all seeds/domains
        # before starting additional methods; never filter by observed scores.
        training_jobs={j['id']:j for j in jobs if j['action']=='train_eval'}
        def priority(job):
            parent=training_jobs.get(job.get('checkpoint_run'),job)
            if parent['action']=='train_eval':return 1 if parent['method'] in primary else 2
            return 0
        jobs.sort(key=priority)
    for external in config.get('external_baselines',[]):
        if not isinstance(external.get('argv'),list) or not external['argv']:
            raise ValueError('External baselines require a concrete argv list, no shell interpolation')
        if not external.get('source_commit') or not external.get('source_url'):
            raise ValueError('External baseline needs source URL and pinned commit')
        jobs.append({'id':external['id'],'action':'external',**external})
    for existing in config.get('existing_adapters',[]):
        if not (Path(existing['adapter'])/'adapter_config.json').exists():raise ValueError('Missing existing adapter: '+existing['adapter'])
        jobs.append({**existing,'action':'existing'})
    if config.get('judge_model'):
        jobs.append({'id':'writing_judge','action':'judge','model':config['judge_model'],'seed':2026})
    if config.get('robust_pilot'):
        jobs.append({'id':'blinded_reviews','action':'review_packets','seed':2026})
    if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate run IDs')
    plan={'schema_version':1,'root':str(root),'data_root':str(data_root),'config':config,'jobs':jobs,'omissions':omissions,
          'source_hashes':source_fingerprint()}
    if config.get('robust_pilot'):
        training=sum(j['action']=='train_eval' for j in jobs)
        plan['declared_budget']={'training_runs':training,'responses_per_run_cap':config['steps']*config['prompts_per_batch']*config['group_size'],
            'training_stop_rule':config.get('training_stop_rule','steps_or_tokens' if config.get('token_budget') else 'fixed_batches'),
            'requested_batches_per_run':config['steps'],'cumulative_token_cap':config.get('token_budget',0),
            'primary_training_runs':sum(j['action']=='train_eval' and j['method'] in primary for j in jobs),
            'learning_preflight_runs':sum(j['action']=='learning_preflight' for j in jobs),
            'learning_preflight_response_cap':sum(j['action']=='learning_preflight' for j in jobs)*config.get('learning_preflight_steps',0)*config['prompts_per_batch']*config['group_size'],
            'policy_generated_tokens_per_run_upper_bound':config['steps']*config['prompts_per_batch']*config['group_size']*config['max_new_tokens'],
            'scope':'Per-run training caps exclude the separately listed learning preflights, evaluation, gradient audits, judge tokens, prompt tokens, rejected and resumed work; actual usage is reported.',
            'primary_comparison':'human_domain_topology vs grpo; pooled and shuffled comparisons are secondary',
            'primary_structure_endpoint':'held-out-human graph similarity with common evaluation reference across methods',
            'outcome_guardrails':['executable correctness','independent writing judge and blinded human risk review']}
    path=root/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Existing immutable plan differs. Use a new output directory.')
    atomic_json(path,plan)
    print(json.dumps({'root':str(root),'sequential_stages':len(jobs),'training_runs':sum(j['action']=='train_eval' for j in jobs),
                      'slurm_jobs_per_submission':1,'profile':config['profile'],'omissions':len(omissions)},indent=2))
    return plan


def prepare_plan(plan):
    from huggingface_hub import HfApi,snapshot_download
    from .data import prepare
    if plan['config'].get('robust_pilot') and any(
            plan['config'].get(k+'_backend','evidence_json')=='evidence_json'
            for k in ('reward_judge','evaluation_judge')):
        # Fail on missing/incompatible decoder dependencies before GPU allocation.
        from .constrained import json_prefix_filter
        from .output_quality import VALIDATOR
        VALIDATOR.check_schema(VALIDATOR.schema)
    root=Path(plan['root']);data=Path(plan['data_root'])
    prepare(data)
    build(root/'priors',held_out=plan['config'].get('held_out_participants',[]))
    models={}
    # Resolve and freeze revisions before allocation; resumes reuse exact snapshots.
    if (root/'prepared.json').exists():
        existing=json.loads((root/'prepared.json').read_text())
        if existing['plan_hash']!=digest(plan):raise ValueError('Prepared plan mismatch')
        return existing
    names=list(plan['config']['models'])
    if plan['config'].get('robust_pilot'):
        names.extend(plan['config'][k] for k in ('structure_embedding_model','reward_judge_model','evaluation_judge_model'))
    if 'dra' in plan['config']['methods']:names.append(plan['config']['embedding_model'])
    if plan['config'].get('judge_model'):names.append(plan['config']['judge_model'])
    for name in dict.fromkeys(names):
        pinned=[plan['config'].get(k+'_revision') for k in ('reward_judge','evaluation_judge')
                if plan['config'].get(k+'_model')==name and plan['config'].get(k+'_revision')]
        revision=pinned[0] if pinned else HfApi().model_info(name).sha
        snapshot_download(name,revision=revision,allow_patterns=['*.json','*.safetensors','*.bin','*.model','*.txt','*.jinja','*.py'])
        models[name]=revision
    prepared={'plan_hash':digest(plan),'models':models,'data_manifest_hash':file_hash(data/'manifest.json'),
              'task_file_hash':file_hash(data/'tasks.jsonl'),
              'prior_manifest_hash':file_hash(root/'priors/manifest.json'),
              'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}}
    if plan['config'].get('robust_pilot'):
        from .structure import build_references
        cfg={**plan['config'],'structure_embedding_revision':models[plan['config']['structure_embedding_model']]}
        build_references(cfg,root/'structure')
        prepared['structure_reference_hash']=file_hash(root/'structure/references.json')
        from .reviews import graph_packet
        graph_packet(root)
        from .data import select_evaluation
        prepared['evaluation_task_ids']=[r['id'] for r in select_evaluation(iter_jsonl(data/'tasks.jsonl'),plan['config'])]
    atomic_json(root/'prepared.json',prepared)
    return prepared


def require_pilot(plan):
    # Checked on the login node before sbatch and again by the worker.
    if not plan['config'].get('required_pilot_root'):return
    prerequisite=Path(plan['config']['required_pilot_root'])
    if not prerequisite.is_absolute():prerequisite=ROOT/prerequisite
    path=prerequisite/'pilot_readiness.json'
    checks=json.loads(path.read_text()).get('checks',{}) if path.exists() else {}
    if not checks or not all(checks.values()):
        missing=[k for k,v in checks.items() if not v]
        raise ValueError('Full training requires completed pilot checks and independent human reviews: '+str(path)+
                         (' (open: '+', '.join(missing)+')' if missing else ' (missing)'))


def check_prepared_plan(plan):
    """Cheap checks shared by login-node submission and the allocated worker."""
    require_pilot(plan)
    root=Path(plan['root']);prepared=json.loads((root/'prepared.json').read_text())
    if prepared['plan_hash']!=digest(plan):raise ValueError('Plan changed after preparation')
    if source_fingerprint()!=plan['source_hashes']:raise ValueError('Source code changed; use a new suite directory')
    return prepared


def run(plan,max_seconds=0):
    root=Path(plan['root']);prepared=check_prepared_plan(plan)
    verify(plan['data_root'])
    if file_hash(Path(plan['data_root'])/'manifest.json')!=prepared['data_manifest_hash']:raise ValueError('Data manifest changed')
    if file_hash(root/'priors/manifest.json')!=prepared['prior_manifest_hash']:raise ValueError('Human prior manifest changed')
    if plan['config'].get('robust_pilot') and file_hash(root/'structure/references.json')!=prepared['structure_reference_hash']:
        raise ValueError('Human graph references changed')
    with (root/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_jobs(plan,max_seconds)


def _run_jobs(plan,max_seconds):
    """Run while the caller holds the suite lock; release it on every exit path."""
    root=Path(plan['root'])
    child=None;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    start=time.monotonic()
    for job in plan['jobs']:
        marker=root/'status'/(job['id']+'.json')
        if marker.exists() and json.loads(marker.read_text()).get('state')=='complete':continue
        if stopping or (max_seconds and time.monotonic()-start>=max_seconds):return 75
        import uuid
        attempt=root/'attempts'/(job['id']+'_'+uuid.uuid4().hex+'.json')
        started_at=time.time()
        atomic_json(attempt,{'job':job['id'],'started_at':started_at,'state':'running'})
        atomic_json(marker,{'state':'running','job':job,'started_at':started_at})
        log_path=root/'logs'/(job['id']+'.log');log_path.parent.mkdir(parents=True,exist_ok=True)
        argv=[sys.executable,'-m','scripts.experiment.thinkaloud_rl.paper.worker','--plan',str(root/'plan.json'),'--job',job['id']]
        print('START',job['id'],'log:',log_path,flush=True)
        with log_path.open('a') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            while child.poll() is None:
                if max_seconds and time.monotonic()-start>=max_seconds and not stopping:stop()
                time.sleep(1)
            rc=child.returncode
        atomic_json(attempt,{'job':job['id'],'started_at':started_at,'finished_at':time.time(),
                            'wall_seconds':time.time()-started_at,'returncode':rc})
        if rc==75 or stopping:
            atomic_json(marker,{'state':'paused','job':job,'returncode':rc})
            from .report import aggregate
            aggregate(root)
            return 75
        if rc:
            atomic_json(marker,{'state':'failed','job':job,'returncode':rc})
            print('FAILED',job['id'],'see',log_path,file=sys.stderr)
            from .report import aggregate
            aggregate(root)
            return rc
        started=json.loads(marker.read_text())['started_at']
        atomic_json(marker,{'state':'complete','job':job,'started_at':started,'finished_at':time.time(),
                           'allocation_wall_seconds':time.time()-started})
        print('COMPLETE',job['id'],flush=True)
    from .report import aggregate,curves
    aggregate(root)
    curves(root)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    init=s.add_parser('init');init.add_argument('--root',type=Path,required=True);init.add_argument('--data-root',type=Path,required=True)
    init.add_argument('--profile',choices=['pilot','main','ablations','existing'],default='pilot');init.add_argument('--config',type=Path)
    for name in ('prepare','run','status','gate'):
        sub=s.add_parser(name);sub.add_argument('--root',type=Path,required=True)
        if name=='run':sub.add_argument('--max-seconds',type=int,default=0)
    args=p.parse_args()
    if args.command=='init':
        cfg=defaults(args.profile)
        if args.config:
            overrides=json.loads(args.config.read_text())
            # Historical configurations keep their original JSON judge protocol.
            for key in ('reward_judge','evaluation_judge'):
                if key+'_model' in overrides and key+'_backend' not in overrides:
                    cfg.pop(key+'_backend',None);cfg.pop(key+'_revision',None)
            cfg.update(overrides)
        create_plan(args.root,cfg,args.data_root);return
    plan=json.loads((args.root/'plan.json').read_text())
    if args.command=='prepare':prepare_plan(plan)
    elif args.command=='run':raise SystemExit(run(plan,args.max_seconds))
    elif args.command=='gate':check_prepared_plan(plan);print('Prepared plan, source code and pilot prerequisite verified')
    else:
        from collections import Counter
        states=Counter(json.loads(p.read_text())['state'] for p in (args.root/'status').glob('*.json'))
        print(json.dumps({'total':len(plan['jobs']),**states},indent=2))


if __name__=='__main__':main()
