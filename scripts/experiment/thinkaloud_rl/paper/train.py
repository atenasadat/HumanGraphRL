"""Single-GPU matched experiments, with atomic optimizer/RNG checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import uuid

import numpy as np
import torch
from transformers import set_seed

from . import model as policy
from .algorithms import COGNITIVE, METHODS, advantages, prefix_arrays, t2t_rewards, dra_rewards,dapo_soft_overlong
from .structure import TOPOLOGY
from .common import atomic_json, digest, read_jsonl, stable_seed
from .scoring import score

STOP=False


def request_stop(*_):
    global STOP
    STOP=True


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint(out,model,tok,optimizer,state,value_model=None,value_optimizer=None):
    root=Path(out)/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    temporary=root/('.tmp_'+uuid.uuid4().hex);temporary.mkdir()
    model.save_pretrained(temporary/'adapter');tok.save_pretrained(temporary/'adapter')
    state={**state,'rng':rng_state(),'optimizer':optimizer.state_dict(),
           'value_model':value_model.state_dict() if value_model is not None else None,
           'value_optimizer':value_optimizer.state_dict() if value_optimizer is not None else None}
    torch.save(state,temporary/'state.pt')
    destination=root/f'step_{state["step"]:06d}_{uuid.uuid4().hex[:8]}'
    os.replace(temporary,destination)
    atomic_json(Path(out)/'latest.json',{'checkpoint':str(destination.resolve()),'step':state['step']})
    atomic_json(Path(out)/'progress.json',{k:v for k,v in state.items()
                if k not in ('rng','optimizer','value_model','value_optimizer')})
    # Keep the two most recent committed checkpoints; evaluation artifacts persist separately.
    old=sorted([p for p in root.glob('step_*') if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in old[:-2]:shutil.rmtree(p)
    return destination


def gradients(model,rollouts,rewards,adv,prior,config,value_model=None):
    params=[p for p in model.parameters() if p.requires_grad]
    task_grad=[torch.zeros_like(p) for p in params]
    shape_grad=[torch.zeros_like(p) for p in params]
    value_data=[];task_loss_value=0.;entropy_value=0.
    batch=len(rollouts);group=len(rollouts[0]);method=config['method'];horizon=config['max_new_tokens']
    total_tokens=sum(len(r['ids']) for g in rollouts for r in g)
    for i,group_rollouts in enumerate(rollouts):
        for j,r in enumerate(group_rollouts):
            logp,all_logp=policy.log_probs(model,r)
            potential,features=prefix_arrays(model_tokenizer(model),r['ids'],prior[r['domain']],horizon,r['prompt_opens_think']) if method in COGNITIVE or value_model is not None else (None,None)
            device=logp.device
            a=adv[i,j].to(device)
            if value_model is not None:
                f=torch.tensor(features,device=device,dtype=torch.float32)
                with torch.no_grad(): a=float(rewards[i][j])-value_model(f).squeeze(-1)
                value_data.append((f.detach().cpu(),float(rewards[i][j])))
            divisor=len(r['ids']) if method=='grpo_sequence' else horizon
            denominator=total_tokens if method=='dapo_one_update' else divisor*batch*group
            loss=-(logp*a).sum()/denominator
            entropy=-(all_logp.exp()*all_logp).sum(-1).sum()/(horizon*batch*group)
            if method=='entropy':loss=loss-config.get('entropy_coefficient',.01)*entropy
            tg=torch.autograd.grad(loss,params,retain_graph=method in COGNITIVE)
            for acc,g in zip(task_grad,tg):acc.add_(g.detach())
            task_loss_value+=float(loss.detach());entropy_value+=float(entropy.detach())
            if method in COGNITIVE:
                # Return-to-go of gamma=1 zero-terminal potential shaping is -Phi(prefix).
                # Fixed divisor preserves the zero-mean baseline property before clipping/Adam.
                shape_adv=-torch.tensor(potential,device=device,dtype=logp.dtype)
                shape_loss=-(logp*shape_adv).sum()/(horizon*batch*group)
                sg=torch.autograd.grad(shape_loss,params)
                for acc,g in zip(shape_grad,sg):acc.add_(g.detach())
    return task_grad,shape_grad,{'loss':task_loss_value,'token_entropy_budget_normalized':entropy_value},value_data


def model_tokenizer(model):
    return model._paper_tokenizer


def train(config,tasks,priors,out,lcb_root=None):
    global STOP
    STOP=False
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGUSR1,request_stop)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config_hash=digest(config)
    if config['method'] not in METHODS:raise ValueError(config['method'])
    if config['method'] in TOPOLOGY and not config.get('robust_pilot'):
        raise ValueError('Direct topology methods require the declared robust reward runtime')
    if config['method']=='t2t' and any(t['domain']=='writing' for t in tasks):
        raise ValueError('T2T is restricted to binary verified coding/math rewards')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=config:raise ValueError('Immutable run config changed: '+str(out))
    else:atomic_json(out/'config.json',config)
    if (out/'complete.json').exists():return True
    set_seed(config['seed'])
    latest=json.loads((out/'latest.json').read_text()) if (out/'latest.json').exists() else None
    adapter=Path(latest['checkpoint'])/'adapter' if latest else None
    model,tok=policy.load(config,adapter,trainable=True)
    model._paper_tokenizer=tok
    encoder=None
    if config['method']=='dra':
        from sentence_transformers import SentenceTransformer
        encoder=SentenceTransformer(config['embedding_model'],revision=config['embedding_revision'],
            device='cpu',trust_remote_code=True,local_files_only=config.get('offline',True))
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=0.)
    value_model=value_optimizer=None
    if config['method']=='learned_prefix':
        value_model=torch.nn.Sequential(torch.nn.Linear(7,32),torch.nn.Tanh(),torch.nn.Linear(32,1),torch.nn.Sigmoid()).to(next(model.parameters()).device)
        value_optimizer=torch.optim.Adam(value_model.parameters(),lr=.001)
    state={'step':0,'tokens':0,'seconds':0.,'scale':config.get('potential_scale',.1),'config_hash':config_hash}
    if config.get('robust_pilot'):
        state.update(prompt_tokens=0,rollouts=0,optimizer_updates=0,validation_seconds=0.,reward_usage={},peak_gpu_allocated_bytes=None,
                     consecutive_zero_task_updates=0)
    if latest:
        saved=torch.load(Path(latest['checkpoint'])/'state.pt',map_location='cpu',weights_only=False)
        if saved['config_hash']!=config_hash:raise ValueError('Checkpoint/config mismatch')
        optimizer.load_state_dict(saved.pop('optimizer'))
        if value_model is not None:
            value_model.load_state_dict(saved['value_model']);value_optimizer.load_state_dict(saved['value_optimizer'])
        restore_rng(saved.pop('rng'))
        state={k:saved[k] for k in state}
    else:
        checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
    from .evaluate import evaluate
    runtime=None
    if config.get('robust_pilot'):
        from .runtime import RewardRuntime
        before=rng_state()
        runtime=RewardRuntime(config,'training')
        restore_rng(before)
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    usage_base=state.get('reward_usage',{})
    def usage():
        current=runtime.usage() if runtime else {}
        def merge(a,b):
            return {k:merge(a.get(k,{}),v) if isinstance(v,dict) else
                    a.get(k,0)+v if isinstance(v,(int,float)) else v for k,v in b.items()}
        return merge(usage_base,current)
    def judge_guard(action, before):
        from .output_quality import JudgeResponseError
        try:
            return action()
        except (JudgeResponseError, InterruptedError) as exc:
            # The current batch has not updated the optimizer. Replay it with the
            # same RNG; retain all previously committed weights and optimizer state.
            restore_rng(before)
            checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
            atomic_json(out/'judge_failure.json',{'step':state['step'],'error':str(exc),
                        'resume':'Replay incomplete batch or validation from the committed state'})
            from .diagnostics import training_report
            training_report(out)
            raise
    val_tasks=[t for t in tasks if t['split']=='val']
    train_tasks=[t for t in tasks if t['split']=='train']
    if not train_tasks:raise ValueError('No training tasks')
    if len(train_tasks)<config['prompts_per_batch']:raise ValueError('Too few distinct prompts')
    budget=config.get('token_budget',0)
    validated_step=None
    def validate_current():
        nonlocal validated_step
        before=rng_state();validation_start=time.monotonic()
        atomic_json(out/'pending_validation.json',{'step':state['step']})
        judge_guard(lambda:evaluate(model,tok,val_tasks,config,out/'validation'/f'step_{state["step"]:06d}',lcb_root=lcb_root,
                 samples=config.get('validation_samples',2 if runtime else 1),train_state=state.copy(),runtime=runtime),before)
        restore_rng(before)
        if runtime:state['validation_seconds']+=time.monotonic()-validation_start
        (out/'pending_validation.json').unlink()
        validated_step=state['step']
    if val_tasks and ((out/'pending_validation.json').exists() or (runtime and state['step']==0)):
        if (out/'pending_validation.json').exists():
            pending=json.loads((out/'pending_validation.json').read_text())
            if pending['step']!=state['step']:raise ValueError('Pending validation/checkpoint step mismatch')
        validate_current()
    while state['step']<config['steps'] and (not budget or state['tokens']<budget):
        if STOP:break
        step_rng=rng_state()
        start=time.monotonic();step=state['step']+1
        rng=random.Random(stable_seed(config['seed'],step,'prompt_order'))
        selected=rng.sample(train_tasks,config['prompts_per_batch'])
        if config.get('mixed_sampling')=='balanced' and config.get('train_domain')=='mixed':
            half=config['prompts_per_batch']//2
            selected=[t for domain in ('programming','writing')
                      for t in rng.sample([x for x in train_tasks if x['domain']==domain],half)]
        if config['method']=='dapo_one_update':
            pool=[t for t in train_tasks if t not in selected];rng.shuffle(pool)
            selected+=pool[:config.get('max_dynamic_groups',32)-len(selected)]
        rollouts=[];rewards=[];saved_rows=[]
        for task in selected:
            group=[];group_rewards=[]
            for j in range(config['group_size']):
                if STOP:
                    # No update yet: discard this partial batch and reproduce it on resume.
                    restore_rng(step_rng)
                    checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
                    return False
                r=policy.sample(model,tok,task,config);r['domain']=task['domain']
                scored=judge_guard(lambda:runtime.assess(r['text'],task),step_rng) if runtime else score(r['text'],task,lcb_root=lcb_root,timeout=config.get('code_timeout',6))
                if scored['reward'] is None:raise ValueError('Training requires a defined reward')
                if not math.isfinite(scored['reward']):raise ArithmeticError('Nonfinite training reward')
                group.append(r);group_rewards.append(scored['reward'])
                saved_rows.append({'step':step,'task_id':task['id'],'domain':task['domain'],'sample':j,
                    'text':r['text'],'prompt':task['prompt'],'token_ids':r['ids'],'prompt_tokens':r['prefix'],
                    'truncated':r['truncated'],**scored})
            effective_group=group_rewards
            if config['method']=='dapo_one_update':
                effective_group=[dapo_soft_overlong(y,len(r['ids']),config['max_new_tokens'],
                    min(config.get('overlong_buffer',256),config['max_new_tokens'])) for y,r in zip(group_rewards,group)]
                if max(effective_group)-min(effective_group)<=1e-12:
                    for row in saved_rows[-len(group):]:row['dynamic_sampling_rejected']=True
                    continue
            rollouts.append(group);rewards.append(effective_group)
            if len(rollouts)>=config['prompts_per_batch']:break
        if STOP:
            restore_rng(step_rng)
            checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
            return False
        if not rollouts:
            # Charge discarded generation to the budget; never pretend this was an update.
            state.update(step=step,tokens=state['tokens']+sum(len(r['token_ids']) for r in saved_rows),
                         seconds=state['seconds']+time.monotonic()-start)
            atomic_json(out/'steps'/f'{step:06d}.json',{'metrics':{**state,'skipped':'all_dynamic_groups_equal'},'rollouts':saved_rows})
            if step%config.get('checkpoint_every',5)==0:checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
            continue
        effective=t2t_rewards(rewards,[[len(r['ids']) for r in g] for g in rollouts],config['max_new_tokens'],config.get('t2t_alpha',.2)) if config['method']=='t2t' else rewards
        if encoder is not None:
            effective=dra_rewards(rewards,encoder.encode([r['text'] for g in rollouts for r in g],show_progress_bar=False))
        adv=advantages(effective,config['method'])
        gt,gs,metrics,value_data=gradients(model,rollouts,rewards,adv,priors,config,value_model)
        nt=sum(float(g.float().square().sum()) for g in gt)**.5
        ns=sum(float(g.float().square().sum()) for g in gs)**.5
        if runtime:state['consecutive_zero_task_updates']=state['consecutive_zero_task_updates']+1 if nt<=1e-12 else 0
        scale=state['scale'] if config['method'] in COGNITIVE else 0.
        cosine=sum(float((a.float()*b.float()).sum()) for a,b in zip(gt,gs))/(nt*ns) if nt*ns else None
        optimizer.zero_grad(set_to_none=True)
        for p,a,b in zip(params,gt,gs):p.grad=a+scale*b
        norm=float(torch.nn.utils.clip_grad_norm_(params,config.get('grad_clip',1.),error_if_nonfinite=True))
        if norm:
            optimizer.step()
            if runtime:state['optimizer_updates']+=1
        if value_model is not None:
            value_optimizer.zero_grad()
            for features,target in value_data:
                device=next(value_model.parameters()).device
                loss=(value_model(features.to(device)).squeeze(-1)-target).square().mean()/len(value_data)
                loss.backward()
            value_optimizer.step()
        # Legacy norm-share controller, strictly lagged; fixed scale is the default experiment.
        if config.get('controller','fixed')=='lagged_norm':
            state['scale']=min(config.get('scale_cap',10.),.35/.65*nt/ns) if nt>1e-12 and ns>1e-12 else 0.
        generated=sum(len(r['token_ids']) for r in saved_rows)
        state.update(step=step,tokens=state['tokens']+generated,seconds=state['seconds']+time.monotonic()-start)
        if runtime:
            state['prompt_tokens']+=sum(r['prompt_tokens'] for r in saved_rows)
            state['rollouts']+=len(saved_rows);state['reward_usage']=usage()
            state['peak_gpu_allocated_bytes']=max(state.get('peak_gpu_allocated_bytes') or 0,torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        row={**metrics,'step':step,'generated_tokens':generated,'cumulative_tokens':state['tokens'],
            'training_seconds':state['seconds'],'mean_reward':float(np.mean(rewards)),
            'mean_verifier_reward_all_generated':float(np.mean([r['reward'] for r in saved_rows])),
            'task_grad_norm':nt,'shape_grad_norm':ns,'scale':scale,'cosine':cosine,'preclip_norm':norm,
            'shape_norm_share':scale*ns/(nt+scale*ns) if nt+scale*ns else 0.,
            'no_task_gradient':nt<=1e-12,'shape_only_update':nt<=1e-12 and scale*ns>0,
            'dynamic_groups_rejected':sum(r.get('dynamic_sampling_rejected',False) for r in saved_rows)/config['group_size'],
            'truncated_fraction':sum(r['truncated'] for r in saved_rows)/len(saved_rows)}
        if runtime:
            row.update(mean_output_reward=float(np.mean([r['outcome_reward'] for r in saved_rows])),
                       mean_structure_bonus=float(np.mean([r['structure_bonus'] for r in saved_rows])),
                       output_risk_fraction=float(np.mean([r['output_assessment']['risk_detected'] for r in saved_rows])),
                       reward_usage=state['reward_usage'],prompt_tokens=state['prompt_tokens'],rollouts=state['rollouts'],
                       peak_gpu_allocated_bytes=state['peak_gpu_allocated_bytes'])
        atomic_json(out/'steps'/f'{step:06d}.json',{'metrics':row,'rollouts':saved_rows})
        if runtime and state['consecutive_zero_task_updates']>=config.get('max_zero_gradient_steps',10):
            checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
            from .diagnostics import training_report
            training_report(out)
            raise RuntimeError('No outcome-driven gradient for consecutive batches; inspect reward/format/difficulty before spending more pilot compute')
        print(json.dumps({'run':out.name,**row}),flush=True)
        if step%config.get('checkpoint_every',5)==0 or STOP:
            checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
        if val_tasks and step%config.get('eval_every',20)==0 and not STOP:
            validate_current()
    if runtime and val_tasks and not STOP and state['step']%config.get('eval_every',20) and validated_step!=state['step']:
        validate_current()
    if runtime:state['reward_usage']=usage()
    checkpoint(out,model,tok,optimizer,state,value_model,value_optimizer)
    complete=state['step']>=config['steps'] or bool(budget and state['tokens']>=budget)
    if complete:atomic_json(out/'complete.json',{**state,'stop_reason':'token_budget' if budget and state['tokens']>=budget else 'steps'})
    from .diagnostics import training_report
    training_report(out)
    return complete


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--priors',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lcb-root',type=Path)
    a=p.parse_args()
    success=train(json.loads(a.config.read_text()),read_jsonl(a.tasks),json.loads(a.priors.read_text()),a.output,a.lcb_root)
    raise SystemExit(0 if success else 75)
