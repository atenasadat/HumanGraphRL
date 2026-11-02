"""Frozen-policy gradient audit and blinded cognitive-label review packets."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import statistics

from .common import ROOT, atomic_json, digest, read_jsonl, stable_seed, write_jsonl
from ..reasoning_shaper import PURE_MOVES, classify_reasoning_step, segment_thought_stream


def gradient_audit(config,tasks,priors,out,lcb_root=None,adapter=None):
    import torch
    from transformers import set_seed
    from .model import load,sample
    from .scoring import score
    from .train import gradients
    from .algorithms import GradientMoments,advantages
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec={'config':config,'tasks':tasks,'priors':priors,'adapter':str(adapter) if adapter else None}
    if (out/'spec.json').exists() and json.loads((out/'spec.json').read_text())!=spec:
        raise ValueError('Gradient audit inputs changed')
    atomic_json(out/'spec.json',spec)
    set_seed(config['seed'])
    model,tok=load(config,adapter=adapter,trainable=True);model._paper_tokenizer=tok
    repeats=config.get('audit_groups',16)
    runtime=None
    if config.get('robust_pilot'):
        from .runtime import RewardRuntime
        runtime=RewardRuntime({**config,'method':'grpo'},'training')
    if repeats<4:raise ValueError('At least four groups required for calibration/measurement split')
    reports=[]
    for task in tasks:
        dest=out/(digest(task['id'])+'.json')
        if dest.exists():reports.append(json.loads(dest.read_text()));continue
        moments={name:[GradientMoments(),GradientMoments()] for name in priors}
        for rep in range(repeats):
            from . import train as trainer
            if trainer.STOP:raise InterruptedError
            set_seed(stable_seed(config['seed'],'gradient_audit',task['id'],rep))
            group=[];rewards=[]
            for j in range(config['group_size']):
                r=sample(model,tok,task,config);r['domain']=task['domain']
                reward=(runtime.assess(r['text'],task) if runtime else score(r['text'],task,lcb_root=lcb_root,timeout=config.get('code_timeout',6)))['reward']
                group.append(r);rewards.append(reward)
                atomic_json(out/'rollouts'/f'{digest([task["id"],rep,j])}.json',
                    {'task_id':task['id'],'group':rep,'sample':j,'text':r['text'],'reward':reward,'token_ids':r['ids']})
            adv=advantages([rewards])
            for variant,bp in priors.items():
                audit_config={**config,'method':variant}
                gt,gs,_,_=gradients(model,[group],[rewards],adv,{task['domain']:bp},audit_config)
                x=torch.cat([g.detach().float().cpu().flatten() for g in gt])
                y=torch.cat([g.detach().float().cpu().flatten() for g in gs])
                moments[variant][int(rep>=repeats//2)].add(x,y)
        result={'task_id':task['id'],'checkpoint':str(adapter) if adapter else 'untrained_LoRA_on_frozen_base','variants':{}}
        for name,(calibration,heldout) in moments.items():
            calibration_report=calibration.report(config.get('potential_scale',.1))
            # Estimate on independent groups, then evaluate without refitting.
            fitted=max(-10.,min(10.,calibration_report['optimal_scale_exploratory']))
            result['variants'][name]={'calibration':calibration_report,
                'heldout_fixed_scale':heldout.report(config.get('potential_scale',.1)),
                'heldout_calibrated_scale':heldout.report(fitted)}
        atomic_json(dest,result);reports.append(result)
    atomic_json(out/'summary.json',{'tasks':reports,'interpretation':
        'Variance is within task at a fixed policy, before gradient clipping/Adam. '
        'No automatic efficacy claim or publication gate is inferred from this small audit.'})


def label_packet(output,model_samples=None,per_stratum=40,seed=2026):
    here=ROOT/'processed_data/cognitive_priors_and_audits_from_transcripts';candidates=[]
    for r in json.loads((here/'verbalizations_extracted_segments.json').read_text()):
        for u in segment_thought_stream(r['text']):
            candidates.append({'text':u,'source':'human_programming','participant':r['participant']})
    for p in (ROOT/'processed_data/writing_task_segments_extracted_from_transcripts').glob('*_proposal_verbalizations.json'):
        for r in json.loads(p.read_text())['segments']:
            candidates.append({'text':r['text'],'source':'human_writing','participant':p.name.split('_')[0]})
    if model_samples:
        for p in Path(model_samples).rglob('samples/*.json'):
            r=json.loads(p.read_text())
            if 'text' not in r:continue
            trace=re.search(r'<think>(.*?)</think>',r['text'],re.S)
            if not trace:continue
            for u in segment_thought_stream(trace[1]):
                candidates.append({'text':u,'source':'model_'+r['domain'],'participant':None})
    strata=defaultdict(list);seen=set()
    for r in candidates:
        key=digest(r['text'])
        if key in seen:continue
        seen.add(key);r['prediction']=classify_reasoning_step(r['text'])
        strata[(r['source'],r['prediction'])].append(r)
    rng=random.Random(seed);selected=[]
    for key,rows in sorted(strata.items()):
        rng.shuffle(rows)
        for r in rows[:per_stratum]:
            selected.append({**r,'id':digest([seed,r['text']]),'stratum_size':len(rows),
                             'sampled_stratum_size':min(per_stratum,len(rows))})
    rng.shuffle(selected)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'private_key.jsonl',selected)
    write_jsonl(out/'reviewer_a.jsonl',[{'id':r['id'],'text':r['text'],'label':None,'notes':''} for r in selected])
    write_jsonl(out/'reviewer_b.jsonl',[{'id':r['id'],'text':r['text'],'label':None,'notes':''} for r in selected])
    atomic_json(out/'instructions.json',{'labels':[*PURE_MOVES,'unknown','mixed'],
        'instruction':'Label the dominant cognitive action; use unknown or mixed when unsupported. '
        'Review independently, blinded to source and automatic prediction. Do not treat wording alone as correctness.',
        'sampling':'Source x predicted-label stratified; inverse-inclusion weights included in private key',
        'status':'pending independent judgments; this script does not create gold labels'})


def label_report(key,a,b,output):
    key={r['id']:r for r in read_jsonl(key)}
    a={r['id']:r['label'] for r in read_jsonl(a)};b={r['id']:r['label'] for r in read_jsonl(b)}
    labels=(*PURE_MOVES,'unknown','mixed')
    if set(a)!=set(key) or set(b)!=set(key):raise ValueError('Review IDs must exactly match packet')
    if any(x not in labels for d in (a,b) for x in d.values()):raise ValueError('Incomplete/invalid labels')
    agreement=sum(a[i]==b[i] for i in key)/len(key)
    ca,cb=Counter(a.values()),Counter(b.values());chance=sum(ca[l]*cb[l] for l in labels)/len(key)**2
    accepted={i:a[i] for i in key if a[i]==b[i] and a[i] in PURE_MOVES}
    confusion={g:{p:0. for p in PURE_MOVES} for g in PURE_MOVES}
    for i,gold in accepted.items():
        row=key[i];weight=row['stratum_size']/row['sampled_stratum_size']
        confusion[gold][row['prediction']]+=weight
    per_class={}
    for label in PURE_MOVES:
        tp=confusion[label][label];pred=sum(confusion[g][label] for g in PURE_MOVES);actual=sum(confusion[label].values())
        precision=tp/pred if pred else 0.;recall=tp/actual if actual else 0.
        per_class[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
    atomic_json(output,{'agreement_unweighted':agreement,'cohens_kappa_unweighted':(agreement-chance)/(1-chance) if chance<1 else None,
        'consensus_coverage':len(accepted)/len(key),'weighted_confusion_on_consensus':confusion,
        'per_class_on_consensus':per_class,'macro_f1_on_consensus':statistics.mean(r['f1'] for r in per_class.values()),
        'limitation':'Disagreements/unknowns are reported, not converted into gold. Consensus subset can be easier.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('packet');pack.add_argument('--output',type=Path,required=True);pack.add_argument('--model-samples',type=Path)
    pack.add_argument('--per-stratum',type=int,default=40)
    report=sub.add_parser('report')
    for name in ('key','a','b','output'):report.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.command=='packet':label_packet(args.output,args.model_samples,args.per_stratum)
    else:label_report(args.key,args.a,args.b,args.output)
