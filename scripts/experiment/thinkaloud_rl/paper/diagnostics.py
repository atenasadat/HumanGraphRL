"""Measured budgets, validation trajectories, robustness and honest readiness gates."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from .common import atomic_json


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def trend(points, tolerance=.01, count=4):
    tail=points[-count:]
    if len(tail)<count:
        return {'status':'insufficient_validation_points','points':len(points),'tail_delta':None}
    xs=[p[0] for p in tail];ys=[p[1] for p in tail]
    xm=sum(xs)/len(xs);ym=sum(ys)/len(ys)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/max(sum((x-xm)**2 for x in xs),1e-12)
    delta=slope*(max(xs)-min(xs))
    return {'status':'still_improving_extend_horizon' if delta>tolerance else
            'degrading_investigate' if delta < -tolerance else 'flat_tail_not_proof_of_convergence',
            'points':len(points),'tail_delta':delta,'tail_min':min(ys),'tail_max':max(ys)}


def training_report(out):
    out=Path(out);config=read(out/'config.json',{})
    records=[read(p) for p in sorted((out/'steps').glob('*.json'))]
    completed=read(out/'complete.json',read(out/'progress.json',{}))
    validations=[];series=defaultdict(list)
    for p in sorted((out/'validation').glob('*/summary.json')):
        s=read(p);state=s.get('training_state') or {}
        for domain,metrics in s['domains'].items():
            for key in ('pass@1','mean_writing_proxy','mean_output_quality','output_risk_fraction',
                        'mean_human_train_similarity','mean_human_held_out_similarity','valid_structure_fraction'):
                value=metrics.get(key)
                if value is None:continue
                row={'step':state.get('step',0),'training_generated_tokens':state.get('tokens',0),
                     'training_seconds':state.get('seconds',0),'domain':domain,'metric':key,'value':value}
                validations.append(row);series[(domain,key)].append((row['training_generated_tokens'],value))
    with (out/'validation_curves.csv').open('w',newline='') as f:
        fields=['step','training_generated_tokens','training_seconds','domain','metric','value']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(validations)
    metrics=[r['metrics'] for r in records]
    finite=all(not isinstance(v,float) or math.isfinite(v) for m in metrics for v in m.values())
    usage=defaultdict(lambda:{'calls':0,'input_tokens':0,'generated_tokens':0,'seconds':0.})
    if config.get('telemetry_root'):
        for p in (Path(config['telemetry_root'])/config.get('id',out.name)).glob('*.json'):
            event=read(p);u=usage[event['role']];u['calls']+=1
            u['input_tokens']+=event['prompt_tokens'];u['generated_tokens']+=event['output_tokens'];u['seconds']+=event['seconds']
    slopes={domain+':'+key:{**trend([(x,-y) for x,y in points] if key=='output_risk_fraction' else points,
                        config.get('plateau_tolerance',.01),config.get('validation_tail_points',4)),
                        'higher_is_better':key!='output_risk_fraction'}
            for (domain,key),points in series.items()}
    risks=[m['output_risk_fraction'] for m in metrics if 'output_risk_fraction' in m]
    trunc=[m['truncated_fraction'] for m in metrics if 'truncated_fraction' in m]
    report={'run':out.name,'complete':(out/'complete.json').exists(),'requested_updates':config.get('steps'),
        'completed_updates':completed.get('step',len(metrics)),'optimizer_updates':completed.get('optimizer_updates'),
        'committed_training_generated_tokens':completed.get('tokens'),
        'committed_training_prompt_tokens':completed.get('prompt_tokens'),
        'training_wall_seconds':completed.get('seconds'),'validation_wall_seconds':completed.get('validation_seconds'),
        'peak_gpu_allocated_bytes':completed.get('peak_gpu_allocated_bytes'),
        'measured_generation_usage':dict(usage),'validation_trends':slopes,
        'health':{'finite_logged_metrics':finite,'mean_truncation_fraction':sum(trunc)/len(trunc) if trunc else None,
                  'mean_output_risk_fraction':sum(risks)/len(risks) if risks else None,
                  'zero_task_gradient_steps':sum(m.get('no_task_gradient',False) for m in metrics)},
        'limitations':['Flat validation tails are not a convergence proof.',
            'Wall time is allocated job time, not active GPU-kernel time.',
            'Generation ledger includes completed retry calls; hard-killed partial calls are unmeasured.',
            'Semantic graph similarity is not evidence of human cognition or logical faithfulness.',
            'Judge screening cannot certify absence of hallucinations.']}
    atomic_json(out/'training_report.json',report)
    return report


def suite_diagnostics(root):
    from .reviews import review_status
    root=Path(root);plan=read(root/'plan.json',{'config':{}});config=plan['config']
    runs=[training_report(p.parent) for p in sorted((root/'runs').glob('*/config.json'))]
    review=review_status(root)
    attempts=[read(p) for p in (root/'attempts').glob('*.json')]
    expected={j['id'] for j in plan.get('jobs',[]) if j['action']=='train_eval'}
    finished={r['run'] for r in runs if r['complete']}
    checks={
        'all_training_complete':bool(runs) and (expected<=finished if expected else all(r['complete'] for r in runs)),
        'all_stages_complete':bool(plan.get('jobs')) and all(read(root/'status'/(j['id']+'.json'),{}).get('state')=='complete' for j in plan.get('jobs',[])),
        'configured_training_seed_count':len(set(config.get('seeds',[])))>=config.get('minimum_training_seeds',3),
        'held_out_humans':bool(config.get('held_out_participants')),
        'live_judge_preflight':read(root/'quality_preflight.json',{}).get('passed',False),
        'human_graph_review_complete':review['graph']['complete'],
        'human_output_review_complete':review['output']['complete'],
        'finite_metrics':bool(runs) and all(r['health']['finite_logged_metrics'] for r in runs),
        'nonzero_optimizer_updates':bool(runs) and all((r.get('optimizer_updates') or 0)>0 for r in runs),
        'truncation_within_predeclared_limit':bool(runs) and all(r['health']['mean_truncation_fraction'] is not None and
            r['health']['mean_truncation_fraction']<=config.get('max_truncation_fraction',.25) for r in runs),
        'allocation_attempt_accounting_complete':all('wall_seconds' in a for a in attempts),
        'no_increasing_validation_tail':bool(runs) and all(r['validation_trends'] and
            all(v['status']=='flat_tail_not_proof_of_convergence' for v in r['validation_trends'].values()) for r in runs),
    }
    for kind in ('graph','output'):
        review_row=review[kind]
        checks['human_'+kind+'_review_adequate']=bool(review_row['complete'] and
            review_row['mean_human_score']>=config.get('review_min_score',3) and
            review_row['mean_absolute_score_disagreement']<=config.get('review_max_disagreement',1) and
            review_row['either_reviewer_risk_fraction']<=config.get('review_max_risk_fraction',.1))
    calibration=review['output'].get('judge_calibration',{})
    checks['judge_calibrated_against_humans']=bool(review['output']['complete'] and
        (calibration.get('risk_recall') or 0)>=config.get('judge_min_risk_recall',.8) and
        calibration.get('score_mae') is not None and calibration['score_mae']<=config.get('judge_max_score_mae',.25))
    measured=defaultdict(lambda:{'calls':0,'input_tokens':0,'generated_tokens':0,'seconds':0.})
    for p in (root/'telemetry').glob('*/*.json'):
        e=read(p);u=measured[e['role']];u['calls']+=1;u['input_tokens']+=e['prompt_tokens'];u['generated_tokens']+=e['output_tokens'];u['seconds']+=e['seconds']
    coverage=[];cue=[];threshold=[]
    for p in (root/'runs').glob('*/test/summary.json'):
        for m in read(p)['domains'].values():
            if m.get('valid_structure_fraction') is not None:coverage.append(m['valid_structure_fraction'])
            if m.get('mean_cue_removal_score_shift') is not None:cue.append(m['mean_cue_removal_score_shift'])
            if m.get('mean_threshold_score_range') is not None:threshold.append(m['mean_threshold_score_range'])
    checks['valid_graph_coverage']=bool(coverage) and min(coverage)>=config.get('min_valid_graph_fraction',.7)
    checks['cue_word_sensitivity_within_limit']=bool(cue) and max(cue)<=config.get('max_cue_score_shift',.1)
    checks['clustering_sensitivity_within_limit']=bool(threshold) and max(threshold)<=config.get('max_threshold_score_range',.2)
    result={'runs':runs,'checks':checks,'review_status':review,'generation_usage_all_stages':dict(measured),
        'seed_uncertainty':'Single-seed pilot: cross-seed variability is unmeasured' if len(set(config.get('seeds',[])))==1 else 'Multiple seeds; inspect paired intervals',
        'allocation_wall_seconds_measured_attempts':sum(a.get('wall_seconds',0) for a in attempts),
        'unfinished_allocation_attempts':sum('wall_seconds' not in a for a in attempts),
        'readiness':'requires_review_or_more_calibration' if not all(checks.values()) else 'diagnostic_checks_complete_not_scientific_certification',
        'open_items':[k for k,v in checks.items() if not v]}
    atomic_json(root/'pilot_readiness.json',result)
    lines=['# Pilot diagnostics','', 'Status: '+result['readiness'],'',
           '## Missing evidence','']+[f'- {x}' for x in result['open_items']]
    lines+=['','## Measured training budgets','',
            '| Run | Updates | Generated tokens | Training seconds | Validation seconds |',
            '|---|---:|---:|---:|---:|']
    for r in runs:lines.append(f"| {r['run']} | {r['completed_updates']} | {r['committed_training_generated_tokens']} | {r['training_wall_seconds']} | {r['validation_wall_seconds']} |")
    lines+=['','See each run\'s `training_report.json` and `validation_curves.csv` for token/time curves, judge usage, health and tail trends.',
            '', 'No automated check certifies human-like thinking, truthfulness, or publication readiness.']
    (root/'pilot_report.md').write_text('\n'.join(lines)+'\n')
    return result
