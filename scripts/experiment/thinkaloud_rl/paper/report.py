"""Paired task/seed comparisons, honest coverage, and regenerable learning curves."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from .common import atomic_json,paired_bootstrap,read_jsonl


def aggregate(root):
    root=Path(root)
    runs=[];scores={};groups=defaultdict(dict);structure_scores={};risk_scores={}
    writing={}
    if (root/'writing_scores.jsonl').exists():
        for r in read_jsonl(root/'writing_scores.jsonl'):
            key=(r['run'],r['task_id'],r['sample'])
            if key in writing:raise ValueError('Duplicate imported writing judgment')
            if r.get('score') is not None:writing[key]=r['score']
    for path in sorted((root/'runs').glob('*/test/summary.json')):
        summary=json.loads(path.read_text());config=summary['config'];name=path.parents[1].name
        domain=config.get('train_domain','untrained');method=config.get('method','base')
        row={'run':name,'model':config['model'],'train_domain':domain,'method':method,'seed':config['seed'],
             'training_tokens':(summary.get('training_state') or {}).get('tokens',0),
             'training_seconds':(summary.get('training_state') or {}).get('seconds',0),
             'domains':summary['domains']}
        runs.append(row);bytask=defaultdict(list);writing_tasks=defaultdict(list)
        structure_tasks=defaultdict(list);risk_tasks=defaultdict(list)
        for p in (path.parent/'samples').glob('*.json'):
            r=json.loads(p.read_text())
            if type(r.get('success')) is bool:bytask[(r['domain'],r['task_id'])].append(float(r['success']))
            if r['domain']=='writing':
                value=r.get('output_assessment',{}).get('quality_score')
                imported=writing.get((name,r['task_id'],r['sample']))
                writing_tasks[r['task_id']].append(value if value is not None else imported/10 if imported is not None else None)
            if 'structure' in r:structure_tasks[(r['domain'],r['task_id'])].append(r['structure'].get('human_held_out_similarity',0.))
            if 'output_assessment' in r:risk_tasks[(r['domain'],r['task_id'])].append(float(r['output_assessment']['risk_detected']))
        for task,values in writing_tasks.items():
            if all(v is not None for v in values):bytask[('writing',task)]=values
        scores[name]={key:sum(values)/len(values) for key,values in bytask.items()}
        structure_scores[name]={key:sum(values)/len(values) for key,values in structure_tasks.items()}
        risk_scores[name]={key:sum(values)/len(values) for key,values in risk_tasks.items()}
        quality=[v for (d,_),v in scores[name].items() if d=='writing']
        if 'writing' in row['domains']:
            row['domains']['writing'].update(mean_writing_score=sum(quality)/len(quality) if quality else None,
                                             fully_judged_tasks=len(quality))
        groups[(config['model'],domain,method)][config['seed']]=name
    comparisons=[]
    for (model,domain,method),methods in sorted(groups.items()):
        if method in ('grpo','base'):continue
        baseline=groups.get((model,domain,'grpo'),{})
        seeds=sorted(set(methods)&set(baseline))
        if not seeds:continue
        for evaluated in ('programming','math','writing'):
            keys=set.intersection(*[set(scores[n]) for s in seeds for n in (methods[s],baseline[s])])
            keys=sorted(k for k in keys if k[0]==evaluated)
            if not keys:continue
            matrix=[[scores[methods[s]][k]-scores[baseline[s]][k] for k in keys] for s in seeds]
            comparisons.append({'model':model,'train_domain':domain,'method':method,'baseline':'grpo',
                'evaluated_domain':evaluated,**paired_bootstrap(matrix),
                'metric':'writing_judge_proxy_0_to_1' if evaluated=='writing' else 'pass@1',
                'inference':'Exploratory CI; not multiplicity-adjusted. Predeclare primary comparisons.'})
        for metric,table in [('held_out_human_graph_similarity',structure_scores),('output_risk_fraction',risk_scores)]:
            for evaluated in ('programming','writing','math'):
                if not all(table.get(n) for s in seeds for n in (methods[s],baseline[s])):continue
                keys=set.intersection(*[set(table[n]) for s in seeds for n in (methods[s],baseline[s])])
                keys=sorted(k for k in keys if k[0]==evaluated)
                if not keys:continue
                matrix=[[table[methods[s]][k]-table[baseline[s]][k] for k in keys] for s in seeds]
                comparisons.append({'model':model,'train_domain':domain,'method':method,'baseline':'grpo',
                    'evaluated_domain':evaluated,'metric':metric,**paired_bootstrap(matrix),
                    '_matrix':matrix,
                    'inference':'Exploratory paired seed/task interval; not a human-cognition or factuality certificate.'})
    primary=[c for c in comparisons if c['method']=='human_domain_topology' and
        c['metric']=='held_out_human_graph_similarity' and c['evaluated_domain'] in
        (('programming','writing') if c['train_domain']=='mixed' else (c['train_domain'],))]
    for c in primary:
        adjusted=paired_bootstrap(c['_matrix'],repetitions=10000,confidence=1-.05/len(primary))
        c.update(primary_endpoint=True,simultaneous_interval=adjusted['interval'],
                 simultaneous_family_size=len(primary),simultaneous_confidence=adjusted['confidence_level'],
                 interval_method='Bonferroni-adjusted crossed seed/task bootstrap; three-seed uncertainty remains unstable')
    for c in comparisons:c.pop('_matrix',None)
    judged=any(r['domains'].get('writing',{}).get('mean_output_quality') is not None for r in runs)
    output={'runs':runs,'paired_comparisons':comparisons,'writing_quality':'Independent automated judge proxies; inspect human calibration, risk flags and coverage' if judged else 'Imported judgments remain quality proxies; inspect coverage and judge validation' if writing else 'Unmeasured until judge results are imported',
            'limitations':['Heuristic cognitive labels remain unvalidated until independent audit.',
                'Older writing runs use a constraint proxy; robust pilot uses a frozen evidence-aware judge proxy.',
                'Small pilots and single-seed results are diagnostic.',
                'Compare token/GPU-hour learning curves; equal update counts do not imply equal compute.',
                'No automatic superiority or significance claim is made.']}
    atomic_json(root/'report.json',output)
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text())['config'].get('robust_pilot'):
        from .diagnostics import suite_diagnostics
        output['pilot_diagnostics']=suite_diagnostics(root)
        atomic_json(root/'report.json',output)
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run','model','train_domain','method','seed','training_tokens','training_seconds',
                                          'eval_domain','pass@1','pass@4','truncated_fraction','unscored_tasks','mean_writing_score','fully_judged_tasks',
                                          'mean_output_quality','output_risk_fraction','mean_human_held_out_similarity','valid_structure_fraction',
                                          'mean_cue_removal_score_shift','mean_threshold_score_range'])
        writer.writeheader()
        for r in runs:
            for d,metrics in r['domains'].items():
                writer.writerow({**{k:v for k,v in r.items() if k!='domains'},'eval_domain':d,
                    **{k:metrics.get(k) for k in ('pass@1','pass@4','truncated_fraction','unscored_tasks','mean_writing_score','fully_judged_tasks',
                        'mean_output_quality','output_risk_fraction','mean_human_held_out_similarity','valid_structure_fraction',
                        'mean_cue_removal_score_shift','mean_threshold_score_range')}})
    return output


def curves(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    fig,axes=plt.subplots(3,2,figsize=(11,12),squeeze=False)
    for run in sorted((root/'runs').glob('*')):
        vals=defaultdict(list)
        for p in sorted((run/'validation').glob('*/summary.json')):
            d=json.loads(p.read_text());state=d.get('training_state') or {}
            for domain,metric in [('programming','pass@1'),('writing','mean_output_quality')]:
                value=d['domains'].get(domain,{}).get(metric)
                if value is None and domain=='writing':value=d['domains'].get(domain,{}).get('mean_writing_proxy')
                if value is not None:
                    vals[domain].append((state['tokens'],state['seconds']/3600,value))
            values=[x['mean_human_held_out_similarity'] for x in d['domains'].values() if x.get('mean_human_held_out_similarity') is not None]
            if values:vals['structure'].append((state['tokens'],state['seconds']/3600,sum(values)/len(values)))
        for row,domain in enumerate(('programming','writing','structure')):
            points=vals[domain]
            if points:
                for column in (0,1):
                    axes[row,column].plot([x[column] for x in points],[x[2] for x in points],label=run.name)
    for row,ylabel in enumerate(('Validation coding Pass@1','Validation writing score (judge or legacy proxy)','Held-out human graph similarity (proxy)')):
        for column,xlabel in enumerate(('Generated training tokens','Training wall-clock hours')):
            ax=axes[row,column]
            ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=5)
    fig.tight_layout();fig.savefig(root/'learning_curves.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plots',action='store_true');a=p.parse_args();aggregate(a.root)
    if a.plots:curves(a.root)
