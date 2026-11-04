"""Held-out, cross-domain evaluation; report proxy and validated success separately."""
import argparse
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'data').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from transformers import set_seed
from scripts.experiment.thinkaloud_rl.train import load_model, choose_device, generate
from scripts.experiment.thinkaloud_rl.multidomain_tasks import select_tasks, outcome_reward, cognitive_metrics


def summarize(records, judgments=None):
    judgments=judgments or {}
    result={}
    for domain in ('programming','writing'):
        rows=[r for r in records if r['domain']==domain]
        judged=[judgments[r['sample_id']] for r in rows if r['sample_id'] in judgments]
        if any(type(v) is not bool for v in judged): raise ValueError('Judgments must be JSON booleans')
        result[domain]={**cognitive_metrics([r['text'] for r in rows]),'samples':len(rows),
            'mean_rubric_proxy':sum(r['rubric_proxy'] for r in rows)/len(rows) if rows else None,
            'task_success_rate':sum(judged)/len(judged) if judged else None,
            'validated_samples':len(judged),
            'success_status':'externally_validated' if judged else 'unmeasured: supply independently validated judgments'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--samples-per-task',type=int,default=4)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--judgments',type=Path,help='JSON sample_id -> boolean success; never inferred from rubric proxy')
    p.add_argument('--rescore-only',action='store_true')
    args=p.parse_args()
    if args.samples_per_task<1: raise ValueError('samples-per-task must be positive')
    config=json.loads((args.run_dir/'config.json').read_text())
    records_path=args.run_dir/'evaluation_samples.jsonl'
    if args.rescore_only:
        records=[json.loads(line) for line in records_path.read_text().splitlines()]
    else:
        if records_path.exists(): raise FileExistsError(f'Refusing to overwrite {records_path}')
        set_seed(args.seed); device=choose_device(args.device)
        model,tok=load_model(config['model'],device,args.local_files_only,args.run_dir/'adapter',
                            load_in_4bit=config.get('load_in_4bit',False)); model.eval()
        records=[]
        with records_path.open('w') as handle:
            for task in select_tasks('mixed','eval'):
                for i in range(args.samples_per_task):
                    _,_,ids,text=generate(model,tok,task,device,args.max_new_tokens)
                    row={'sample_id':f"{task['id']}:{i}",'task_id':task['id'],'domain':task['domain'],
                         'prompt':task['prompt'],'text':text,'rubric_proxy':outcome_reward(text,task),'tokens':len(ids)}
                    records.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
    judgments=json.loads(args.judgments.read_text()) if args.judgments else None
    result={'train_domain':config['train_domain'],'condition':config['condition'],'seed':args.seed,
            'samples_per_task':args.samples_per_task,'max_new_tokens':args.max_new_tokens,
            'held_out':True,'domains':summarize(records,judgments),
            'limitations':'Tiny prompt suite; heuristic cognitive labels; no generalization claim without validated task outcomes and multi-seed uncertainty.'}
    (args.run_dir/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
