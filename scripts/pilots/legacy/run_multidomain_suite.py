"""Build/run identical-condition, multi-seed train + cross-domain eval commands."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'processed_data').is_dir())
CONDITIONS=('baseline','domain_specific','unified')


def commands(args):
    jobs=[]
    for seed in args.seeds:
        for domain in args.domains:
            for condition in CONDITIONS:
                out=args.output_root/f'{domain}_{condition}_seed{seed}'
                shared=['--local-files-only'] if args.local_files_only else []
                train=[sys.executable,'-m','scripts.experiment.thinkaloud_rl.train','--model',args.model,
                       '--steps',str(args.steps),'--group-size',str(args.group_size),'--seed',str(seed),
                       '--train-domain',domain,'--condition',condition,'--shaping-weight','0' if condition=='baseline' else '1',
                       '--max-new-tokens',str(args.max_new_tokens),'--output-dir',str(out),*shared]
                if args.load_in_4bit: train.append('--load-in-4bit')
                evaluate=[sys.executable,'-m','scripts.experiment.thinkaloud_rl.evaluate','--run-dir',str(out),
                          '--seed',str(seed+10000),'--samples-per-task',str(args.eval_samples),
                          '--max-new-tokens',str(args.max_new_tokens),*shared]
                jobs.append({'seed':seed,'train_domain':domain,'condition':condition,'train':train,'evaluate':evaluate})
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--group-size',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--eval-samples',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--domains',nargs='+',choices=['programming','writing','mixed'],default=['programming','writing','mixed'])
    p.add_argument('--output-root',type=Path,default=ROOT/'pilots/results/legacy_runs/multidomain_suite')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--load-in-4bit',action='store_true')
    p.add_argument('--execute',action='store_true',help='Actually launch training (default prints commands)')
    p.add_argument('--job-index',type=int,help='Select a zero-based SLURM array task')
    args=p.parse_args(); jobs=commands(args)
    if args.job_index is not None:
        if not 0<=args.job_index<len(jobs): p.error('job-index outside generated suite')
        jobs=[jobs[args.job_index]]
    args.output_root.mkdir(parents=True,exist_ok=True)
    name='suite_manifest.json' if args.job_index is None else f'suite_manifest_{args.job_index}.json'
    (args.output_root/name).write_text(json.dumps(jobs,indent=2)+'\n')
    for job in jobs:
        for stage in ('train','evaluate'):
            command=job[stage]; print(shlex.join(command),flush=True)
            if args.execute: subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__': main()
