"""Download once on a login node; freeze data, splits, and upstream revisions."""
import argparse
from collections import Counter
import json
import heapq
from pathlib import Path
import random
import re
import subprocess
import sys

from .common import ROOT, atomic_json, digest, file_hash, read_jsonl, iter_jsonl, write_jsonl

SOURCE_ORIGINS = {
    'google-research-datasets/mbpp': 'public_crowdsourced_programming_benchmark',
    'HuggingFaceH4/MATH-500': 'public_math_benchmark_subset',
    'livecodebench/code_generation_lite': 'public_programming_contest_benchmark',
    'WritingBench': 'public_model_augmented_human_refined_writing_benchmark',
}


def checkout(url, directory, revision=None):
    directory = Path(directory)
    if not directory.exists():
        subprocess.run(['git', 'clone', '--depth', '1', url, str(directory)], check=True)
    if revision:
        subprocess.run(['git', '-C', str(directory), 'fetch', '--depth', '1', 'origin', revision], check=True)
        subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    return subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'], text=True).strip()


def hf_dataset(name, config=None, **kwargs):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(name).sha
    return load_dataset(name, config, revision=revision, **kwargs), revision


def normalized_prompt(prompt):
    return re.sub(r'\W+', ' ', prompt.casefold()).strip()


def validate_tasks(rows):
    ids, prompts = set(), {};count=0
    for r in rows:
        count+=1
        if not all(k in r for k in ('id','domain','split','prompt','scorer','source')):
            raise ValueError('Task missing required fields: ' + str(r.get('id')))
        if r['source'] not in SOURCE_ORIGINS or r.get('data_origin') != SOURCE_ORIGINS[r['source']]:
            raise ValueError('Unverified or locally synthetic task source: ' + str(r.get('id')))
        if not r.get('source_revision') or r.get('source_row_id') is None:
            raise ValueError('Missing upstream provenance: ' + str(r.get('id')))
        if r['split'] not in ('train','val','test'):
            raise ValueError('Unknown data split: ' + str(r.get('id')))
        if r['domain']=='writing' and (r['scorer']!='external_judge' or not r.get('criteria')):
            raise ValueError('Writing requires source criteria and independent judging')
        if r['id'] in ids or not r['prompt'].strip():
            raise ValueError('Duplicate ID or empty prompt: ' + r['id'])
        ids.add(r['id'])
        key = normalized_prompt(r['prompt'])
        if key in prompts and prompts[key] != r['split']:
            raise ValueError('Exact normalized prompt overlap across splits: ' + r['id'])
        prompts[key] = r['split']
        if r['scorer'] == 'assertions' and not r.get('tests'):
            raise ValueError('No executable tests: ' + r['id'])
        if r['scorer'] == 'lcb':
            io = json.loads(r['evaluation_sample']['input_output'])
            if not io['inputs'] or len(io['inputs']) != len(io['outputs']):
                raise ValueError('Missing/misaligned LCB tests: ' + r['id'])
        if r['scorer'] == 'math' and not str(r.get('answer','')).strip():
            raise ValueError('Missing math ground truth: ' + r['id'])
    return {'tasks': count, 'exact_overlap': 0,
            'limitation': 'Exact matching does not rule out semantic overlap or pretraining contamination'}


def writing_tasks(rows, revision):
    """Public WritingBench records; derived split, never an official benchmark score."""
    result=[]
    for row in rows:
        # The human transcripts and frozen graph encoder are English. Retain an
        # explicit English subset rather than treating Chinese graphs as comparable.
        if row.get('lang')!='en':continue
        prompt=row.get('query',row.get('Query'));criteria=row.get('checklist',row.get('Checklist'))
        if not isinstance(prompt,str) or not criteria or 'index' not in row:
            raise ValueError('Unknown WritingBench schema')
        # Identical normalized prompts stay together; preserve original source text.
        bucket=int(digest(['writingbench_split_v1',normalized_prompt(prompt)])[:8],16)%10
        split='train' if bucket<6 else 'val' if bucket==6 else 'test'
        result.append({'id':'writingbench_'+str(row['index']),'domain':'writing','split':split,
            'source':'WritingBench','source_revision':revision,'source_row_id':row['index'],
            'scorer':'external_judge','prompt':prompt,'criteria':criteria,'evidence':prompt,
            'evidence_status':'supplied_benchmark_material_not_independently_verified',
            'domain1':row.get('domain1'),'domain2':row.get('domain2'),'language':row.get('lang'),
            'split_protocol':'writingbench_split_v1_normalized_prompt_60_10_30'})
    return result


def prepare(directory, lcb_release='release_v6', lcb_start='2024-08-01'):
    directory = Path(directory).resolve()
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        return verify(directory)
    directory.mkdir(parents=True, exist_ok=True)
    revisions = {}
    rows = []
    mbpp, revisions['google-research-datasets/mbpp'] = hf_dataset('google-research-datasets/mbpp','full')
    for source_split, target_split in [('train','train'),('validation','val')]:
        for r in mbpp[source_split]:
            # Signature example is public prompt information, remaining assertions stay in verifier.
            rows.append({'id':f'mbpp_{r["task_id"]}', 'domain':'programming','split':target_split,
                'source':'google-research-datasets/mbpp', 'scorer':'assertions',
                'prompt':r['text']+'\nReturn Python code. Use the function name and arguments shown here:\n'+r['test_list'][0],
                'tests':r['test_list'], 'test_setup':r.get('test_setup_code','')})
    math, revisions['HuggingFaceH4/MATH-500'] = hf_dataset('HuggingFaceH4/MATH-500')
    for i,r in enumerate(math['test']):
        rows.append({'id':f'math500_{i}', 'domain':'math','split':'test','source':'HuggingFaceH4/MATH-500',
            'scorer':'math','prompt':r['problem']+'\nPut the final answer in \\boxed{...}.',
            'answer':r['answer'], 'subject':r.get('subject'), 'level':r.get('level')})
    lcb_root = directory/'upstream/LiveCodeBench'
    revisions['LiveCodeBench_git'] = checkout('https://github.com/LiveCodeBench/LiveCodeBench.git', lcb_root)
    sys.path.insert(0,str(lcb_root))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    lcb, revisions['livecodebench/code_generation_lite'] = hf_dataset(
        'livecodebench/code_generation_lite', split='test', version_tag=lcb_release, trust_remote_code=True)
    for r in lcb:
        problem = CodeGenerationProblem(**r)
        if problem.contest_date.date().isoformat() < lcb_start:
            continue
        rows.append({'id':'lcb_'+str(problem.question_id), 'domain':'programming','split':'test',
            'source':'livecodebench/code_generation_lite', 'scorer':'lcb',
            'prompt':problem.question_content+'\nProvide a complete Python implementation.\n'+(problem.starter_code or ''),
            'evaluation_sample':problem.get_evaluation_sample(), 'date':problem.contest_date.isoformat(),
            'difficulty':problem.difficulty.value})
    wb_root = directory/'upstream/WritingBench'
    revisions['WritingBench_git'] = checkout('https://github.com/X-PLUG/WritingBench.git', wb_root)
    rows += writing_tasks(read_jsonl(wb_root/'benchmark_query/benchmark_all.jsonl'),revisions['WritingBench_git'])
    for row in rows:
        row['data_origin']=SOURCE_ORIGINS[row['source']]
        row.setdefault('source_revision',revisions['WritingBench_git' if row['source']=='WritingBench' else row['source']])
        row.setdefault('source_row_id',row['id'])
    checks = validate_tasks(rows)
    write_jsonl(directory/'tasks.jsonl', rows)
    # Small training/validation indexes avoid scanning giant hidden-test payloads.
    for split in ('train','val'):
        write_jsonl(directory/(split+'_tasks.jsonl'),(r for r in rows if r['split']==split))
    counts = Counter((r['source'],r['domain'],r['split']) for r in rows)
    manifest = {'schema_version':2, 'data_policy':'public_sources_no_local_synthetic_tasks',
                'writing_split':'English_subset_derived_60_10_30_not_official_WritingBench_evaluation',
                'revisions':revisions, 'lcb_release':lcb_release,'lcb_start':lcb_start,
                'task_counts':[{'source':s,'domain':d,'split':p,'tasks':n}
                               for (s,d,p),n in sorted(counts.items())],
                'checks':checks,'files':{name:file_hash(directory/name) for name in
                    ('tasks.jsonl','train_tasks.jsonl','val_tasks.jsonl')},
                'upstream_roots':{'lcb':str(lcb_root), 'writingbench':str(wb_root)}}
    atomic_json(manifest_path,manifest)
    return manifest


def verify(directory):
    directory = Path(directory)
    m = json.loads((directory/'manifest.json').read_text())
    if m.get('schema_version')!=2 or m.get('data_policy')!='public_sources_no_local_synthetic_tasks':
        raise ValueError('Old or unverified data manifest; prepare a new public-data directory')
    for name,h in m['files'].items():
        if file_hash(directory/name) != h:
            raise ValueError('Prepared data changed: '+name)
    for repo,key in [('LiveCodeBench','LiveCodeBench_git'),('WritingBench','WritingBench_git')]:
        if key not in m.get('revisions',{}):continue
        path=directory/'upstream'/repo
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=m['revisions'][key]:raise ValueError('Upstream evaluator revision changed: '+repo)
        subprocess.run(['git','-C',str(path),'diff','--exit-code','--quiet','HEAD'],check=True)
    validate_tasks(iter_jsonl(directory/'tasks.jsonl'))
    return m


def task_rows(directory, split=None):
    directory=Path(directory)
    subset=directory/(str(split)+'_tasks.jsonl')
    manifest=directory/'manifest.json'
    tracked=json.loads(manifest.read_text()).get('files',{}) if manifest.exists() else {}
    path=subset if split and subset.name in tracked else directory/'tasks.jsonl'
    return (r for r in iter_jsonl(path) if split is None or r['split']==split)


def select(rows, split, domains, limit=0, seed=2026):
    # Stable hash ordering rather than the easiest/first N benchmark problems.
    if limit<0:raise ValueError('Task limit cannot be negative')
    pools={domain:[] for domain in domains}
    for r in rows:
        if r['split']!=split or r['domain'] not in pools:continue
        pool=pools[r['domain']]
        if not limit:pool.append(r);continue
        item=(-int(digest([seed,r['id']]),16),r['id'],r)
        if len(pool)<limit:heapq.heappush(pool,item)
        elif item[:2]>pool[0][:2]:heapq.heapreplace(pool,item)
    result=[]
    for domain in domains:
        pool=[item[2] for item in pools[domain]] if limit else pools[domain]
        pool.sort(key=lambda r:digest([seed,r['id']]))
        result.extend(pool[:limit] if limit else pool)
    if not result:
        raise ValueError(f'No tasks for {split}/{domains}')
    return result


def select_evaluation(rows,config):
    partition=config.get('evaluation_partition','all')
    if partition not in ('all','pilot_diagnostic','final_holdout'):raise ValueError('Unknown evaluation partition')
    if partition!='all':
        rows=(r for r in rows if r['split']!='test' or
              ((int(digest(['evaluation_partition_v1',r['id']])[:8],16)%5==0)==(partition=='pilot_diagnostic')))
    return select(rows,'test',['programming','writing','math'],config['eval_limit_per_domain'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--lcb-release',default='release_v6')
    p.add_argument('--lcb-start',default='2024-08-01')
    a=p.parse_args()
    print(json.dumps(verify(a.output) if a.verify_only else prepare(a.output,a.lcb_release,a.lcb_start),indent=2))
