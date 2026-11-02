"""Strict outcome scoring. Writing quality is never inferred from length/keywords."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

from .common import final_answer


def extract_code(text):
    text=final_answer(text)
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    return blocks[-1].strip() if blocks else text.strip()


def math_score(text, target):
    from math_verify import parse, verify, LatexExtractionConfig
    answer=final_answer(text)
    if not answer:
        return False
    # Explicitly enclose the WHOLE gold expression in math delimiters.
    gold=str(target).strip().strip('$')
    parsed_gold=parse('$'+gold+'$', extraction_config=[LatexExtractionConfig()])
    if not parsed_gold:
        raise ValueError('Unparseable gold expression: '+str(target))
    parsed_answer=parse(answer)
    return bool(parsed_answer and verify(parsed_gold,parsed_answer))


def writing_proxy(text, task):
    answer=final_answer(text)
    words=len(answer.split())
    if not words:
        return 0.
    low,high=task.get('min_words',120),task.get('max_words',180)
    length=min(1.,words/low,high/words)
    groups=[r'\b(objective|aim|investigat|test)',r'\b(control|random|compar)',
            r'\b(metric|measur|evaluat)',r'\b(limit|risk|feasib)']
    coverage=sum(bool(re.search(p,answer,re.I)) for p in groups)/len(groups)
    return .7*coverage+.3*length


def run_code(text, task, timeout=6, lcb_root=None, runner_prefix=None):
    code=extract_code(text)
    if not code:
        return {'reward':0.,'success':False,'status':'no_final_code'}
    if task['scorer']=='assertions' and not task.get('tests'):
        raise ValueError('Refusing code evaluation without tests')
    if task['scorer']=='lcb':
        io=json.loads(task['evaluation_sample']['input_output'])
        if not io['inputs'] or len(io['inputs'])!=len(io['outputs']):
            raise ValueError('Invalid LCB test cases')
        if not lcb_root:
            raise ValueError('Official LiveCodeBench checkout required')
    worker=Path(__file__).with_name('code_worker.py')
    # Process isolation, resource limits and a scrubbed environment. Not an OS security sandbox.
    with tempfile.TemporaryDirectory(prefix='topgrpo_eval_') as tmp:
        path=Path(tmp)
        (path/'request.json').write_text(json.dumps({'code':code,'task':task,'timeout':timeout,'lcb_root':str(lcb_root or '')}))
        command=[sys.executable,str(worker),str(path/'request.json'),str(path/'result.json')]
        if runner_prefix:
            command=list(runner_prefix)+command
        env={k:v for k,v in os.environ.items() if k in ('PATH','LD_LIBRARY_PATH','SYSTEMROOT')}
        env.update(HOME=tmp,TMPDIR=tmp,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (path/'worker.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=tmp,env=env,stdout=log,stderr=log,start_new_session=True)
            # Official LCB applies a separate timeout to every test case.
            count=len(json.loads(task['evaluation_sample']['input_output'])['inputs']) if task['scorer']=='lcb' else 1
            try:
                proc.wait(timeout=(timeout+1)*count+15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                return {'reward':0.,'success':False,'status':'timeout'}
        if not (path/'result.json').exists():
            raise RuntimeError('Code worker failed: '+(path/'worker.log').read_text()[-2000:])
        result=json.loads((path/'result.json').read_text())
        if result.get('status')=='harness_error':
            raise RuntimeError('Evaluator infrastructure failure: '+result['error'])
        return result


def score(text, task, **kwargs):
    kind=task['scorer']
    if kind in ('assertions','lcb'):
        return run_code(text,task,**kwargs)
    if kind=='math':
        passed=math_score(text,task['answer'])
        return {'reward':float(passed),'success':passed,'status':'verified_math'}
    if kind=='writing_proxy':
        return {'reward':writing_proxy(text,task),'success':None,'status':'heuristic_writing_proxy'}
    if kind=='external_judge':
        return {'reward':None,'success':None,'status':'awaiting_blinded_judgment'}
    raise ValueError('Unknown scorer: '+kind)
