"""Small dependency-free artifact and statistics utilities."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'data').is_dir())


def validate_training_schedule(config):
    steps=config['steps'];budget=config.get('token_budget',0)
    if type(steps) is not int or steps<=0:raise ValueError('steps must be a positive integer batch count')
    if type(budget) is not int or budget<0:raise ValueError('token_budget must be a nonnegative integer; 0 disables it')
    rule=config.get('training_stop_rule')
    if rule not in (None,'fixed_batches'):raise ValueError('Unknown training_stop_rule')
    if rule=='fixed_batches' and budget!=0:
        raise ValueError('fixed_batches requires token_budget=0; measure tokens without stopping on them')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_jsonl(path):
    return list(iter_jsonl(path))


def iter_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def final_answer(text):
    if text.lstrip().startswith('<think>'):
        # Preserve literal tags inside the final artifact (e.g. Python strings).
        _,separator,answer=text.partition('</think>')
        return answer.strip() if separator else ''
    # Some reasoning-model templates already emit the opening tag in the prompt.
    if '<think>' not in text and '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip()
    if '<think>' in text and '</think>' not in text:
        return ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def pass_at_k(n, c, k):
    if not (isinstance(k, int) and 1 <= k <= n and 0 <= c <= n):
        raise ValueError('Require 1 <= k <= n and 0 <= correct <= n')
    if n - c < k:
        return 1.
    return 1. - math.prod((n - c - i) / (n - i) for i in range(k))


def paired_bootstrap(matrix, repetitions=2000, seed=2026, confidence=.95):
    """Crossed resampling of paired training seeds and task IDs, not rollouts."""
    if not matrix or not matrix[0] or len({len(row) for row in matrix}) != 1:
        raise ValueError('A nonempty rectangular seed x task matrix is required')
    if not 0<confidence<1:raise ValueError('Invalid confidence level')
    rng = random.Random(seed)
    ns, nt = len(matrix), len(matrix[0])
    means = []
    for _ in range(repetitions):
        seeds = rng.choices(range(ns), k=ns)
        tasks = rng.choices(range(nt), k=nt)
        means.append(sum(matrix[s][t] for s in seeds for t in tasks) / (ns * nt))
    means.sort()
    alpha=1-confidence
    interval=[means[int(alpha/2*repetitions)],means[min(repetitions-1,int((1-alpha/2)*repetitions))]]
    return {'difference': sum(map(sum, matrix)) / (ns * nt),
            'ci95':interval if confidence==.95 else None,'interval':interval,'confidence_level':confidence,
            'seeds': ns, 'tasks': nt, 'bootstrap_repetitions': repetitions,
            'limitation': 'Single-seed interval excludes training-seed uncertainty' if ns == 1 else None}


def source_fingerprint():
    paths = list((ROOT/'scripts/experiment/thinkaloud_rl/paper').glob('*.py'))
    paths += list((ROOT/'scripts/experiment/thinkaloud_rl/paper/judge_templates').glob('*.txt'))
    paths += [ROOT/'scripts/experiment/thinkaloud_rl/reasoning_shaper.py',
              ROOT/'scripts/experiment/thinkaloud_rl/multidomain_tasks.py',
              ROOT/'scripts/experiment/thinkaloud_rl/prepare_multidomain.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def stable_seed(*parts):
    return int(digest(parts)[:8], 16) % (2**31 - 1)
