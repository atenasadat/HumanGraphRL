"""Generate reasoning-model chain-of-thought for every think-aloud participant's task.

Each model receives the participant's exact task text as a single user message, with no
system prompt or added instruction, through the model's own chat template. Decoding uses
each model's officially recommended sampling settings. Every (model, subject) pair gets
SAMPLES independent samples with fixed seeds. Results go to data/Simulated CoT data/:

  {domain}_{subject}_{model}_s{k}_CoT.txt      reasoning trace
  {domain}_{subject}_{model}_s{k}_output.txt   final answer
  _records/{same stem}.json                    seed, revision, tokens, finish reason, status

Truncated or malformed generations are kept and flagged in their record, never dropped
or regenerated with a different seed. Re-running resumes; finished samples are skipped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
OUT = DATA / 'Simulated CoT data'
SAMPLES = 3
MAX_NEW_TOKENS = 32768
# Qwen3-8B's native context; one budget for all models keeps the comparison fair.
MAX_MODEL_LEN = 40960
DOMAINS = {
    'writing': DATA / 'writing_instructions',
    'programming': DATA / 'programming_instructions',
}
MODELS = {
    # Sampling values are each model card's recommendation for reasoning use.
    'R1-Distill-Qwen-7B': dict(repo='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                               revision='916b56a44061fd5cd7d6a8fb632557ed4f724f60',
                               sampling=dict(temperature=0.6, top_p=0.95), template={}, format='think', dtype='bfloat16'),
    'Qwen3-8B': dict(repo='Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218',
                     sampling=dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
                     template={'enable_thinking': True}, format='think', dtype='bfloat16'),
    # Native MXFP4 weights; 'medium' is the model's default reasoning effort.
    'gpt-oss-20b': dict(repo='openai/gpt-oss-20b', revision='6cee5e81ee83917806bbde320786a8fb61efebee',
                        sampling=dict(temperature=1.0, top_p=1.0), template={'reasoning_effort': 'medium'},
                        format='harmony', dtype='auto'),
}
HARMONY = re.compile(r'<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|$)', re.S)
END_TOKENS = ('<|im_end|>', '<｜end▁of▁sentence｜>', '<|endoftext|>', '<|return|>')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def tasks(domains):
    rows = []
    for domain in domains:
        folder = DOMAINS[domain]
        manifest = json.loads((folder / 'manifest.json').read_text())
        for s in manifest['subjects']:
            text = (folder / s['file']).read_text()
            if sha(text) != s['sha256']:
                raise ValueError(f"{domain} {s['subject']}: instruction file changed since its manifest was built")
            rows.append({'domain': domain, 'subject': s['subject'], 'text': text, 'instruction_sha256': s['sha256'],
                         'instruction_status': s['status']})
    return rows


def split(text, fmt, finish):
    """Return (reasoning, answer, status) without altering either part."""
    for token in END_TOKENS:
        if text.endswith(token):
            text = text[:-len(token)]
    if fmt == 'harmony':
        messages = HARMONY.findall(text)
        reasoning = '\n\n'.join(m.strip() for c, m in messages if c == 'analysis')
        finals = [m.strip() for c, m in messages if c == 'final']
        answer = finals[-1] if finals else ''
        if not messages:
            return text.strip(), '', 'truncated' if finish == 'length' else 'unparsed_format'
        if not finals:
            return reasoning, '', 'truncated_in_reasoning' if finish == 'length' else 'no_final_channel'
    else:
        head, sep, tail = text.partition('</think>')
        reasoning = head.removeprefix('<think>').strip()
        answer = tail.strip()
        if not sep:
            return reasoning, '', 'truncated_in_reasoning' if finish == 'length' else 'no_reasoning_end_marker'
    if finish == 'length':
        return reasoning, answer, 'truncated_in_answer'
    return reasoning, answer, 'complete' if reasoning and answer else ('empty_answer' if reasoning else 'empty_reasoning')


def engine_options(name, profile='standard'):
    """Change execution memory usage, never the model, precision or token budget."""
    options = dict(dtype=MODELS[name]['dtype'], max_model_len=MAX_MODEL_LEN,
                   gpu_memory_utilization=0.9, seed=0)
    if profile == 'l4':
        options.update(enforce_eager=True, max_num_seqs=1,
                       max_num_batched_tokens=512, enable_chunked_prefill=True,
                       cpu_offload_gb=4.0 if name == 'Qwen3-8B' else 0.0)
    elif profile != 'standard':
        raise ValueError(f'Unknown hardware profile: {profile}')
    return options


def check_inputs(domains):
    rows = tasks(domains)
    if not rows:
        raise ValueError('No task instructions found')
    for domain in domains:
        count = sum(t['domain'] == domain for t in rows)
        if not count:
            raise ValueError(f'{domain}: no task instructions found')
        print(f'{domain}: {count} verified task instructions')
    return rows


def run(name, domains, batch, profile='standard', smoke_test=False):
    # Validate files before importing CUDA libraries or allocating GPU memory.
    rows = check_inputs(domains)
    if batch < 1:
        raise ValueError('Batch must be positive')
    if smoke_test:
        rows = rows[:1]
    record_dir = OUT / ('_smoke' if smoke_test else '_records')
    pending = []
    for t in rows:
        for k in ([1] if smoke_test else range(1, SAMPLES + 1)):
            stem = f"{t['domain']}_{t['subject']}_{name}_s{k}"
            if smoke_test or not (record_dir / f'{stem}.json').exists():
                pending.append((t, k, stem))
    print(f'{name}: {len(pending)} samples pending', flush=True)
    if not pending:
        return
    # The capability check initializes CUDA; forked workers cannot reuse that state.
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    spec = MODELS[name]
    if not torch.cuda.is_available():
        raise RuntimeError('An NVIDIA GPU is required; run on the GPU VM, not Cloud Shell')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('These BF16/MXFP4 models require an Ampere or newer NVIDIA GPU')
    options = engine_options(name, profile)
    print(f'Engine settings: {json.dumps(options)}', flush=True)
    tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])
    llm = LLM(model=spec['repo'], revision=spec['revision'], tokenizer_revision=spec['revision'],
              **options)
    env = {'vllm': vllm.__version__, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
           'hardware_profile': profile, 'engine_options': options}
    record_dir.mkdir(parents=True, exist_ok=True)
    token_budget = 8 if smoke_test else MAX_NEW_TOKENS
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        prompts, params = [], []
        for t, k, stem in chunk:
            # No system prompt: the participant's task is the entire user turn.
            ids = tok.apply_chat_template([{'role': 'user', 'content': t['text']}], tokenize=True,
                                          add_generation_prompt=True, **spec['template'])
            ids = list(ids['input_ids'] if isinstance(ids, dict) else ids)
            if len(ids) + token_budget > MAX_MODEL_LEN:
                raise ValueError(f'{stem}: prompt plus generation budget exceeds {MAX_MODEL_LEN} tokens')
            seed = int(sha(f'{name}|{t["domain"]}|{t["subject"]}|{k}')[:8], 16)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(max_tokens=token_budget, seed=seed, skip_special_tokens=False,
                                         **spec['sampling']))
        began = time.time()
        outputs = llm.generate(prompts, params)
        seconds = time.time() - began
        for (t, k, stem), prompt, param, out in zip(chunk, prompts, params, outputs):
            gen = out.outputs[0]
            if not gen.token_ids:
                raise RuntimeError(f'{stem}: GPU returned no generated tokens')
            reasoning, answer, status = split(gen.text, spec['format'], gen.finish_reason)
            if not smoke_test:
                atomic(OUT / f'{stem}_CoT.txt', reasoning)
                atomic(OUT / f'{stem}_output.txt', answer)
            record = {'domain': t['domain'], 'subject': t['subject'], 'model': name, 'sample': k,
                      'repo': spec['repo'], 'revision': spec['revision'], 'seed': param.seed,
                      'sampling': spec['sampling'], 'chat_template_kwargs': spec['template'],
                      'run_kind': 'smoke' if smoke_test else 'generation',
                      'max_new_tokens': token_budget, 'instruction_sha256': t['instruction_sha256'],
                      'instruction_status': t['instruction_status'],
                      'rendered_prompt': tok.decode(prompt['prompt_token_ids']),
                      'prompt_tokens': len(prompt['prompt_token_ids']), 'completion_tokens': len(gen.token_ids),
                      'finish_reason': gen.finish_reason, 'status': status, 'raw_completion': gen.text,
                      'cot_sha256': sha(reasoning), 'output_sha256': sha(answer),
                      'batch_seconds': seconds, 'batch_size': len(chunk), 'environment': env}
            atomic(record_dir / f'{stem}.json', json.dumps(record, ensure_ascii=False, indent=1))
        done = start + len(chunk)
        print(f'{name}: {done}/{len(pending)} saved ({seconds:.0f}s for {len(chunk)})', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--model', choices=MODELS)
    action.add_argument('--check-inputs', action='store_true', help='Validate input files without GPU dependencies')
    p.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    p.add_argument('--profile', choices=['standard', 'l4'], default='standard')
    p.add_argument('--batch', type=int, help='Requests per save; default: 64 standard, 1 L4')
    p.add_argument('--smoke-test', action='store_true', help='Generate 8 test tokens into _smoke only')
    args = p.parse_args()
    if args.check_inputs:
        if args.smoke_test:
            p.error('--smoke-test requires --model')
        check_inputs(args.domains)
    else:
        batch = args.batch if args.batch is not None else (1 if args.profile == 'l4' else 64)
        run(args.model, args.domains, batch, args.profile, args.smoke_test)
