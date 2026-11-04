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


def run(name, domains, batch):
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    spec = MODELS[name]
    (OUT / '_records').mkdir(parents=True, exist_ok=True)
    pending = []
    for t in tasks(domains):
        for k in range(1, SAMPLES + 1):
            stem = f"{t['domain']}_{t['subject']}_{name}_s{k}"
            if not (OUT / '_records' / f'{stem}.json').exists():
                pending.append((t, k, stem))
    print(f'{name}: {len(pending)} samples pending', flush=True)
    if not pending:
        return
    tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])
    llm = LLM(model=spec['repo'], revision=spec['revision'], tokenizer_revision=spec['revision'],
              dtype=spec['dtype'], max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9, seed=0)
    env = {'vllm': vllm.__version__, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        prompts, params = [], []
        for t, k, stem in chunk:
            # No system prompt: the participant's task is the entire user turn.
            ids = tok.apply_chat_template([{'role': 'user', 'content': t['text']}], tokenize=True,
                                          add_generation_prompt=True, **spec['template'])
            ids = list(ids['input_ids'] if isinstance(ids, dict) else ids)
            seed = int(sha(f'{name}|{t["domain"]}|{t["subject"]}|{k}')[:8], 16)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(max_tokens=MAX_NEW_TOKENS, seed=seed, skip_special_tokens=False,
                                         **spec['sampling']))
        began = time.time()
        outputs = llm.generate(prompts, params)
        seconds = time.time() - began
        for (t, k, stem), prompt, param, out in zip(chunk, prompts, params, outputs):
            gen = out.outputs[0]
            reasoning, answer, status = split(gen.text, spec['format'], gen.finish_reason)
            atomic(OUT / f'{stem}_CoT.txt', reasoning)
            atomic(OUT / f'{stem}_output.txt', answer)
            record = {'domain': t['domain'], 'subject': t['subject'], 'model': name, 'sample': k,
                      'repo': spec['repo'], 'revision': spec['revision'], 'seed': param.seed,
                      'sampling': spec['sampling'], 'chat_template_kwargs': spec['template'],
                      'max_new_tokens': MAX_NEW_TOKENS, 'instruction_sha256': t['instruction_sha256'],
                      'instruction_status': t['instruction_status'],
                      'rendered_prompt': tok.decode(prompt['prompt_token_ids']),
                      'prompt_tokens': len(prompt['prompt_token_ids']), 'completion_tokens': len(gen.token_ids),
                      'finish_reason': gen.finish_reason, 'status': status, 'raw_completion': gen.text,
                      'cot_sha256': sha(reasoning), 'output_sha256': sha(answer),
                      'batch_seconds': seconds, 'batch_size': len(chunk), 'environment': env}
            atomic(OUT / '_records' / f'{stem}.json', json.dumps(record, ensure_ascii=False, indent=1))
        done = start + len(chunk)
        print(f'{name}: {done}/{len(pending)} saved ({seconds:.0f}s for {len(chunk)})', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    p.add_argument('--batch', type=int, default=64, help='Requests per vLLM call; results are saved after each')
    args = p.parse_args()
    run(args.model, args.domains, args.batch)
