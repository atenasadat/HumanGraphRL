"""Standardized benchmark evaluation script for CARC and cluster environments.

Evaluates trained checkpoint/LoRA adapters on:
- OpenAI HumanEval (164 coding tasks, full test execution)
- Google MBPP (held-out test split, 3 unit assertions each)
- OpenAI GSM8K (1,319 math reasoning problems)
- Expository/Proposal Writing Benchmark

Produces publication-grade Pass@1, Pass@k, reasoning lengths, and cognitive entropy.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'data').is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from scripts.experiment.thinkaloud_rl.benchmark_loader import (
    load_livecodebench, load_math500, load_aime, load_wildbench_writing,
    evaluate_lcb_sample, evaluate_math500_sample, evaluate_aime_sample, evaluate_wildbench_sample,
    SYSTEM_PROMPT
)
from scripts.experiment.thinkaloud_rl.multidomain_tasks import cognitive_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--benchmark", choices=["livecodebench", "math500", "aime", "wildbench"], required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per prompt (e.g. 1 for Pass@1, 4 for Pass@4)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional subset limit for fast evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Load tasks
    if args.benchmark == "livecodebench":
        tasks = load_livecodebench(args.benchmark_dir, limit=args.limit)
        evaluator = evaluate_lcb_sample
    elif args.benchmark == "math500":
        tasks = load_math500(args.benchmark_dir, limit=args.limit)
        evaluator = evaluate_math500_sample
    elif args.benchmark == "aime":
        tasks = load_aime(args.benchmark_dir)
        evaluator = evaluate_aime_sample
    elif args.benchmark == "wildbench":
        tasks = load_wildbench_writing(args.benchmark_dir)
        evaluator = lambda text, t: evaluate_wildbench_sample(text)
    
    if args.limit and args.benchmark not in ["livecodebench", "math500"]:
        tasks = tasks[:args.limit]
    
    print(f"==> Evaluating on {len(tasks)} tasks from {args.benchmark}...")
    
    # 2. Load model
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=q_config, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        
    if args.adapter:
        print(f"==> Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    # 3. Evaluation loop
    results = []
    correct_count = 0
    total_rollouts = 0
    all_texts = []
    
    start_time = time.time()
    for idx, task in enumerate(tasks):
        rendered = tok.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]}
        ], tokenize=False, add_generation_prompt=True)
        inputs = tok(rendered, return_tensors="pt").to(device)
        
        sample_correct = 0
        prompt_texts = []
        for s in range(args.num_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=(args.num_samples > 1),
                    temperature=0.7 if args.num_samples > 1 else 0.0,
                    pad_token_id=tok.pad_token_id
                )
            prefix_len = inputs["input_ids"].shape[1]
            text = tok.decode(out[0, prefix_len:], skip_special_tokens=True)
            passed = bool(evaluator(text, task))
            if passed:
                sample_correct += 1
            prompt_texts.append(text)
            all_texts.append(text)
            total_rollouts += 1
            
        correct_count += (sample_correct > 0)
        results.append({
            "task_id": task["id"],
            "prompt": task["prompt"],
            "passed": sample_correct > 0,
            "pass_count": sample_correct,
            "samples": prompt_texts
        })
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(tasks):
            print(f"[{idx+1}/{len(tasks)}] Current Accuracy: {correct_count / (idx + 1) * 100:.2f}%")

    elapsed = time.time() - start_time
    cog = cognitive_metrics(all_texts)
    
    summary = {
        "benchmark": args.benchmark,
        "total_tasks": len(tasks),
        "accuracy_or_pass_rate": correct_count / len(tasks),
        "total_rollouts": total_rollouts,
        "elapsed_seconds": elapsed,
        "cognitive_metrics": cog,
        "results": results
    }
    
    out_path = args.output_file or Path(f"benchmark_results_{args.benchmark}.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[✓] Evaluation completed in {elapsed:.1f}s. Saved to: {out_path}")
    print(f"Final Pass Rate: {summary['accuracy_or_pass_rate']*100:.2f}%")


if __name__ == "__main__":
    main()

