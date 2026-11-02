"""Cross-model comparative benchmark evaluation runner.

Evaluates multiple recent open-source models in a single continuous job:
1. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (2025 SOTA Reasoning)
2. Qwen/Qwen2.5-Coder-7B-Instruct (2024-2025 SOTA Coding)
3. Qwen/Qwen2.5-7B-Instruct (Base General Reasoning)
4. meta-llama/Llama-3.1-8B-Instruct (Meta Open Weights Standard)

Minimizes CARC SLURM queue wait times by executing in a single GPU session.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'processed_data').is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from scripts.experiment.thinkaloud_rl.benchmark_loader import (
    load_livecodebench, load_math500, load_aime, load_wildbench_writing,
    evaluate_lcb_sample, evaluate_math500_sample, evaluate_aime_sample, evaluate_wildbench_sample,
    SYSTEM_PROMPT
)
from scripts.experiment.thinkaloud_rl.multidomain_tasks import cognitive_metrics

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of HuggingFace models to benchmark")
    parser.add_argument("--benchmarks", nargs="+", default=["livecodebench", "math500", "aime", "wildbench"])
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pilots/results/legacy_runs/multimodel_benchmark_results")
    parser.add_argument("--limit", type=int, default=50, help="Problems per benchmark for fast turnaround")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def render_prompt(tok, user_prompt):
    try:
        return tok.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"


def evaluate_single_model(model_name, benchmarks, benchmark_dir, output_dir, limit=50, max_tokens=512, load_in_4bit=True):
    print(f"\n{'='*70}\n[+] Evaluating Model: {model_name}\n{'='*70}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Load Tokenizer & Model
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        
    if load_in_4bit and device == "cuda":
        q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=q_config, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32, trust_remote_code=True).to(device)
    model.eval()

    model_slug = model_name.replace("/", "_")
    model_results = {"model": model_name, "benchmarks": {}}

    # 2. Iterate through benchmarks
    for bench in benchmarks:
        print(f"\n--> Running benchmark: {bench.upper()} on {model_name}...")
        if bench == "livecodebench":
            tasks = load_livecodebench(benchmark_dir, limit=limit)
            evaluator = evaluate_lcb_sample
        elif bench == "math500":
            tasks = load_math500(benchmark_dir, limit=limit)
            evaluator = evaluate_math500_sample
        elif bench == "aime":
            tasks = load_aime(benchmark_dir)
            evaluator = evaluate_aime_sample
        elif bench == "wildbench":
            tasks = load_wildbench_writing(benchmark_dir)
            evaluator = lambda text, t: evaluate_wildbench_sample(text)
        else:
            continue
            
        correct = 0
        all_texts = []
        task_outputs = []
        
        for idx, task in enumerate(tasks):
            prompt_str = render_prompt(tok, task["prompt"])
            inputs = tok(prompt_str, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tok.pad_token_id
                )
            prefix_len = inputs["input_ids"].shape[1]
            text = tok.decode(out[0, prefix_len:], skip_special_tokens=True)
            passed = bool(evaluator(text, task))
            if passed:
                correct += 1
            all_texts.append(text)
            task_outputs.append({"task_id": task["id"], "passed": passed, "text": text[:300]})
            
        pass_rate = correct / len(tasks) if tasks else 0.0
        cog = cognitive_metrics(all_texts)
        
        print(f"[✓] {bench.upper()} Pass Rate: {pass_rate*100:.2f}% ({correct}/{len(tasks)})")
        model_results["benchmarks"][bench] = {
            "pass_rate": pass_rate,
            "correct": correct,
            "total": len(tasks),
            "cognitive_metrics": cog,
            "samples": task_outputs[:5]
        }
        
    # Save individual model report
    out_file = output_dir / f"{model_slug}_results.json"
    out_file.write_text(json.dumps(model_results, indent=2))
    print(f"Saved results for {model_name} to: {out_file}")
    
    # Clean up GPU memory before next model
    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return model_results


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    print(f"Starting Multi-Model Evaluation across {len(args.models)} models: {args.models}")
    
    for model_name in args.models:
        try:
            res = evaluate_single_model(
                model_name,
                benchmarks=args.benchmarks,
                benchmark_dir=args.benchmark_dir,
                output_dir=args.output_dir,
                limit=args.limit,
                max_tokens=args.max_tokens,
                load_in_4bit=args.load_in_4bit
            )
            summary[model_name] = {b: res["benchmarks"][b]["pass_rate"] for b in res["benchmarks"]}
        except Exception as e:
            print(f"[!] Error evaluating {model_name}: {e}")
            summary[model_name] = {"error": str(e)}
            
    summary_path = args.output_dir / "multimodel_comparative_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*70}\n[✓] ALL MODELS COMPLETED!\nSummary table written to: {summary_path}\n{'='*70}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

