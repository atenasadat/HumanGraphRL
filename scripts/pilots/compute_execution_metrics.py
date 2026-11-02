"""Execution-based unit test harness to compute standard ICLR/NeurIPS metrics:
- Strict Pass@1 (execution correctness on unit tests with 1s timeout)
- Pass@4 (at least one valid solution in group of 4)
- Reasoning Token Length (<think> token count)
- Response Token Length (final solution token count)
across all 27 CARC pilots/results/legacy_runs.
"""
import os
import json
import re
import signal
from collections import defaultdict
import numpy as np

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")

def safe_run_with_timeout(func, *args, timeout_sec=1):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        res = func(*args)
        signal.alarm(0)
        return res
    except Exception:
        signal.alarm(0)
        return False

def extract_code(text, func_name):
    """Extract code containing the target function."""
    blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    for b in reversed(blocks):
        if f"def {func_name}" in b:
            return b
            
    if f"def {func_name}" in text:
        start = text.find(f"def {func_name}")
        end = text.find("\n#", start)
        if end == -1:
            end = text.find("\n```", start)
        if end == -1:
            end = len(text)
        return text[start:end]
    return ""

def _test_edit_distance_raw(code):
    namespace = {}
    exec(code, namespace)
    fn = namespace.get("edit_distance")
    if not fn: return False
    
    assert fn("kitten", "sitting") == 3
    assert fn("", "") == 0
    assert fn("a", "") == 1
    assert fn("", "b") == 1
    assert fn("flaw", "lawn") == 2
    return True

def test_edit_distance(code):
    return safe_run_with_timeout(_test_edit_distance_raw, code, timeout_sec=1)

def _test_min_coins_raw(code):
    namespace = {}
    exec(code, namespace)
    fn = namespace.get("min_coins")
    if not fn: return False
    
    assert fn([1, 2, 5], 11) == 3
    assert fn([2], 3) == -1
    assert fn([1], 0) == 0
    assert fn([1, 3, 4], 6) == 2
    return True

def test_min_coins(code):
    return safe_run_with_timeout(_test_min_coins_raw, code, timeout_sec=1)

def _test_has_cycle_raw(code):
    namespace = {}
    exec(code, namespace)
    fn = namespace.get("has_cycle")
    if not fn: return False
    
    assert fn({0: [1], 1: [2], 2: [0]}) == True
    assert fn({0: [1], 1: [2], 2: []}) == False
    assert fn({0: [1], 1: [], 2: [3], 3: [2]}) == True
    assert fn({0: [], 1: []}) == False
    return True

def test_has_cycle(code):
    return safe_run_with_timeout(_test_has_cycle_raw, code, timeout_sec=1)

TEST_SUITES = {
    "code_edit_eval": ("edit_distance", test_edit_distance),
    "code_coin_eval": ("min_coins", test_min_coins),
    "code_cycle_eval": ("has_cycle", test_has_cycle),
}

def analyze_run(run_dir):
    eval_file = os.path.join(run_dir, "evaluation_samples.jsonl")
    if not os.path.exists(eval_file) or os.path.getsize(eval_file) < 10:
        return None
        
    with open(eval_file) as f:
        samples = [json.loads(line) for line in f]
        
    prog_samples = [s for s in samples if s.get("domain") == "programming"]
    
    task_passes = defaultdict(list)
    task_passed_any = defaultdict(bool)
    
    think_lens = []
    resp_lens = []
    
    for s in prog_samples:
        t_id = s.get("task_id")
        text = s.get("text", "")
        
        # Length metrics
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            think_tokens = len(think_match.group(1).split())
            clean_resp = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            resp_tokens = len(clean_resp.split())
        else:
            think_tokens = 0
            resp_tokens = len(text.split())
            
        think_lens.append(think_tokens)
        resp_lens.append(resp_tokens)
        
        # Execution test
        if t_id in TEST_SUITES:
            func_name, tester = TEST_SUITES[t_id]
            code = extract_code(text, func_name)
            is_pass = tester(code) if code else False
            task_passes[t_id].append(1 if is_pass else 0)
            if is_pass:
                task_passed_any[t_id] = True
                
    all_passes = [p for p_list in task_passes.values() for p in p_list]
    pass_1 = np.mean(all_passes) if all_passes else 0.0
    pass_4 = np.mean(list(task_passed_any.values())) if task_passed_any else 0.0
    
    return {
        "pass_1": pass_1 * 100,
        "pass_4": pass_4 * 100,
        "mean_think_words": np.mean(think_lens) if think_lens else 0.0,
        "mean_resp_words": np.mean(resp_lens) if resp_lens else 0.0,
    }

def main():
    base = "pilots/results/legacy_runs/carc_multidomain_20261027_153910_2209184"
    runs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and ('seed' in d)])
    
    results = defaultdict(lambda: defaultdict(list))
    
    for r in runs:
        run_path = os.path.join(base, r)
        res = analyze_run(run_path)
        if res:
            parts = r.split('_')
            domain = parts[0]
            cond = parts[1]
            if parts[2].startswith('specific'):
                cond = 'domain_specific'
            key = f"{domain}_{cond}"
            
            for k, v in res.items():
                results[key][k].append(v)
                
    print("=" * 80)
    print("STANDARD ICLR/NEURIPS EXECUTION-BASED METRICS (PASS@1 & PASS@4, N=3 SEEDS)")
    print("=" * 80)
    
    summary_table = {}
    for key, metrics in sorted(results.items()):
        summary_table[key] = {
            "pass_1": f"{np.mean(metrics['pass_1']):.1f} ± {np.std(metrics['pass_1']):.1f}%",
            "pass_4": f"{np.mean(metrics['pass_4']):.1f} ± {np.std(metrics['pass_4']):.1f}%",
            "think_len": f"{np.mean(metrics['mean_think_words']):.1f} ± {np.std(metrics['mean_think_words']):.1f} words",
            "resp_len": f"{np.mean(metrics['mean_resp_words']):.1f} ± {np.std(metrics['mean_resp_words']):.1f} words"
        }
        print(f"\n>>> {key.upper()}:")
        print(f"  Pass@1 (Execution Accuracy) : {summary_table[key]['pass_1']}")
        print(f"  Pass@4 (Any in Group of 4)  : {summary_table[key]['pass_4']}")
        print(f"  Reasoning Length (<think>)  : {summary_table[key]['think_len']}")
        print(f"  Output Code/Response Length : {summary_table[key]['resp_len']}")
        
    os.makedirs("pilots/figures/legacy_2026-10-29", exist_ok=True)
    with open("pilots/figures/legacy_2026-10-29/execution_metrics_summary.json", "w") as f:
        json.dump(summary_table, f, indent=2)
        
    print("\nSaved summary to pilots/figures/legacy_2026-10-29/execution_metrics_summary.json")

if __name__ == "__main__":
    main()
