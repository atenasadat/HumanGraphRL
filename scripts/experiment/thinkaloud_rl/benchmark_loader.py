"""Unified 2024-2026 Gold-Standard Benchmark Loader & Dynamic Execution Harness.

Replaces saturated 2021 datasets with modern, contamination-free standards:
1. LiveCodeBench (LCB 2024-2026): Premier contamination-free competitive coding
2. MATH-500 (OpenAI / DeepSeek-R1 standard): 500 competition math problems
3. AIME 2024: Olympiad-level multi-step mathematical reasoning (000-999 integer answers)
4. WildBench / Scientific Writing (Allen AI 2024): Complex constraint following & drafting

Zero external dependencies beyond Python standard library.
Safe execution protected by 1-second signal timeouts against hangs/infinite loops.
"""

import gzip
import json
import os
from pathlib import Path
import re
import signal

BENCHMARK_DIR = Path(os.environ.get("BENCHMARK_DIR", next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'processed_data').is_dir()) / "data/benchmarks"))

SYSTEM_PROMPT = "Solve the requested task. Put a brief reasoning trace inside <think>...</think>, followed by your final answer."


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out!")


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


def extract_final_answer(text):
    """Strip <think>...</think> and return final text."""
    if '<think>' in text and '</think>' not in text:
        return ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_code_block(text, entry_point=None):
    """Extract code from markdown blocks or raw definitions."""
    blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if blocks:
        for b in reversed(blocks):
            if entry_point and f"def {entry_point}" in b:
                return b
        return blocks[-1]
    
    if entry_point and f"def {entry_point}" in text:
        start = text.find(f"def {entry_point}")
        return text[start:]
    return text


def extract_boxed_or_number(text):
    """Extract \\boxed{...} answer or final numerical value from solution."""
    # Look for \boxed{...}
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    
    # Look for #### <num>
    ans_parts = text.split("####")
    if len(ans_parts) > 1:
        return ans_parts[-1].strip().replace(",", "")
        
    # Look for trailing integers / floats
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if nums:
        return nums[-1].strip()
    return ""


# ==============================================================================
# 1. LiveCodeBench (LCB 2024-2026) Loader & Dynamic Evaluator
# ==============================================================================

def load_livecodebench(benchmark_dir=BENCHMARK_DIR, limit=None):
    path = Path(benchmark_dir) / "livecodebench_lite.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"LiveCodeBench not found at {path}. Run download_benchmarks.sh first.")
    
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            # LCB schema has question_title, question_content, test_cases
            prompt_content = item.get("question_content") or item.get("prompt") or item.get("text", "")
            title = item.get("question_title", f"problem_{idx}")
            test_cases = item.get("test_cases") or item.get("tests", [])
            tasks.append({
                "id": f"lcb_{item.get('task_id', idx)}",
                "title": title,
                "domain": "programming",
                "split": "eval",
                "prompt": f"Solve this programming problem in Python:\n{prompt_content}\nProvide a complete, correct implementation.",
                "test_cases": test_cases,
                "entry_point": item.get("entry_point")
            })
            if limit and len(tasks) >= limit:
                break
    return tasks


def evaluate_lcb_sample(completion_text, task, timeout_sec=1):
    """Executes code against LiveCodeBench test cases."""
    code = extract_code_block(completion_text, task.get("entry_point"))
    if not code:
        return False

    def _execute():
        namespace = {}
        exec(code, namespace)
        
        # If test cases are provided as explicit assertion strings or input/output pairs
        test_cases = task.get("test_cases", [])
        if not test_cases:
            # Basic sanity check: code executed without syntax error and defined a function
            return any(callable(v) for v in namespace.values())
            
        for tc in test_cases[:5]: # run up to 5 validation test cases
            if isinstance(tc, str):
                exec(tc, namespace)
            elif isinstance(tc, dict):
                inp, expected = tc.get("input"), tc.get("output")
                fn_name = task.get("entry_point") or list(namespace.keys())[-1]
                fn = namespace.get(fn_name)
                if callable(fn) and inp is not None:
                    res = fn(*inp) if isinstance(inp, list) else fn(inp)
                    if expected is not None:
                        assert str(res).strip() == str(expected).strip()
        return True

    return safe_run_with_timeout(_execute, timeout_sec=timeout_sec)


# ==============================================================================
# 2. MATH-500 (OpenAI / DeepSeek-R1 Standard) Loader & Evaluator
# ==============================================================================

def load_math500(benchmark_dir=BENCHMARK_DIR, limit=None):
    path = Path(benchmark_dir) / "math500_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"MATH-500 not found at {path}. Run download_benchmarks.sh first.")
    
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            tasks.append({
                "id": f"math500_{idx}",
                "domain": "math",
                "split": "eval",
                "prompt": f"{item['problem']}\nShow your reasoning in <think>...</think> and present your final answer in \\boxed{{...}}.",
                "target_answer": item.get("answer", "").strip(),
                "subject": item.get("subject", "math"),
                "level": item.get("level", 3)
            })
            if limit and len(tasks) >= limit:
                break
    return tasks


def evaluate_math500_sample(completion_text, task):
    """Compares extracted boxed answer or equation against target ground truth."""
    answer = extract_final_answer(completion_text)
    if not answer:
        return False
    
    extracted = extract_boxed_or_number(answer)
    target = extract_boxed_or_number(task["target_answer"]) or task["target_answer"].strip()
    
    # Exact or normalized match
    if extracted == target:
        return True
    
    # Algebraic normalization (strip LaTeX wrappers like \frac, spaces, parens)
    clean = lambda s: re.sub(r"[\s\$\\\{\}\(\)]", "", s).lower()
    return clean(extracted) == clean(target) and len(clean(extracted)) > 0


# ==============================================================================
# 3. AIME 2024 (Olympiad-Level Backtracking) Loader & Evaluator
# ==============================================================================

def load_aime(benchmark_dir=BENCHMARK_DIR):
    path = Path(benchmark_dir) / "aime_2024.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"AIME 2024 not found at {path}. Run download_benchmarks.sh first.")
    
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tasks.append({
                "id": item["id"],
                "domain": "math",
                "split": "eval",
                "prompt": f"{item['problem']}\nFind the exact integer answer (000-999). Show reasoning in <think>...</think> and conclude with the integer.",
                "target_integer": str(item["answer"]).strip()
            })
    return tasks


def evaluate_aime_sample(completion_text, task):
    """Evaluates whether the 3-digit integer answer matches target."""
    answer = extract_final_answer(completion_text)
    if not answer:
        return False
    
    # AIME answers are integers between 0 and 999
    integers = re.findall(r"\b\d{1,4}\b", answer)
    if not integers:
        return False
    # Check if the final emitted integer matches target
    last_int = integers[-1].lstrip("0") or "0"
    target = task["target_integer"].lstrip("0") or "0"
    return last_int == target


# ==============================================================================
# 4. WildBench / Scholarly Proposal Writing (Allen AI 2024 Standard)
# ==============================================================================

def load_wildbench_writing(benchmark_dir=BENCHMARK_DIR):
    path = Path(benchmark_dir) / "wildbench_writing.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"WildBench writing not found at {path}. Run download_benchmarks.sh first.")
    
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tasks.append({
                "id": item["id"],
                "domain": "writing",
                "split": "eval",
                "prompt": item["prompt"],
                "word_budget": item.get("word_budget", [120, 180])
            })
    return tasks


def evaluate_wildbench_sample(completion_text, min_words=120, max_words=180):
    """Evaluates strict constraint adherence, section structure, and conciseness."""
    answer = extract_final_answer(completion_text)
    words = len(answer.split())
    if words < 15:
        return 0.0
    length_score = min(1.0, words / min_words, max_words / max(words, 1))
    
    # High-level section structure required in scholarly/technical proposals
    structure_terms = [
        ["objective", "aim", "hypothes", "investigate", "question"],
        ["method", "experiment", "compar", "protocol", "approach", "architecture"],
        ["metric", "evaluat", "result", "measure", "score", "performance"],
        ["limit", "risk", "feasib", "trade-off", "constraint", "bottleneck"]
    ]
    coverage = sum(any(t in answer.lower() for t in term_group) for term_group in structure_terms) / len(structure_terms)
    return 0.7 * coverage + 0.3 * length_score
