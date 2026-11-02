"""Disjoint train/evaluation prompts and explicitly heuristic outcome rubrics.

Code is inspected statically, never executed. Scores measure rubric compliance,
not functional correctness or scientific writing quality. External validated
outcomes can be supplied for evaluation via --judgments in evaluate.py.
"""
from collections import Counter
import math
import re
from scripts.experiment.thinkaloud_rl.reasoning_shaper import PURE_MOVES, classify_reasoning_step, segment_thought_stream

TASKS = [
 {'id':'code_lcs_train','domain':'programming','split':'train','prompt':'Implement longest_common_subsequence(a, b) in Python, returning the length of the longest common subsequence. Explain the recurrence and complexity.', 'concepts':['def ', 'return', 'for ', 'max(', 'len(']},
 {'id':'code_knapsack_train','domain':'programming','split':'train','prompt':'Implement knapsack(weights, values, capacity) for 0/1 knapsack in Python using O(capacity) memory. Explain why the capacity loop runs backwards.', 'concepts':['def ', 'return', 'range(', 'max(', 'capacity']},
 {'id':'code_toposort_train','domain':'programming','split':'train','prompt':'Implement topological_sort(graph) using Kahn’s algorithm in Python. Detect cycles and explain complexity.', 'concepts':['def ', 'return', 'indegree|in_degree', 'queue|deque', 'cycle']},
 {'id':'write_abstract_train','domain':'writing','split':'train','prompt':'Write a 120–180 word proposal abstract for testing whether uncertainty displays improve calibration in AI-assisted medical image review. Include objective, method, evaluation metric, and limitation. Do not invent results or citations.', 'concepts':['objective|aim|investigate|test', 'random|compar|experiment', 'calibrat', 'limit|risk', 'uncertainty']},
 {'id':'write_grant_train','domain':'writing','split':'train','prompt':'Write a 120–180 word grant introduction for studying low-cost sensors for urban heat exposure. State a research gap, hypothesis, significance, evaluation plan, and feasibility risk. Do not invent citations.', 'concepts':['gap|lack|unknown', 'hypothes|predict', 'heat', 'evaluat|compar|validat', 'risk|limit|feasib']},
 {'id':'write_tradeoff_train','domain':'writing','split':'train','prompt':'Write a 120–180 word technical proposal comparing retrieval augmentation with fine-tuning for an internal research assistant. Include privacy, latency, cost, a proposed experiment, and a justified recommendation.', 'concepts':['privac', 'latency|speed', 'cost', 'experiment|compar|test', 'recommend|propos|choose']},
 {'id':'code_edit_eval','domain':'programming','split':'eval','prompt':'Implement edit_distance(a, b) in Python for unit-cost insertion, deletion and substitution. Explain boundary cases and time complexity.', 'concepts':['def ', 'return', 'min(', 'len(', 'range(']},
 {'id':'code_coin_eval','domain':'programming','split':'eval','prompt':'Implement min_coins(coins, amount) in Python for unlimited coin supply; return -1 if impossible. Explain the recurrence and base case.', 'concepts':['def ', 'return', 'min(', '-1', 'amount']},
 {'id':'code_cycle_eval','domain':'programming','split':'eval','prompt':'Implement has_cycle(graph) in Python using DFS to detect a cycle in a directed graph, including disconnected components. Explain the invariant.', 'concepts':['def ', 'return', 'visit|color|state', 'for ', 'cycle']},
 {'id':'write_abstract_eval','domain':'writing','split':'eval','prompt':'Write a 120–180 word proposal abstract testing whether peer feedback improves introductory statistics learning. Include objective, controlled method, evaluation metric, and limitation. Do not invent results or citations.', 'concepts':['objective|aim|investigate|test', 'random|control|compar', 'learn|score', 'limit|risk', 'feedback']},
 {'id':'write_grant_eval','domain':'writing','split':'eval','prompt':'Write a 120–180 word grant introduction for studying acoustic monitoring of pollinators. State a research gap, hypothesis, significance, evaluation plan, and feasibility risk. Do not invent citations.', 'concepts':['gap|lack|unknown', 'hypothes|predict', 'pollinat', 'evaluat|compar|validat', 'risk|limit|feasib']},
 {'id':'write_tradeoff_eval','domain':'writing','split':'eval','prompt':'Write a 120–180 word technical proposal comparing on-device and cloud speech recognition for field interviews. Include privacy, latency, cost, a proposed experiment, and a justified recommendation.', 'concepts':['privac', 'latency|speed', 'cost', 'experiment|compar|test', 'recommend|propos|choose']},
]
SYSTEM = 'Solve the requested task. Put a brief reasoning trace inside <think>...</think>, followed by your final answer.'


def select_tasks(domain, split):
    return [t for t in TASKS if t['split']==split and (domain=='mixed' or t['domain']==domain)]


def final_answer(text):
    if '<think>' in text and '</think>' not in text:
        return ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def outcome_reward(text, task):
    answer = final_answer(text).lower()
    if len(answer.split()) < 15: return 0.0
    coverage = sum(any(term in answer for term in c.split('|')) for c in task['concepts'])/len(task['concepts'])
    if task['domain']=='writing':
        n = len(answer.split())
        length = min(1., n/120, 180/max(n,1))
        return .8*coverage+.2*length
    return coverage


def cognitive_metrics(texts):
    counts = Counter({m:0 for m in PURE_MOVES}); next_explore = Counter()
    complete = 0
    for text in texts:
        match = re.search(r'<think>(.*?)</think>',text,re.S)
        if not match: continue
        complete += 1
        moves = [classify_reasoning_step(u) for u in segment_thought_stream(match[1])]
        counts.update(moves)
        next_explore.update(b for a,b in zip(moves,moves[1:]) if a=='explore')
    n = sum(counts.values()); e = sum(next_explore.values())
    entropy = lambda c,total: -sum(v/total*math.log2(v/total) for v in c.values() if v) if total else None
    return {'move_counts': dict(counts), 'backtrack_frequency': counts['backtrack']/n if n else None,
            'explore_frequency': counts['explore']/n if n else None,
            'move_entropy_bits': entropy(counts,n),
            'explore_successor_entropy_bits': entropy(next_explore,e),
            'explore_entropy_definition': 'H(next move | current move=explore), pooled within responses',
            'complete_trace_fraction': complete/len(texts) if texts else None,
            'label_method': 'shared_regex_v1_unvalidated'}
