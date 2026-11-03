"""Explicit semantic-recurrence graph rewards; all structural labels remain proxies.

Nodes cluster similar reasoning steps; edges record successive transitions. This
captures recurrence/branching, not verified logical dependence or private cognition.
No regex move labels are used to construct these graphs.
"""
from collections import defaultdict
import math
from pathlib import Path
import random
import re

import numpy as np

from .common import ROOT, atomic_json, digest, file_hash, read_jsonl, stable_seed

TOPOLOGY = {'human_domain_topology', 'human_pooled_topology', 'shuffled_topology',
            'neutral_topology', 'llm_topology', 'small_world_topology'}
FEATURES = ('unique_fraction', 'edge_density', 'branch_fraction', 'revisit_fraction',
            'reciprocity', 'clustering', 'normalized_path_length', 'transition_entropy')


def steps(text, generated=True):
    if generated:
        m = re.search(r'<think>(.*?)</think>', text, re.S)
        if not m:
            return []
        text = m[1]
    units = [x.strip() for x in re.split(r'\n+|(?<=[.!?])\s+', text) if x.strip()]
    # Bound sentence length rather than silently truncating the encoder's input.
    return [' '.join(words[i:i + 80]) for unit in units
            for words in [unit.split()] for i in range(0, len(words), 80)]


def graph(vectors, threshold=.7):
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Graph requires nonempty finite step embeddings')
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 0):
        raise ValueError('Zero step embedding')
    x = x / norms[:, None]
    n = len(x); parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    similarities = x @ x.T
    for i in range(n):
        for j in range(i):
            if similarities[i, j] >= threshold:
                parent[find(i)] = find(j)
    roots = {}; labels = []
    for i in range(n):
        key = find(i)
        if key not in roots: roots[key] = len(roots)
        labels.append(roots[key])
    k = len(roots)
    edges = {(a, b) for a, b in zip(labels, labels[1:]) if a != b}
    adj = {i: set() for i in range(k)}
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    clustering = []
    for neighbors in adj.values():
        if len(neighbors) >= 2:
            links = sum(b in adj[a] for a in neighbors for b in neighbors if a != b)
            clustering.append(links / (len(neighbors) * (len(neighbors) - 1)))
    distances = []
    for start in range(k):
        seen = {start: 0}; queue = [start]
        for node in queue:
            for neighbor in sorted(adj[node]):
                if neighbor not in seen:
                    seen[neighbor] = seen[node] + 1; queue.append(neighbor)
        distances.extend(d for node, d in seen.items() if node != start)
    path = sum(distances) / len(distances) if distances else 0.
    counts = defaultdict(int)
    for pair in zip(labels, labels[1:]): counts[pair] += 1
    total = sum(counts.values())
    entropy = -sum(c / total * math.log(c / total) for c in counts.values()) if total else 0.
    features = [k / n, len(edges) / max(k * (k - 1), 1),
                sum(sum(a == node for a, _ in edges) > 1 for node in range(k)) / k,
                (n - k) / max(n - 1, 1),
                sum((b, a) in edges for a, b in edges) / max(len(edges), 1),
                sum(clustering) / len(clustering) if clustering else 0.,
                path / max(k - 1, 1), entropy / max(math.log(max(n - 1, 2)), 1e-8)]
    return {'features': dict(zip(FEATURES, features)), 'nodes': k, 'steps': n,
            'edges': sorted([list(e) for e in edges]), 'assignments': labels,
            'mean_shortest_path': path,
            'small_world_score': .5 * features[5] + 1 / (1 + path) if k >= 3 else 0.,
            'graph_definition': 'semantic recurrence nodes and temporal transitions; not logical dependency edges'}


def vector(g):
    return [g['features'][name] for name in FEATURES]


def reference(records, label='human participants'):
    # Give each participant equal weight, irrespective of transcript length.
    participants = defaultdict(list)
    for r in records: participants[r['participant']].append(r['features'])
    means = np.array([np.mean(rows, axis=0) for rows in participants.values()])
    if len(means) < 2: raise ValueError(f'At least two {label} per reference; found {len(means)}')
    center = np.median(means, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(means - center), axis=0), .1)
    return {'center': center.tolist(), 'scale': scale.tolist(),
            'participants': sorted(participants), 'windows': len(records)}


def similarity(g, ref):
    z = (np.asarray(vector(g)) - np.asarray(ref['center'])) / np.asarray(ref['scale'])
    return float(math.exp(-.5 * float(np.mean(z ** 2))))


class GraphEncoder:
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        self.config = config
        self.model = SentenceTransformer(config['structure_embedding_model'],
            revision=config['structure_embedding_revision'], device='cpu',
            local_files_only=config.get('offline', True), trust_remote_code=False)

    def encode(self, units):
        lengths = [len(self.model.tokenizer.encode(s, add_special_tokens=True)) for s in units]
        if any(n > self.model.max_seq_length for n in lengths):
            raise ValueError('Structural encoder input would be truncated; revise segmentation')
        return self.model.encode(units, normalize_embeddings=True, show_progress_bar=False)

    def describe(self, units, threshold=None):
        return graph(self.encode(units), threshold or self.config['structure_threshold'])


def human_sessions():
    from .human_sources import session_texts
    return {domain:{pid:[s for text in rows for s in steps(text,False)] for pid,rows in people.items()}
            for domain,people in session_texts().items()}


def build_references(config, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    encoder = GraphEncoder(config); sessions = human_sessions()
    records = []; shuffled = []; width = config['human_graph_window']
    held = set(config['held_out_participants'])
    diagnostic=set(config['human_diagnostic_participants'])
    for domain, people in sessions.items():
        for pid, units in sorted(people.items()):
            for start in range(0, len(units), width):
                window = units[start:start + width]
                if len(window) < config['min_graph_steps']: continue
                embeddings = encoder.encode(window)
                g = graph(embeddings, config['structure_threshold'])
                base = {'domain': domain, 'participant': pid, 'start': start,
                        'split': 'diagnostic' if pid in diagnostic else 'held_out' if pid in held else 'train', 'text_steps': window}
                records.append({**base, 'features': vector(g), 'graph': g})
                order = list(range(len(window)))
                random.Random(stable_seed(domain, pid, start, 'shuffle')).shuffle(order)
                shuffled.append({**base, 'features': vector(graph(embeddings[order], config['structure_threshold']))})
    refs = {}
    for domain in ('programming', 'writing', 'pooled'):
        match = lambda r: domain == 'pooled' or r['domain'] == domain
        for split in ('train', 'diagnostic','held_out'):
            subset = [r for r in records if match(r) and r['split'] == split]
            # Domain-qualified IDs avoid treating unrelated P4 participants as one person.
            qualified = [{**r, 'participant': r['domain'] + ':' + r['participant']} for r in subset]
            refs[domain + '_' + split] = reference(qualified)
        refs[domain + '_shuffled'] = reference([{**r, 'participant': r['domain'] + ':' + r['participant']}
            for r in shuffled if match(r) and r['split'] == 'train'])
    obj = {'config': {k: config[k] for k in ('structure_embedding_model', 'structure_embedding_revision',
            'structure_threshold', 'human_graph_window', 'held_out_participants','human_diagnostic_participants')},
           'feature_names': FEATURES, 'references': refs, 'human_windows': records,
           'source_hash': digest(sessions), 'validation_status': 'pending_blinded_human_graph_review'}
    atomic_json(out/'references.json', obj)
    return obj


class StructureScorer:
    def __init__(self, config):
        import json
        self.config = config; self.encoder = GraphEncoder(config)
        p = Path(config['structure_reference_path'])
        if config.get('structure_reference_hash') and file_hash(p)!=config['structure_reference_hash']:
            raise ValueError('Frozen human graph reference changed')
        self.references = json.loads(p.read_text())['references']
        self.llm_reference=None
        if config.get('llm_reference_path') and Path(config['llm_reference_path']).exists():
            if config.get('llm_reference_hash') and file_hash(config['llm_reference_path'])!=config['llm_reference_hash']:
                raise ValueError('Frozen LLM reference changed')
            self.llm_reference=json.loads(Path(config['llm_reference_path']).read_text())['references']
        if config.get('method')=='llm_topology' and self.llm_reference is None:
            raise ValueError('Missing frozen LLM reference; run training-only structural calibration first')

    def score(self, text, domain, method):
        units = steps(text)
        if not self.config['min_graph_steps'] <= len(units) <= self.config['max_graph_steps']:
            return {'valid': False, 'steps': len(units), 'reward': 0., 'human_train_similarity':0.,
                    'human_held_out_similarity':0.,'reason': 'missing_short_or_oversized_trace'}
        embeddings = self.encoder.encode(units)
        g = graph(embeddings, self.config['structure_threshold'])
        source = domain if method == 'human_domain_topology' and domain in ('programming', 'writing') else 'pooled'
        ref = self.references[source + ('_shuffled' if method == 'shuffled_topology' else '_train')]
        reward = similarity(g, ref)
        if method == 'neutral_topology':
            reward = similarity(g, {'center': [.5] * len(FEATURES), 'scale': [.25] * len(FEATURES)})
        if method == 'small_world_topology': reward = g['small_world_score']
        if method == 'llm_topology':reward=similarity(g,self.llm_reference['pooled'])
        # Held-out human references never enter the reward.
        evaluation_source=domain if domain in ('programming','writing') else 'pooled'
        held = similarity(g, self.references[evaluation_source + '_'+self.config.get('human_eval_partition','held_out')])
        threshold_scores = {str(t): similarity(graph(embeddings, t), ref)
                            for t in self.config['structure_sensitivity_thresholds']}
        cleaned = [re.sub(r'\b(?:perhaps|maybe|therefore|alternatively|finally)\b', '', s, flags=re.I) for s in units]
        cue_score = similarity(self.encoder.describe(cleaned), ref) if all(s.strip() for s in cleaned) else None
        return {'valid': True, 'steps': len(units), 'reward': reward,
                'human_train_similarity': similarity(g, self.references[source + '_train']),
                'human_held_out_similarity': held, 'graph': g,
                'held_out_evaluation_reference':evaluation_source,
                'threshold_sensitivity': threshold_scores, 'cue_removed_similarity': cue_score,
                'status': 'semantic_graph_proxy_pending_human_validation'}


def calibrate_llm(config, tasks, out):
    """Frozen base-model reference from TRAIN prompts only; no final-test feedback."""
    from .model import load,sample
    from transformers import set_seed
    from .common import write_jsonl
    import json
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if any(t['split']!='train' for t in tasks):
        raise ValueError('LLM structural reference must use training prompts only')
    if len({t['id'] for t in tasks})!=len(tasks):
        raise ValueError('Duplicate calibration task IDs')
    cache=out.with_suffix('.calibration');cache.mkdir(parents=True,exist_ok=True)
    keys=('model','model_revision','structure_embedding_model','structure_embedding_revision',
          'structure_threshold','min_graph_steps','max_graph_steps','max_new_tokens',
          'max_prompt_tokens','group_size','system_prompt','load_in_4bit','device')
    signature=digest({'protocol':'training_prompt_calibration_v2',
                      'config':{k:config.get(k) for k in keys},'tasks':tasks})
    signature_file=cache/'signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text())!=signature:
        raise ValueError('Calibration settings changed; use a new calibration directory')
    atomic_json(signature_file,signature)
    encoder=model=tok=None
    records=[];samples=[]
    for task in tasks:
        for j in range(config['group_size']):
            from . import train
            if train.STOP:raise InterruptedError
            path=cache/'samples'/(digest([task['id'],j])+'.json')
            if path.exists():row=json.loads(path.read_text())
            else:
                if model is None:model,tok=load(config)
                set_seed(stable_seed('base_structure_profile',task['id'],j))
                r=sample(model,tok,task,{**config,'telemetry_role':'base_profile_policy'})
                row={'task_id':task['id'],'domain':task['domain'],'sample':j,'text':r['text'],
                     'token_ids':r['ids'],'prompt_tokens':r['prefix'],'truncated':r['truncated'],
                     'status':'generated'}
                # Save the generation even if graph encoding or a later gate fails.
                atomic_json(path,row)
            if row['status']=='generated':
                units=steps(row['text'])
                reason=('missing_or_incomplete_think_block' if not re.search(r'<think>.*?</think>',row['text'],re.S)
                        else 'trace_too_short' if len(units)<config['min_graph_steps']
                        else 'trace_too_long' if len(units)>config['max_graph_steps'] else None)
                row.update(status='complete',valid=reason is None,reason=reason,steps=len(units),graph=None)
                if row['valid']:
                    if encoder is None:encoder=GraphEncoder(config)
                    row['graph']=encoder.describe(units)
                atomic_json(path,row)
            samples.append(row)
            if row['valid']:
                records.append({'participant':task['id'],'domain':task['domain'],'features':vector(row['graph'])})
    # Coverage and ALL traces must survive a failed reference-data gate.
    write_jsonl(out.with_suffix('.samples.jsonl'),samples)
    coverage={}
    for domain in ('programming','writing'):
        subset=[r for r in samples if r['domain']==domain]
        reasons=defaultdict(int)
        for r in subset:
            if not r['valid']:reasons[r['reason']]+=1
        coverage[domain]={'samples':len(subset),'valid_samples':sum(r['valid'] for r in subset),
                          'valid_prompts':len({r['task_id'] for r in subset if r['valid']}),
                          'truncated_samples':sum(r['truncated'] for r in subset),'rejections':dict(reasons)}
    audit_path=out.with_suffix('.coverage.json')
    insufficient=[d for d,c in coverage.items() if c['valid_prompts']<2]
    atomic_json(audit_path,{'coverage':coverage,'passed':not insufficient,
                          'minimum_valid_prompts_per_domain':2,'signature':signature})
    if insufficient:
        counts=', '.join(f'{d}={coverage[d]["valid_prompts"]}' for d in insufficient)
        raise ValueError('LLM calibration requires valid reasoning graphs from at least two distinct '
                         f'training prompts per domain; found {counts}. Inspect {audit_path} and '
                         f'{out.with_suffix(".samples.jsonl")}. Saved generations will be reused on resume.')
    refs={}
    for domain in ('programming','writing','pooled'):
        subset=[r for r in records if domain=='pooled' or r['domain']==domain]
        refs[domain]=reference(subset,label='distinct training prompts')
    human=json.loads(Path(config['structure_reference_path']).read_text())['references']
    gaps={d:{feature:refs[d]['center'][i]-human[d+'_train']['center'][i] for i,feature in enumerate(FEATURES)}
          for d in refs}
    atomic_json(out,{'references':refs,'human_llm_feature_gaps':gaps,
        'train_task_ids':[t['id'] for t in tasks],'model':config['model'],'model_revision':config['model_revision'],
        'status':'frozen_training_prompt_calibration_not_test_data',
        'limitation':'Semantic recurrence features can depend on topic and trace length; compare length-stratified results and independent human review.'})
