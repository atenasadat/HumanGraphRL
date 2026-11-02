"""One declared reward contract shared by training, validation and final evaluation."""
import math
import time
from pathlib import Path

from .common import digest,file_hash
from .output_quality import OutputJudge
from .scoring import score
from .structure import StructureScorer, TOPOLOGY


class RewardRuntime:
    def __init__(self, config, phase='training'):
        self.config = config; self.phase = phase
        self.structure = StructureScorer(config)
        self.judge = OutputJudge(config, phase)
        self.seconds = 0.; self.scored = 0

    def assess(self, text, task):
        from .train import rng_state,restore_rng
        before=rng_state()
        try:return self._assess(text,task)
        finally:restore_rng(before)

    def _assess(self, text, task):
        start = time.monotonic(); method = self.config.get('method', 'grpo')
        structure = self.structure.score(text, task['domain'], method)
        if task['domain'] == 'writing':
            output = self.judge.assess(text, task)
            base = {'reward': output['output_reward'], 'success': None, 'status': output['status']}
        else:
            base = score(text, task, lcb_root=self.config.get('lcb_root'), timeout=self.config.get('code_timeout', 6))
            output = {'output_reward': base['reward'], 'factuality_status': 'executable_or_symbolic_verifier',
                      'risk_detected': not bool(base['success']), 'quality_score': None}
        outcome = float(base['reward'])
        if not math.isfinite(outcome): raise ValueError('Nonfinite final-output reward')
        # Human differences explicitly drive a rollout-level reward, not a zero-mean prefix baseline.
        structural = structure['reward'] if method in TOPOLOGY else 0.
        gate = not output.get('risk_detected', True) and structure['valid']
        bonus = self.config['structure_reward_weight'] * structural if gate else 0.
        total = self.config['outcome_reward_weight'] * outcome + bonus
        if not math.isfinite(total): raise ValueError('Nonfinite joint reward')
        self.seconds += time.monotonic() - start; self.scored += 1
        return {**base, 'reward': total, 'outcome_reward': outcome, 'structure_bonus': bonus,
                'structure': structure, 'output_assessment': output,
                'reward_contract': 'verified_or_judged_output_plus_quality_gated_direct_structure_v1'}

    def usage(self):
        return {'scored_rollouts': self.scored, 'reward_seconds': self.seconds, 'judge': self.judge.usage()}


def attach(config, plan, prepared):
    root = Path(plan['root'])
    config.update(structure_reference_path=str(root/'structure/references.json'),
                  structure_embedding_revision=prepared['models'][config['structure_embedding_model']],
                  reward_judge_revision=prepared['models'][config['reward_judge_model']],
                  evaluation_judge_revision=prepared['models'][config['evaluation_judge_model']],
                  quality_cache_root=str(root/'quality_cache'),
                  telemetry_root=str(root/'telemetry'),
                  llm_reference_path=str(root/'structure'/('llm_'+config.get('model','').replace('/','_')+'.json')),
                  lcb_root=str(Path(plan['data_root'])/'upstream/LiveCodeBench'))
    config['structure_reference_hash'] = prepared['structure_reference_hash']
    if Path(config['llm_reference_path']).exists():config['llm_reference_hash']=file_hash(config['llm_reference_path'])
    return config
