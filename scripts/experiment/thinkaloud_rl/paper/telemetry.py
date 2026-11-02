"""Append-only completed-generation accounting, including retries and discarded groups."""
from pathlib import Path
import time
import uuid
from .common import atomic_json


def generation(config, prompt_tokens, output_tokens, seconds):
    if not config.get('telemetry_root'): return
    atomic_json(Path(config['telemetry_root'])/config.get('id','unnamed')/(uuid.uuid4().hex+'.json'),
        {'kind':'generation','role':config.get('telemetry_role','training_policy'),
         'model':config['model'],'prompt_tokens':prompt_tokens,'output_tokens':output_tokens,
         'seconds':seconds,'finished_at':time.time()})
