"""CPU-only regression checks; these do not establish actual GPU compatibility."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.simulated_cot import generate as g


class GenerationChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        folder = root / 'writing'
        folder.mkdir()
        self.prompt = 'Write a fictional proposal.'
        (folder / 'P4.txt').write_text(self.prompt)
        self.manifest = folder / 'manifest.json'
        self.manifest.write_text(json.dumps({'subjects': [dict(
            subject='P4', file='P4.txt', sha256=g.sha(self.prompt), status='verified')]}))
        self.out = root / 'outputs'
        self.addCleanup(patch.stopall)
        patch.object(g, 'DOMAINS', {'writing': folder}).start()
        patch.object(g, 'OUT', self.out).start()
        self.calls = []
        self.loads = []
        owner = self

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                if messages != [{'role': 'user', 'content': owner.prompt}]:
                    raise AssertionError('Participant prompt was changed')
                return [1, 2, 3]

            def decode(self, ids):
                return 'rendered participant prompt'

        class LLM:
            def __init__(self, **kwargs):
                owner.loads.append(kwargs)

            def generate(self, prompts, params):
                owner.calls.extend((prompt, vars(param)) for prompt, param in zip(prompts, params))
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text='<think>reasoning</think>answer', token_ids=[4, 5], finish_reason='stop')])
                    for _ in prompts]

        modules = {
            'torch': SimpleNamespace(__version__='test', cuda=SimpleNamespace(
                is_available=lambda: True, get_device_capability=lambda: (8, 9),
                get_device_name=lambda: 'test L4')),
            'vllm': SimpleNamespace(__version__='test', LLM=LLM,
                                   SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
                                   TokensPrompt=lambda **kwargs: kwargs),
            'transformers': SimpleNamespace(AutoTokenizer=Tokenizer),
        }
        patch.dict('sys.modules', modules).start()

    def run_generator(self, profile='standard', smoke=False):
        with contextlib.redirect_stdout(io.StringIO()):
            g.run('Qwen3-8B', ['writing'], 1, profile, smoke)

    def test_l4_preserves_model_prompt_precision_seeds_and_token_budget(self):
        self.run_generator()
        standard_calls, standard_engine = list(self.calls), self.loads[-1]
        patch.object(g, 'OUT', self.out / 'l4').start()
        self.calls.clear()
        self.run_generator('l4')
        self.assertEqual(self.calls, standard_calls)
        for key in ('model', 'revision', 'tokenizer_revision', 'dtype', 'max_model_len', 'seed'):
            self.assertEqual(self.loads[-1][key], standard_engine[key])
        self.assertEqual(self.loads[-1]['max_num_seqs'], 1)
        self.assertGreater(self.loads[-1]['cpu_offload_gb'], 0)
        records = list((self.out / 'l4' / '_records').glob('*.json'))
        self.assertEqual(len(records), 3)
        self.assertEqual(json.loads(records[0].read_text())['max_new_tokens'], 32768)

    def test_smoke_does_not_count_as_production_and_resume_skips_saved_samples(self):
        self.run_generator('l4', smoke=True)
        self.assertFalse((self.out / '_records').exists())
        self.assertFalse(list(self.out.glob('*_CoT.txt')))
        self.assertEqual(self.calls[0][1]['max_tokens'], 8)
        self.assertEqual(len(list((self.out / '_smoke').glob('*.json'))), 1)
        self.calls.clear()
        self.run_generator('l4')
        self.assertEqual(len(self.calls), 3)
        before = {p.name: p.read_bytes() for p in (self.out / '_records').glob('*.json')}
        self.calls.clear()
        self.loads.clear()
        self.run_generator('l4')
        self.assertEqual(self.calls, [])
        self.assertEqual(self.loads, [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.out / '_records').glob('*.json')})

    def test_changed_input_fails_before_loading_gpu(self):
        (self.manifest.parent / 'P4.txt').write_text('changed prompt')
        with self.assertRaisesRegex(ValueError, 'instruction file changed'):
            self.run_generator('l4')
        self.assertEqual(self.loads, [])
        self.assertFalse(self.out.exists())

    def test_empty_domain_is_rejected_before_loading_gpu(self):
        self.manifest.write_text('{"subjects": []}')
        with self.assertRaisesRegex(ValueError, 'No task instructions'):
            self.run_generator('l4')
        self.assertEqual(self.loads, [])


class CloudLauncherChecks(unittest.TestCase):
    def test_missing_inputs_prevent_cloud_resource_creation(self):
        source = Path(g.__file__).parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'scripts' / 'simulated_cot'
            package.mkdir(parents=True)
            for filename in ('__init__.py', 'generate.py', 'gcp_vm.sh'):
                shutil.copyfile(source / filename, package / filename)
            (root / 'scripts' / '__init__.py').touch()
            fake_bin = root / 'bin'
            fake_bin.mkdir()
            marker = root / 'gcloud-was-called'
            fake_gcloud = fake_bin / 'gcloud'
            fake_gcloud.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 99\n')
            fake_gcloud.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(package / 'gcp_vm.sh'), 'create'],
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('manifest.json', result.stderr)
            self.assertFalse(marker.exists(), 'Cloud API called before inputs were validated')

    def test_transfer_failure_stops_only_the_new_vm(self):
        source = Path(g.__file__).parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'scripts' / 'simulated_cot'
            package.mkdir(parents=True)
            for filename in ('__init__.py', 'generate.py', 'gcp_vm.sh'):
                shutil.copyfile(source / filename, package / filename)
            (root / 'scripts' / '__init__.py').touch()
            for domain in ('writing', 'programming'):
                folder = root / 'data' / f'{domain}_instructions'
                folder.mkdir(parents=True)
                (folder / 'P1.txt').write_text('Task')
                (folder / 'manifest.json').write_text(json.dumps({'subjects': [dict(
                    subject='P1', file='P1.txt', sha256=g.sha('Task'), status='fixture')]}))
            fake_bin = root / 'bin'
            fake_bin.mkdir()
            calls = root / 'calls'
            fake_gcloud = fake_bin / 'gcloud'
            fake_gcloud.write_text(
                f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{calls}"\n'
                'if [ "$2" = scp ]; then exit 7; fi\nexit 0\n')
            fake_gcloud.chmod(0o755)
            fake_git = fake_bin / 'git'
            fake_git.write_text('#!/bin/sh\nif [ "$1" = archive ]; then touch "$4"; fi\nexit 0\n')
            fake_git.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
                       COT_PROJECT='test-project', COT_VM='test-new-vm', COT_ZONE='us-central1-a')
            result = subprocess.run(['bash', str(package / 'gcp_vm.sh'), 'create'],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 7)
            events = calls.read_text().splitlines()
            self.assertEqual(len(events), 3)
            self.assertTrue(events[0].startswith('compute instances create test-new-vm '))
            self.assertTrue(events[1].startswith('compute scp '))
            self.assertTrue(events[2].startswith('compute instances stop test-new-vm '))
            self.assertIn('--project=test-project', events[2])


if __name__ == '__main__':
    unittest.main()
