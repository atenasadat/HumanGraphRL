"""Quota selection and one-VM safeguards; no real cloud resources are created."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.simulated_cot import find_vm as f


class FindVMChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logs = Path(self.tmp.name)

    def run_search(self, zones):
        with contextlib.redirect_stdout(io.StringIO()):
            return f.search('test-project', 'test-vm', zones, self.logs)

    def test_quota_overrides_defaults_and_skips_unapproved_or_other_gpu_types(self):
        quota = {'dimensionsInfos': [
            {'dimensions': {}, 'details': {}},
            {'dimensions': {'gpu_family': 'NVIDIA_L4'},
             'applicableLocations': ['us-east1', 'us-west1'], 'details': {'quotaValue': '1'}},
            {'dimensions': {'gpu_family': 'NVIDIA_L4', 'region': 'us-west1'}, 'details': {}},
            {'dimensions': {'gpu_family': 'NVIDIA_H100', 'region': 'us-central1'},
             'details': {'quotaValue': '8'}},
            {'dimensions': {'gpu_family': 'NVIDIA_L4', 'region': 'us-central1'},
             'details': {'quotaValue': '1'}},
        ]}
        self.assertEqual(f.l4_quota(quota, 'us-east1'), 1)
        self.assertEqual(f.l4_quota(quota, 'us-west1'), 0)
        self.assertEqual(f.l4_quota(quota, 'europe-west1'), 0)
        catalog = [dict(name='nvidia-l4',
                        zone=f'https://example/zones/{z}')
                   for z in ['us-east1-c', 'us-central1-c', 'us-west1-a', 'europe-west1-b']]
        catalog.append(dict(name='nvidia-tesla-t4', zone='us-east1-b'))
        self.assertEqual(f.candidate_zones(catalog, quota), ['us-central1-c', 'us-east1-c'])
        self.assertEqual(f.candidate_zones(catalog, quota, ['us-east1']), ['us-east1-c'])

    def test_live_gcloud_value_field_selects_approved_l4_zone(self):
        # Shape returned by the user's actual Cloud Shell request.
        quota = {'dimensionsInfos': [{
            'applicableLocations': ['us-central1'],
            'details': {'value': '1'},
            'dimensions': {'gpu_family': 'NVIDIA_L4', 'region': 'us-central1'},
        }]}
        catalog = [{'name': 'nvidia-l4', 'maximumCardsPerInstance': 8,
                    'zone': 'https://www.googleapis.com/compute/v1/projects/reasonedllm/zones/us-central1-a'}]
        self.assertEqual(f.l4_quota(quota, 'us-central1'), 1)
        self.assertEqual(f.l4_quota(quota, 'us-east1'), 0)
        self.assertEqual(f.candidate_zones(catalog, quota), ['us-central1-a'])

    def test_explicit_zero_value_overrides_old_quota_value(self):
        quota = {'dimensionsInfos': [{
            'details': {'value': '0', 'quotaValue': '1'},
            'dimensions': {'gpu_family': 'NVIDIA_L4', 'region': 'us-central1'},
        }]}
        self.assertEqual(f.l4_quota(quota, 'us-central1'), 0)

    def test_metadata_lookup_reports_progress_and_parses_json(self):
        executable = self.logs / 'gcloud'
        executable.write_text('#!/bin/sh\nprintf \'[{"name":"nvidia-l4","zone":"us-east1-b"}]\'\n')
        executable.chmod(0o755)
        output = io.StringIO()
        with patch.dict(os.environ, PATH=str(self.logs) + os.pathsep + os.environ['PATH']), \
                contextlib.redirect_stdout(output):
            result = f.cloud_json('test-project', 'compute', 'accelerator-types', 'list')
        self.assertEqual(result[0]['name'], 'nvidia-l4')
        self.assertIn('Checking compute accelerator-types list', output.getvalue())

    def test_metadata_lookup_timeout_kills_stalled_process(self):
        executable = self.logs / 'gcloud'
        executable.write_text('#!/bin/sh\nexec sleep 30\n')
        executable.chmod(0o755)
        original_popen = subprocess.Popen
        children = []
        def start_process(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            children.append(process)
            return process
        with patch.dict(os.environ, PATH=str(self.logs) + os.pathsep + os.environ['PATH']), \
                patch.object(f.subprocess, 'Popen', side_effect=start_process), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                f.cloud_json('test-project', 'compute', 'accelerator-types', 'list', timeout=0.1)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll(), 'Timed-out gcloud process was left running')

    def test_metadata_lookup_heartbeat_before_slow_response(self):
        from unittest.mock import MagicMock
        process = MagicMock()
        process.communicate.side_effect = [subprocess.TimeoutExpired('gcloud', 10), ('[]', '')]
        process.returncode = 0
        process.poll.return_value = 0
        process.__enter__.return_value = process
        output = io.StringIO()
        with patch.object(f.subprocess, 'Popen', return_value=process), \
                contextlib.redirect_stdout(output):
            self.assertEqual(f.cloud_json('test-project', 'compute', 'instances', 'list'), [])
        self.assertIn('Still checking compute instances list', output.getvalue())

    def test_capacity_failure_then_success_stops_before_third_zone(self):
        with patch.object(f, 'require_no_vm'), patch.object(f, 'attempt', side_effect=[
                (1, 'ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS'), (0, 'created and staged')]) as attempt:
            zone = self.run_search(['us-east1-b', 'us-east1-c', 'us-west1-a'])
        self.assertEqual(zone, 'us-east1-c')
        self.assertEqual(attempt.call_count, 2)

    def test_failed_command_that_left_a_vm_never_tries_another_zone(self):
        with patch.object(f, 'require_no_vm', side_effect=[None, RuntimeError('VM exists')]), \
                patch.object(f, 'attempt', return_value=(1, 'ZONE_RESOURCE_POOL_EXHAUSTED')) as attempt:
            with self.assertRaisesRegex(RuntimeError, 'VM exists'):
                self.run_search(['us-east1-b', 'us-east1-c'])
        self.assertEqual(attempt.call_count, 1)

    def test_regional_quota_error_skips_remaining_zones_in_that_region(self):
        with patch.object(f, 'require_no_vm'), patch.object(f, 'attempt', side_effect=[
                (1, "Quota 'GPU_FAMILY' exceeded"), (0, 'created')]) as attempt:
            zone = self.run_search(['us-east1-b', 'us-east1-c', 'us-west1-a'])
        self.assertEqual(zone, 'us-west1-a')
        self.assertEqual([c.args[2] for c in attempt.call_args_list], ['us-east1-b', 'us-west1-a'])

    def test_permission_error_is_not_retried_across_zones(self):
        with patch.object(f, 'require_no_vm'), \
                patch.object(f, 'attempt', return_value=(1, 'PERMISSION_DENIED')) as attempt:
            with self.assertRaisesRegex(RuntimeError, 'Unexpected error'):
                self.run_search(['us-east1-b', 'us-east1-c'])
        self.assertEqual(attempt.call_count, 1)

    def test_preexisting_stopped_vm_is_also_rejected(self):
        vm = dict(name='test-vm', zone='https://example/zones/us-east1-c', status='TERMINATED')
        with patch.object(f, 'existing', return_value=[vm]):
            with self.assertRaisesRegex(RuntimeError, 'TERMINATED'):
                f.require_no_vm('test-project', 'test-vm')


if __name__ == '__main__':
    unittest.main()
