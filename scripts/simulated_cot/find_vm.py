"""Try approved L4 zones sequentially; stop at the first created/staged VM."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

from scripts.simulated_cot.generate import check_inputs, DOMAINS

ROOT = Path(__file__).resolve().parents[2]


def cloud_json(project, *args):
    result = subprocess.run(['gcloud', *args, f'--project={project}', '--format=json'],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def l4_quota(info, region):
    """Honor specific dimension overrides over inherited/default quota values."""
    matches = []
    for item in info.get('dimensionsInfos', info.get('dimensionsInfo', [])):
        dims = item.get('dimensions', {})
        if dims.get('gpu_family', 'NVIDIA_L4') != 'NVIDIA_L4':
            continue
        if dims.get('region', region) != region:
            continue
        locations = item.get('applicableLocations', [])
        if locations and region not in locations:
            continue
        matches.append((len(dims), float(item.get('details', {}).get('quotaValue', 0))))
    return max(matches, default=(0, 0))[1]


def candidate_zones(catalog, quota, regions=()):
    zones = set()
    for machine in catalog:
        if (machine.get('name') != 'g2-standard-16' or machine.get('guestCpus', 0) < 16
                or machine.get('memoryMb', 0) < 65536):
            continue
        zone = machine['zone'].rsplit('/', 1)[-1]
        region = zone.rsplit('-', 1)[0]
        if regions and region not in regions:
            continue
        if l4_quota(quota, region) >= 1:
            zones.add(zone)
    # Prefer the regions already discussed, then other US zones, then worldwide.
    def order(zone):
        region = zone.rsplit('-', 1)[0]
        preference = 0 if region == 'us-central1' else 1 if region == 'us-east1' else 2 if region.startswith('us-') else 3
        return preference, zone
    return sorted(zones, key=order)


def existing(project, vm):
    return cloud_json(project, 'compute', 'instances', 'list', f'--filter=name={vm}')


def require_no_vm(project, vm):
    found = existing(project, vm)
    if found:
        summary = ', '.join(f"{v['name']} in {v['zone'].rsplit('/', 1)[-1]} ({v['status']})" for v in found)
        raise RuntimeError(f'An instance already exists: {summary}. No more VMs will be attempted. '
                           'Inspect it with gcloud compute instances list before retrying.')


def attempt(project, vm, zone, log):
    env = dict(os.environ, COT_PROJECT=project, COT_VM=vm, COT_ZONE=zone)
    lines = []
    with log.open('w') as output:
        with subprocess.Popen(['bash', str(ROOT / 'scripts/simulated_cot/gcp_vm.sh'), 'create'],
                              env=env, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                output.write(line)
                output.flush()
                lines.append(line)
            status = process.wait()
    return status, ''.join(lines)


def search(project, vm, zones, log_dir):
    blocked_regions = set()
    for zone in zones:
        region = zone.rsplit('-', 1)[0]
        if region in blocked_regions:
            continue
        require_no_vm(project, vm)
        print(f'\nTrying {zone}: one L4 24GB, 16 vCPUs, 64GB RAM, 200GB disk.', flush=True)
        status, output = attempt(project, vm, zone, log_dir / f'{zone}.log')
        if status == 0:
            print(f'\nSUCCESS: {vm} created and files transferred in {zone}. Search stopped.')
            print('Next: bash scripts/simulated_cot/gcp_vm.sh connect')
            return zone
        # A timeout, interrupted insert, or staging failure can leave an instance.
        # Never try a second zone while an instance with this name exists.
        require_no_vm(project, vm)
        if re.search(r'GPUS_ALL_REGIONS|GPUS-ALL-REGIONS', output):
            raise RuntimeError('Global GPU quota exceeded. Search stopped.')
        if 'QUOTA_EXCEEDED' in output or re.search(r'quota.*exceeded', output, re.I):
            print(f'Skipping the remaining zones in {region}: quota unavailable.', flush=True)
            blocked_regions.add(region)
        elif re.search(r'RESOURCE_POOL_EXHAUSTED|STOCKOUT|RESOURCE_NOT_READY', output):
            print(f'No capacity in {zone}; trying the next zone.', flush=True)
        else:
            raise RuntimeError(f'Unexpected error in {zone}; search stopped. See {log_dir / (zone + ".log")}')
    raise RuntimeError('No VM created: all eligible zones failed. Other regions need approved L4 quota, '
                       'or capacity must become available. See the per-zone logs.')


def main():
    os.chdir(ROOT)
    project = os.environ.get('COT_PROJECT', 'reasonedllm')
    vm = os.environ.get('COT_VM', 'cot-gpu')
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,61}[a-z0-9]|[a-z]', vm):
        raise ValueError('Invalid COT_VM instance name')
    check_inputs(list(DOMAINS))
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', '.']).returncode:
        raise RuntimeError('Commit code changes before searching; the VM receives committed code.')
    work = ROOT / 'processed_data'
    work.mkdir(exist_ok=True)
    with (work / '.cloud_vm_search.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another VM search is already running in this checkout') from None
        require_no_vm(project, vm)
        catalog = cloud_json(project, 'compute', 'machine-types', 'list', '--filter=name=g2-standard-16')
        quota = cloud_json(project, 'quotas', 'info', 'describe',
                           'GPUS-PER-GPU-FAMILY-per-project-region',
                           '--service=compute.googleapis.com', f'--billing-project={project}')
        regions = os.environ.get('COT_REGIONS', '').replace(',', ' ').split()
        zones = candidate_zones(catalog, quota, regions)
        if not zones:
            raise RuntimeError('No supported zones have approved L4 quota. Approve a regional L4 quota first.')
        print(f'{len(zones)} supported, quota-approved zones: {", ".join(zones)}', flush=True)
        print('Trying one zone at a time. A successful creation starts billing; regional prices vary.', flush=True)
        log_dir = work / 'cloud_search' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log_dir.mkdir(parents=True, exist_ok=True)
        search(project, vm, zones, log_dir)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('Search interrupted. Check gcloud compute instances list before retrying; a VM may exist.', file=sys.stderr)
        sys.exit(130)
