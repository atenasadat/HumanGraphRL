# Simulated CoT on Google Cloud L4

Code is on `pipeline-v4-carc-readiness`. The task inputs are private, gitignored
research files: upload the provided `humangraphrl-inputs.zip` to Cloud Shell's
home directory. Do not install vLLM or download weights in Cloud Shell's small
home directory. All model execution happens on the GPU VM.

## Before creating a VM (Cloud Shell)

```bash
cd ~/HumanGraphRL
git fetch origin
git switch --track origin/pipeline-v4-carc-readiness
git pull --ff-only
unzip -o ~/humangraphrl-inputs.zip -d ~/HumanGraphRL
bash scripts/simulated_cot/gcp_vm.sh check
```

If the local branch already exists, use `git switch pipeline-v4-carc-readiness`.
The check validates every task's manifest hash with standard Python; it creates
no VM and requires no CUDA packages.

## Create and connect (Cloud Shell)

To try all supported zones with approved L4 quota, stopping at the first VM
successfully created and staged:

```bash
bash scripts/simulated_cot/gcp_vm.sh create-any
bash scripts/simulated_cot/gcp_vm.sh connect
```

The search keeps the same hardware configuration, prefers `us-central1` and
`us-east1`, then other US regions, then the rest of the world. Regional prices
vary. Regions without approved quota are skipped; the script does not submit
quota requests. To restrict the search, set for example
`COT_REGIONS="us-central1 us-east1"`. The exported `COT_ZONE` is ignored for
`create-any`. Logs are saved under `processed_data/cloud_search/`.

There is no guarantee of live capacity from the zone catalog: the script tests
availability by trying actual creation sequentially. It stops on success, on an
unexpected error, or if an instance with the same name already exists (including
after an ambiguous failed API call). Concurrent searches in the same checkout
are prevented by a lock. Connect/collect/stop discover the VM's actual zone.

To try only a specific zone instead:

```bash
bash scripts/simulated_cot/gcp_vm.sh create
bash scripts/simulated_cot/gcp_vm.sh connect
```

Creation starts a billable `g2-standard-16` (one L4 24GB, 64GB system RAM,
200GB persistent boot disk) in `reasonedllm/us-central1-a`. The script packages
the committed code and checked inputs before creating the VM, then transfers
them. It does not use the outdated default GitHub branch or require GitHub
credentials on the VM. Override `COT_PROJECT`, `COT_ZONE`, or `COT_VM` if needed.
The VM stops after 12 hours per start; this can interrupt a batch. Its persistent
disk continues costing money while stopped. No local SSD is used.

If creation reports resource exhaustion, use another G2-supported zone in
`us-central1`. If file transfer fails after creation, the launcher attempts to
stop the new VM. If that also fails, use the stop command before troubleshooting.

## Install and test (inside the GPU VM)

```bash
cd ~/HumanGraphRL
bash scripts/simulated_cot/gcp_generate.sh drivers
bash scripts/simulated_cot/gcp_generate.sh setup
bash scripts/simulated_cot/gcp_generate.sh smoke
```

The driver action uses Google's official production-branch installer. It can
reboot the VM: reconnect, `cd ~/HumanGraphRL`, and repeat `drivers` if interrupted.
If it finishes but `nvidia-smi` fails, run `sudo reboot`, reconnect, and repeat
`drivers`. Setup creates an isolated venv, installs pinned vLLM 0.11.0, checks
BF16 CUDA execution and downloads the three pinned model revisions (~45GB).
The vLLM wheel supplies its CUDA runtime; no separate CUDA toolkit is needed.

The smoke action loads each actual model and generates eight test tokens. Its
records go to `data/Simulated CoT data/_smoke/`, never production `_records/`.
It checks loading and GPU kernels, not completion quality or full-run duration.
Stop and inspect the relevant `processed_data/cloud_logs/*_smoke.log` if it fails.

## Run and resume (inside the GPU VM)

```bash
tmux new -s cot
cd ~/HumanGraphRL
bash scripts/simulated_cot/gcp_generate.sh run
```

Models run sequentially. Each response is saved before the next. Completed
records are skipped when repeating `run`; a response interrupted before saving
is rerun with the same seed. To run one model, append its name, for example
`bash scripts/simulated_cot/gcp_generate.sh run Qwen3-8B`.
Detach using Ctrl+B, then D. Reconnect using `tmux attach -t cot` after SSH.

The L4 profile preserves model revisions, BF16/native MXFP4 precision, prompts,
sampling, seeds, 40,960-token context and 32,768-token generation budget.
It schedules one request at a time, chunks prefill at 512 tokens and disables
CUDA graphs. Qwen3 offloads up to 4GiB of weights to system RAM; this reduces GPU
memory use but can substantially slow generation. Engine settings are recorded
with every result. No model is silently replaced or newly quantized.
The original CARC launcher keeps its standard engine configuration.

These memory settings require a real L4 smoke test before treating the setup as
verified. No NVIDIA GPU is available in the local development environment.

## Download and stop (back in Cloud Shell)

Detach tmux and run `exit` to leave the VM, then:

```bash
cd ~/HumanGraphRL
bash scripts/simulated_cot/gcp_vm.sh collect
# Run the cloudshell download command printed by collect.
bash scripts/simulated_cot/gcp_vm.sh stop
```

When restarting an automatically stopped VM, use
`gcloud compute instances start cot-gpu --project=reasonedllm --zone=us-central1-a`,
reconnect, and repeat `run`. Delete the VM and its disk after downloading and
verifying the results if no longer needed.

References: [Google GPU drivers](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu),
[G2 limitations](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#G2_limitations),
[vLLM memory configuration](https://docs.vllm.ai/en/v0.11.0/configuration/conserving_memory.html).
