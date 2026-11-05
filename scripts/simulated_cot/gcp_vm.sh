#!/usr/bin/env bash
# Run in Google Cloud Shell. Only "create" starts a billable GPU VM.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PWD"
action="${1:-check}"
PROJECT="${COT_PROJECT:-reasonedllm}"
ZONE="${COT_ZONE:-us-central1-a}"
VM="${COT_VM:-cot-gpu}"

case "$action" in
    check)
        python3 -m scripts.simulated_cot.generate --check-inputs
        bash -n scripts/simulated_cot/gcp_generate.sh
        bash -n scripts/simulated_cot/gcp_vm.sh
        echo 'Cloud Shell checks passed. No VM created.'
        ;;
    create)
        python3 -m scripts.simulated_cot.generate --check-inputs
        git diff --quiet HEAD -- . || { echo 'Commit code changes before packaging the VM.' >&2; exit 1; }
        stage=$(mktemp -d)
        created=false
        finish_create() {
            status=$?
            trap - EXIT
            rm -rf "$stage"
            if (( status != 0 )) && [[ "$created" == true ]]; then
                echo 'VM staging failed; stopping the VM to avoid idle GPU charges.' >&2
                gcloud compute instances stop "$VM" --project="$PROJECT" --zone="$ZONE" --quiet \
                    || echo 'Stop failed; run: bash scripts/simulated_cot/gcp_vm.sh stop' >&2
            fi
            exit "$status"
        }
        trap finish_create EXIT
        # Package code and inputs before the GPU starts accruing charges.
        git archive --format=tar.gz -o "$stage/humangraphrl-code.tar.gz" HEAD
        tar -czf "$stage/humangraphrl-inputs.tar.gz" data/writing_instructions data/programming_instructions
        echo "Creating billable L4 VM $VM in $PROJECT/$ZONE; automatic stop after 12 hours."
        gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" \
            --machine-type=g2-standard-16 --accelerator=type=nvidia-l4,count=1 \
            --image-project=ubuntu-os-cloud --image-family=ubuntu-2204-lts \
            --boot-disk-size=200GB --boot-disk-type=pd-balanced \
            --maintenance-policy=TERMINATE --no-restart-on-failure \
            --max-run-duration=12h --instance-termination-action=STOP
        created=true
        echo "VM is running. If transfer fails, stop it with: bash scripts/simulated_cot/gcp_vm.sh stop"
        gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
            "$stage/humangraphrl-code.tar.gz" "$stage/humangraphrl-inputs.tar.gz" "$VM:~/"
        gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" \
            --command='mkdir -p ~/HumanGraphRL && tar -xzf ~/humangraphrl-code.tar.gz -C ~/HumanGraphRL && tar -xzf ~/humangraphrl-inputs.tar.gz -C ~/HumanGraphRL'
        echo 'Files transferred. Next: bash scripts/simulated_cot/gcp_vm.sh connect'
        ;;
    connect)
        exec gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE"
        ;;
    collect)
        gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" \
            --command="tar -czf ~/cot-results.tar.gz -C ~/HumanGraphRL/data 'Simulated CoT data'"
        mkdir -p processed_data/cloud_downloads
        destination="processed_data/cloud_downloads/cot-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
        gcloud compute scp "$VM:~/cot-results.tar.gz" "$destination" --project="$PROJECT" --zone="$ZONE"
        echo "Results: $PWD/$destination"
        echo "Download with: cloudshell download '$PWD/$destination'"
        ;;
    stop)
        exec gcloud compute instances stop "$VM" --project="$PROJECT" --zone="$ZONE"
        ;;
    *) echo 'Usage: bash scripts/simulated_cot/gcp_vm.sh check|create|connect|collect|stop' >&2; exit 2 ;;
esac
