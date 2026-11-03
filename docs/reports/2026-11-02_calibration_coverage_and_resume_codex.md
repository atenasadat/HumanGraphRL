# LLM calibration coverage and resume

Date: 2026-11-02. Author: Codex.

CARC completed the constrained judge preflight, then failed LLM structural
calibration with `At least two human participants per reference`. This wording
came from a shared reference aggregator. For LLM calibration its grouping unit
is a distinct training prompt, with repeated generations averaged per prompt.
The failure establishes that at least one domain has fewer than two prompts
producing eligible reasoning graphs. It does not establish which domain failed
or why its traces were ineligible. The old code saved samples only after building
the references, so the failure lost both usable and rejected generated traces.

The fix retains the minimum of two distinct eligible training prompts per domain.
Human references still require at least two human participants. Calibration now:

- Atomically saves every generation before graph extraction, including raw trace,
  token IDs, prompt-token count and truncation flag.
- Saves each graph result or rejection reason and resumes cached samples without
  regenerating them. An interrupted graph calculation can reuse the saved text.
- Exports all traces to `structure/llm_Qwen_Qwen2.5-1.5B-Instruct.samples.jsonl`.
- Writes `structure/llm_Qwen_Qwen2.5-1.5B-Instruct.coverage.json` before checking
  reference eligibility. It reports domain counts, valid prompt/sample counts,
  truncation counts, and missing/incomplete think blocks or short/long traces.
- Fails with domain-specific counts and audit paths if coverage is insufficient.
- Rejects changed calibration settings or test prompts instead of reusing an
  incompatible cache. Cache identity includes model/encoder revisions, prompts,
  generation budgets, graph threshold and trace-length bounds.

This is a diagnostic and resume fix. It does not establish that the actual base
model produces enough usable reasoning traces. The saved CARC traces are needed
before changing the generation protocol or model. No human-like reference is
invented and the threshold is not relaxed to get the stage through.

Temporary verification with mocked generation and graph encoding checked a
writing-domain coverage failure, persistence of all eight sampled traces,
zero repeated generations on resubmission, successful independent domain/pooled
references, interruption/resume, changed-settings rejection, train-only inputs,
and the unchanged human-participant minimum. Real base-model calibration has not
been rerun locally.

## CARC update

The source fingerprint changed, so use the launcher's new default root. In a
fresh terminal, with no older exported overrides:

```bash
cd ~/HumanGraphRL
git pull --ff-only
bash scripts/pilots/run_domain_pilot.sh prepare
bash scripts/pilots/run_domain_pilot.sh submit
```

The default is `pilots/results/human_topology_pilot_v3_calibration_audit/`.
Existing dataset/model downloads are reused. Previous calibration generations
were not saved, so those must be generated once more. This implementation saves
them for subsequent resumes. If calibration fails again, inspect its coverage
JSON and sample JSONL before another submission.
