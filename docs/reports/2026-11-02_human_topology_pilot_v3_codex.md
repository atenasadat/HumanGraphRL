# Human-topology pilot v3: implemented protocol and remaining evidence

Date: 2026-11-02  
Author: Codex  
Status: implemented and checked on CPU; full CUDA/judge pilot has not run.

## Objective and final-output reward

The objective is an operational test of learning human-like reasoning organization,
using differences between collected human trajectories and a frozen base LLM.
It does not establish that a model has human cognition or that its reported reasoning
faithfully describes its internal computation.

New structural conditions directly optimize a rollout-level joint reward:

`R = R_output + 0.2 * S_structure`, with the structural bonus disabled for flagged
final outputs or invalid reasoning traces. Coding output reward comes from executable
tests. Writing output reward is a frozen, evidence-aware judge's coherence, relevance,
factual-grounding and feasibility assessment; unsupported/unverifiable material claims,
fabricated citations/results, nonsense and evaluator-injection flags yield zero output
reward and zero structural bonus. A judge parsing failure aborts rather than becoming
a pass or an ordinary task failure. The previous keyword/length writing scorer remains
available for historical configurations and is not the v3 pilot reward.

The outcome-only baseline receives the same output scorer, prompt protocol and samples
without a structural bonus. The older `human` control-variate method is retained as a
separate condition, clearly distinguished from direct structure optimization.

No automated judge can certify factual truth. Evidence must be supplied to verify a
material claim; a judge's unsupported assertions are not accepted as evidence. A quote
claimed as support must occur verbatim in the supplied evidence. Open-ended material
without evidence can remain unverified. Independent human review is required, not
manufactured. Passing preflight fixtures is necessary but not sufficient.

## Human data and structural representation

Programming now uses 571 task-boundary-reviewed transcript cells from 14 participants;
writing uses 3,935 task-only segments from 13 included participants. Source spreadsheets
and writing transcript JSONs are SHA-256 checked. Because the programming review exports
are now active inputs, their former `v0_` folder prefix was removed.

Reasoning sentences are embedded with the pinned `sentence-transformers/all-MiniLM-L6-v2`
encoder on CPU. Similar sentences form recurrence nodes; temporal transitions form
edges. Features include recurrence, branching, directed edge density, reciprocity,
clustering, path length and transition entropy. Human references use 32-step windows
and give each participant equal weight. The reward measures closeness to robust human
feature distributions. This graph is a **semantic-recurrence proxy**, not validated
logical dependency structure or a complete representation of human thinking.

References are separated before fitting:

- Training humans exclude P4, P7, P10 and P11 in both domains: 10 programming and
  nine writing participants remain.
- Diagnostic humans: P4 and P7 in each domain. Used in pilot measurements/review,
  never in the structural reward.
- Final human holdouts: P10 and P11 in each domain. Not exposed in pilot review
  packets or pilot evaluation scores; reserved for full-study evaluation.
- Participant identifiers are domain-qualified; identical P numbers across studies
  do not imply the same person.

A frozen base-policy profile is generated only from training prompts. Its graph
features and differences from human references are saved in `structure/llm_*.json`.
It also supplies the LLM-reference control; no test outputs fit this reference.

All methods use the same domain-specific human evaluation reference for a given
assessment, even when their training rewards use pooled references. This prevents
comparing methods against different measuring sticks.

## Pilot matrix and budgets

- Model: Qwen/Qwen2.5-1.5B-Instruct, LoRA with NF4 on one GPU.
- Contexts: programming, proposal writing, mixed.
- Pilot seed: 17. Full-study configuration retains seeds 17, 29, 43.
- Conditions: outcome-only GRPO; domain-specific human topology; pooled human topology;
  shuffled human topology; frozen-LLM topology; a controlled small-world topology
  analogue; the earlier human-prefix control variate.
- Total: 21 training runs, 150 updates each; four prompts and four responses/prompt.
- Mixed batches contain two programming and two writing prompts. This is a balanced
  task mixture; it does not imply balanced generated-token counts.
- Per run: 600 prompt selections, 2,400 generated training responses, at most 2,457,600
  generated policy tokens before evaluation/audits/judges/retries.
- Validation: step zero and every 15 updates; two responses/task; up to 24 tasks/domain
  (writing has only 18 validation prompts). Checkpoints every five updates.
- Writing training remains a small synthetic 72-prompt constrained-writing dataset.
  This is a diagnostic limitation; the pilot is not evidence of broad writing
  generalization. Coding training uses the prepared MBPP training split.
- Pilot test cap: 50 tasks/domain, four responses/task. Actual available counts are
  reported, not assumed.

Public test datasets are deterministically divided by task ID: approximately 20%
are eligible for pilot diagnostics and the remainder for final evaluation. Full-study
configuration excludes the diagnostic subset, so tuning on pilot feedback does not
reuse those tasks as final tests. Data/model/evaluator revisions and selected evaluation
IDs are frozen during preparation. Pretraining contamination is not ruled out by this
split or by coding-task dates.

The primary comparison is domain-specific human topology versus outcome-only GRPO
on held-out-human graph similarity in the corresponding domain(s). Pooled/shuffled,
LLM-reference and small-world comparisons are secondary. Outcome quality and correctness
are separate guardrails. Primary intervals receive a Bonferroni family adjustment using seed/task bootstrap
resampling; other intervals are exploratory. With one pilot seed, these measure only
task-sampling uncertainty: cross-seed variability is unmeasured. They are not a
publication guarantee or a substitute for power analysis.

The small-world condition is a controlled implementation using this pipeline's
recurrence graphs, not an exact reproduction of SARL's released recipe.

## Mandatory execution and reporting

The launcher uses `configs/human_topology_pilot_v3.json` and a fresh root:

```bash
bash scripts/pilots/run_domain_pilot.sh prepare
bash scripts/pilots/run_domain_pilot.sh submit
```

Default output: `pilots/results/human_topology_pilot_v3/`.
Copy `scripts/`, `processed_data/`, and `data/think-aloud/` to CARC before preparation.
Set `TRAIN_VENV` if the working CUDA environment is elsewhere. Preparation freezes
weights and builds references; it does not submit jobs. Required extra models:

- Training writing judge: microsoft/Phi-3.5-mini-instruct.
- Independent evaluation writing judge: mistralai/Mistral-7B-Instruct-v0.3.
- Structural encoder: sentence-transformers/all-MiniLM-L6-v2.

Judges are frozen and pinned; policy and training judge can occupy GPU memory together.
Actual memory compatibility must pass on the target CUDA allocation. Training and
final-evaluation judge models are distinct. These model choices require live calibration;
they are not asserted to be reliable merely because they are instruction-tuned.

Before training, both judges must pass grounded, fabricated-result, gibberish and
judge-injection fixtures. Failure stops the run. The policy then creates its training-only
structural profile. Initial and final gradient audits run on programming and writing
for the outcome-only and prefix-control methods; they measure the control-variate
hypothesis, not the direct graph-reward effect.

Every training run automatically writes:

- `training_report.json`: actual committed tokens, optimizer updates, wall time,
  peak allocated GPU memory, health, and validation-tail diagnoses.
- `validation_curves.csv`: per-domain output/structure/risk trajectories against
  generated training tokens and training time.
- `steps/*.json`: separate output reward, structural bonus, risks and diagnostics.
- `validation/` and `test/`: complete generations, scores, failures, truncation,
  graph extraction coverage, held-out-human similarity, cue-removal sensitivity,
  clustering-threshold sensitivity and length-stratified structure results.

The suite writes `report.json`, `results.csv`, `learning_curves.pdf`,
`pilot_readiness.json` and `pilot_report.md`. Partial reports are also produced when
a stage fails or checkpoints for resumption. Append-only telemetry records completed
policy and judge generation calls, including retries, rejected and replayed work.
Allocation-attempt timings are separate from committed update time. Hard-killed partial
generations cannot be fully measured and remain an accounting limitation. Wall time is
not GPU-kernel time; FLOPs are not fabricated. Reporting weights are not a guarantee of
matched total compute—judge and prefix-audit overhead are included separately.

The structural proxy is checked under cosine thresholds 0.6/0.7/0.8 and removal of
common discourse cue words. This is a sensitivity diagnostic, not a proof of paraphrase
invariance. Length bins are descriptive controls, not a causal adjustment for every
length effect. Human semantic-graph and final-output packets are generated automatically
for two independent blinded reviewers, with agreement, risk precision/recall against
human consensus, and quality-score error reported. Unknowns/incomplete reviews remain
pending; consensus subsets and included fixtures do not estimate population hallucination
rates without further sampling assumptions.

After independent reviews are filled, refresh readiness with:

```bash
python -m scripts.experiment.thinkaloud_rl.paper.report \
  --root pilots/results/human_topology_pilot_v3 --plots
```

Engineering diagnostic thresholds are predeclared in the config: graph coverage .7,
truncation at most .25, cue-score shift at most .1, threshold-score range at most .2,
human score at least 3/4, reviewer score disagreement at most 1, review risk fraction
at most .1, judge risk recall at least .8 and score MAE at most .25. These are calibration
thresholds, not established scientific constants; changing them requires a new frozen
run root. Validation-tail tests use the last four points and a .01 tolerance. A flat tail
is explicitly not called a convergence proof.

Full-training launcher now uses `human_topology_full_v3.json`. It refuses to start
until the required pilot's automated and independent-human-review checks are complete.
Its 500-update/1M-policy-token limits are provisional caps to revise from pilot
validation; token limits can stop a run before the update cap.

## Verification performed locally

- Pure graph construction, shuffled-control invariants and bounded similarities.
- Strict judgment parsing and zero rewards for flagged/unverifiable outputs.
- Trend detection, incomplete-human-review handling and conservative readiness status.
- Real tiny-Qwen autograd/LoRA/Adam updates and bit-identical interrupted/resumed
  adapter weights, with correct response/token counts and automatic final reports.
- Real cached MiniLM encoder run over all human source text and reference construction.
- Dry-run planning and disjoint public evaluation partitions.

Temporary verification code lives outside the repository; deleted development test
folders were not recreated. No new GPU training, large model download or live judge
calibration was performed. Remaining evidence must come from the target CUDA run and
independent reviewers. No software can guarantee hallucination-free prose or establish
human cognition from semantic graph similarity alone.

Human-reference uncertainty is not resampled by the seed/task bootstrap. Only two diagnostic and two final human holdouts per domain are available; population-level human generalization remains a limitation.

The pilot also aborts after ten consecutive batches without an outcome-driven gradient, saving a checkpoint and partial diagnostics. This catches degenerate reward/format/difficulty settings before spending the rest of the matrix budget; the threshold is a declared engineering safeguard.

Pilot-size update: at the user's request, only seed 17 is scheduled (21 training runs, 35 sequential stages). No pilot runs for seeds 29 or 43 had been launched or produced outputs to delete. Full-study seeds remain separate.
