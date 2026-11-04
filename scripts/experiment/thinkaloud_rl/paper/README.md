# CARC paper experiments

The v5 files are a local candidate using specialist writing judges. Live validation
of the replacements is pending; full training remains gated. See
`docs/reports/2026-11-04_specialist_judges_codex.md` for evidence and limitations.

Current pilot/full launchers use the v5 configurations with public data and two-stage generation.
See `docs/reports/2026-11-03_pipeline_repairs_codex.md` for measured checks and
remaining limitations. The direct-topology method is described in
`docs/reports/2026-11-02_human_topology_pilot_v3_codex.md`. It adds evidence-aware
output rewards, semantic graph rewards, one pilot seed, diagnostic/final partitions,
mandatory judge preflight and human-review gates. The older estimator profiles
below remain available for controlled comparisons and historical reevaluation.

The policy first generates reasoning notes, then a final answer conditioned on
those notes. Both sampled sections participate in the joint policy gradient;
formatting delimiters are supplied by the pipeline and are not rewarded actions.
The per-response token limit is split 50%/50%, with temperature 0.7 and no hidden
repetition penalty or top-k/top-p filtering. Training uses the same probabilities.
The v5 profiles select PyTorch SDPA attention to reduce attention-memory overhead;
legacy profiles retain eager attention unless explicitly overridden. Attention
choice is recorded in the configuration and calibration/judge cache identities.
Before the 21 main pilot runs, three short learning preflights require real
optimizer updates, finite metrics, and the declared graph-coverage and truncation
limits in programming, writing and mixed contexts. There are now 38
sequential stages. These checks do not establish improvement or publication readiness.

The v5 pilot uses 150 training batches; the v5 full configuration provisionally
uses 250, to be reconsidered using pilot validation curves before freezing the
experiment. Both explicitly disable the cumulative training token cap
(`token_budget: 0`, `training_stop_rule: fixed_batches`). Each batch samples four
prompts with four responses each: 2,400 responses per completed pilot run and
4,000 per completed full run, excluding retries and other stages. `steps` counts
batches, including tied-reward batches; `optimizer_updates` counts actual nonzero
updates. These quantities need not match. Failed health checks still stop runs.
The fixed-batch contract rejects a nonzero cumulative cap before loading models.

Primary GRPO/human-domain comparisons run before additional methods across all
seeds and domains: six primary pilot runs and 18 primary full runs. All seven
methods remain planned (21/63 total training runs); ordering does not automatically
pause after primary runs or select controls based on favorable results. There are
five additional methods. Tokens, training/validation time and generation usage
remain measurements. Compare validation curves at common measured compute values;
do not call training wall time active GPU time.

Writing rewards use Atla Selene Mini–8B; evaluation uses Prometheus 2–8x7B (about 25 GB in
4-bit, so jobs request an A100). These
specialist models use their native reference-free scoring templates, not the
historical custom JSON schema. Each scores coherence, relevance, factual grounding
and feasibility separately from 1 to 5. The mean of `(score - 1) / 4` is the
quality proxy; any dimension below 3 zeros the output reward and structure bonus.
This gate is our project protocol, not a verified hallucination detector.
Each answer requires four judge calls, with at most one retry per criterion per
invocation. Invalid/truncated outputs fail closed; completed criteria are cached.
Training
checkpoints on a persistent judge failure and saves weights before validation,
including when validation and checkpoint intervals differ. Pending validation
replays on resume. Partial preflight failures retain their completed checks. See
`docs/reports/2026-11-04_specialist_judges_codex.md` for verification and limitations.
The old constrained JSON backend remains available for historical configurations.
A changed source fingerprint requires a
new suite root; existing model and dataset downloads remain cached.
Calibration now saves every generated trace and per-domain coverage, including
rejected traces, and resumes generation per sample. The current pilot launcher
defaults to `pilots/results/human_topology_pilot_v5/`. Its public-data directory is
`/scratch1/$USER/humangraphrl_public_data_v2`. The old prepared corpus contained
local template writing prompts and is intentionally rejected by the new verifier.
Writing now uses the original English WritingBench prompts and criteria with a
documented derived train/validation/test split. WritingBench is model-assisted
and human-refined; it is not an exclusively human-authored corpus. Cached upstream
downloads remain reusable; old results and adapters are preserved.


This is a new, isolated experiment suite. Use `carc_paper_suite.sh`, rather than
the legacy benchmark scripts or their recorded pass rates. It submits **one GPU
job**, runs conditions sequentially, and resumes completed stages and training
checkpoints. No arrays, automatic submissions, or cancellation of other jobs.

## Run on CARC

Copy `scripts/`, `processed_data/` and `data/think-aloud/` to the repository root on CARC. Keep
`pilots/results/legacy_runs/` if reevaluating existing adapters. Run commands from
the repository root. No compatibility symlinks or root marker are required.

Use the working CUDA Python environment from the previous runs at `scripts/experiment/.venv`, or
set `TRAIN_VENV`. Preparation creates `scripts/experiment/.venv-paper`, inherits the existing CUDA
packages, installs additional dependencies there, downloads official datasets,
freezes upstream/model revisions, caches model weights, and verifies prepared data.
It does not allocate a GPU or submit jobs. Run preparation on a node where
downloads are allowed.

```bash
cd /home1/$USER/HumanGraphRL
export PROFILE=existing
export RUN_ROOT=$PWD/pilots/results/legacy_runs/paper_existing_v1
bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh prepare
bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh submit
```

This reevaluates the six saved mixed-domain baseline/unified adapters from seeds
17, 29, 43, plus the original base model, using corrected evaluators. It does not
retrain them. The original adapters are read only. The old training metadata
remains identified as legacy; their unknown training costs are not invented.

Then run the mechanism pilot:

```bash
export PROFILE=pilot
export RUN_ROOT=$PWD/pilots/results/human_topology_pilot_v5
bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh prepare
bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh submit
```

`prepare` and `submit` must use the same `RUN_ROOT`, `PAPER_DATA`, and
`PAPER_VENV`. Defaults are printed by the launcher. To resume after timeout,
repeat `submit` with the same root. Completed work is skipped. Inspect progress:

```bash
bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh status
tail -f "$RUN_ROOT"/logs/*.log
```

The default allocation is one GPU, 64 GB host memory, and 12 hours. Choose a
different limit through normal Slurm flags, for example:

```bash
sbatch --time=24:00:00 --export=ALL,PAPER_MAX_SECONDS=84600 \
  scripts/experiment/thinkaloud_rl/carc_paper_suite.sh worker
```

No node name or GPU model is pinned. The NF4 GPU preflight rejects unsupported
CUDA/bitsandbytes builds. One allocation can process many runs; the full matrix
will generally need multiple resumed allocations. Queue availability and runtime
are not predictable. Shorter allocations may be easier to schedule.

## Profiles and custom settings

| Profile | Purpose | Default workload |
|---|---|---|
| `existing` | Corrected reevaluation | Six existing adapters + three seeded base evaluations; 50 tasks/domain, four samples/task |
| `pilot` | Mechanism check | Qwen2.5-1.5B, one seed, seven methods × programming/writing/mixed, 150 batches; output/structure/human-review audits; up to 50 diagnostic tasks/domain |
| `ablations` | Isolate human prior | Qwen2.5-7B, three seeds, ten conditions, programming training; 250 batches, cumulative token cap disabled |
| `main` | Main matched comparisons/transfer | Qwen2.5-7B and Mistral-7B, three seeds, programming/writing/mixed training; 250 batches, cumulative token cap disabled |

`main` is a large matrix: 93 training runs per model (186 total), plus base,
synthetic-prior, and gradient-audit stages. T2T is excluded from writing/mixed
training because their rewards are nonbinary. Do not start the whole matrix just
because it fits in one submission script. First inspect the pilot and freeze a
smaller primary comparison if compute is limited.

A smaller ready-to-run primary comparison is provided in
`paper/configs/primary.json`: one 7B model, seven methods, three seeds, programming
training, and evaluation on all three domains (21 training runs). After the pilot:

```bash
export PROFILE=main
export PAPER_CONFIG=$PWD/scripts/experiment/thinkaloud_rl/paper/configs/primary.json
export RUN_ROOT=$PWD/pilots/results/legacy_runs/paper_primary_v1
bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh prepare
bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh submit
```

Override defaults using a JSON file via `PAPER_CONFIG=/path/config.json` during
the first `prepare`. Example:

```json
{
  "models": ["Qwen/Qwen2.5-7B-Instruct"],
  "methods": ["grpo", "dr_grpo", "shrinkage", "human", "shuffled", "synthetic"],
  "seeds": [17, 29, 43],
  "train_domains": ["programming", "writing", "mixed"],
  "steps": 250,
  "token_budget": 0,
  "max_new_tokens": 2048,
  "eval_max_new_tokens": 4096,
  "eval_samples": 4,
  "eval_limit_per_domain": 100,
  "potential_scale": 0.1,
  "controller": "fixed",
  "prior_source": "unified"
}
```

Planning only, without downloads/training:

```bash
python -m scripts.experiment.thinkaloud_rl.paper.suite init \
  --root pilots/results/legacy_runs/paper_plan --data-root /scratch1/$USER/topgrpo_paper_data \
  --profile main --config /path/config.json
```

Plans, prepared tasks, code hashes, model revisions, and run settings are
immutable. Use a fresh root for changed source/configuration. Historical configs
can retain an explicit cumulative token cap; it is checked between batches and
can overshoot. Current v4 configs prohibit that cap. The per-response
`max_new_tokens` limit remains active and truncation is audited. Dynamic-resampling
methods in older profiles can generate different response counts even at equal
batch counts; always report their actual usage.

## Methods actually implemented

These are **matched, single-update implementations**, not reproductions of
published headline scores or their full distributed training recipes. All use
fresh on-policy samples at temperature 1 with no top-k/top-p truncation, LoRA,
no reference KL, and a fixed maximum generation budget. There is one policy
update per fresh batch; importance ratios initially equal one, so PPO clipping
would be inactive. Gradient clipping and Adam remain nonlinear; the zero-mean
control-variate argument concerns the raw score-function gradient.

* `grpo`: group-centered, standard-deviation-normalized advantages and fixed
  budget token divisor. This is the direct outcome-only control for the human
  conditions, and is explicitly different from original sequence-mean GRPO.
* `grpo_sequence`: the same advantages with per-response-length normalization.
* `dr_grpo`: mean-only advantages, fixed budget token divisor.
  [Dr. GRPO](https://arxiv.org/abs/2503.20783).
* `rloo`: leave-one-out reward baseline; fixed budget divisor.
* `shrinkage`: cross-prompt, leave-one-out shrinkage baseline from Eqs. 13–16 of
  [Zeng et al., v2](https://arxiv.org/abs/2511.03710v2), without advantage standardization.
  Requires at least two distinct prompts per batch; four is recommended.
* `t2t`: Eq. 13 of [T2T v4](https://arxiv.org/abs/2602.04265v4), using detached
  group pass rate, alpha=.2, length range [0, token budget]. Binary rewards only.
* `dra`: inverse cosine-row-sum reward weighting from Algorithm 1 of
  [DRA-GRPO](https://aclanthology.org/2026.findings-acl.685/), across the flattened
  batch. Uses frozen `jinaai/jina-embeddings-v2-small-en` on CPU. Its additional
  encoding time is included in step timing. Nonpositive densities are errors.
  The shared outcome reward is retained rather than importing the paper's
  combined format/cosine reward. Report this controlled setting explicitly.
* `dapo_one_update`: dynamic removal of equal-reward groups, soft overlong
  penalty, and token-mean loss. All rejected generations count toward compute.
  Sampling is capped at 32 groups/batch; partial/empty batches are logged.
  This does **not** claim to reproduce multi-update Clip-Higher DAPO. Soft
  penalty reaches -1 at the token budget. See [DAPO](https://arxiv.org/abs/2503.14476).
* `entropy`: outcome-only GRPO plus token-entropy bonus; coefficient .01.
* `learned_prefix`: a lagged two-layer value baseline on seven prefix features,
  trained after the policy update. This is a cheap learned control, not a
  pretrained process reward model.
* `human`, `shuffled`, `uniform`, `frequency`, `transition`, `phase_free`,
  `synthetic`: identical outcome estimator plus `-Phi(prefix)` control variate.
  `shuffled` permutes moves within each human session, preserving counts;
  `synthetic` uses complete traces from the same base model on training prompts.
  Priors use the same pseudocount smoothing. Fixed scale is the primary default;
  `lagged_norm` reproduces the earlier 35% component-norm controller as an ablation.

Human-vs-synthetic sequence counts and lengths are recorded, but not asserted
matched. Human phase uses event position; generated phase uses token position
within the fixed budget. The `phase_free` control tests sensitivity to this
choice. `prior_source=domain_specific` selects matching-domain priors; `unified`
uses the pooled prior. Use separate roots to compare these choices. Frozen
human participant exclusion is configured with `held_out_participants`.

For a strict writing→programming prior transfer claim, choose `prior_source=writing`;
for programming→writing choose `prior_source=programming`. A pooled prior contains
both domains and must not be described as unseen-domain prior transfer. Math is
never used for training in the provided profiles.

## Data and outcome evaluation

* Coding training/validation: official MBPP splits with executable assertions.
  The prompt includes one public assertion to specify the interface. Remaining
  assertions remain verifier information.
* Coding test: official LiveCodeBench release_v6, dates on/after 2024-08-01,
  public **and private** tests through its official evaluator. Dates reduce
  some contamination risk; they do not certify absence from every model's pretraining.
* Math transfer: all MATH-500 tasks, or a deterministic hash-selected subset.
  Math-Verify 0.8.0 parses the **entire** gold expression, including nested fractions.
* Writing training: an explicitly synthetic 72-prompt constrained-writing set,
  with 18 validation prompts on held-out topics. Reward is a length/structure
  proxy, never called writing quality. Its small scale limits generalization claims.
* Writing test: official WritingBench queries/rubrics, kept entirely out of training.
  Quality is null until independent judgments are supplied. It is never converted
  into a Boolean pass rate from a positive proxy score.

All dataset/model revisions, prepared task hashes, complete generations, token
IDs, truncation flags, seeds, and errors are saved. Missing code tests and
evaluator infrastructure failures abort instead of becoming passes. Coding runs
in a child process with timeout, output limits, fresh directory, and a scrubbed
environment. This is **process isolation, not a hostile-code security sandbox**;
use a dedicated compute account/container for stronger isolation. The frozen CPU semantic encoder runs during reference preparation. Generated solution code is evaluated only by worker scoring.

## Audits and reporting

The frozen-policy audit repeatedly samples the same validation prompts without
updating weights. It measures exact gradient covariance/variance traces, task
and shape mean norms, fixed-scale variance ratios, and a coefficient fitted on
one half of groups and assessed on the other. It does not infer variance
reduction from gradient-norm shares. These are raw gradients before clipping
and optimizer preconditioning. Main/primary profiles also audit final programming
checkpoints for `grpo` and `human` at every seed. Set `audit_trained_methods` to
change this list. Intermediate checkpoints are not audited automatically; initial
and final audits alone do not establish the mechanism throughout training.

```bash
python -m scripts.experiment.thinkaloud_rl.paper.report --root "$RUN_ROOT" --plots
python -m scripts.experiment.thinkaloud_rl.paper.audit packet \
  --output "$RUN_ROOT/label_review" --model-samples "$RUN_ROOT/runs"
```

Have two reviewers independently fill `reviewer_a.jsonl` and `reviewer_b.jsonl`.
No script can manufacture independent labels. Report precision/recall, confusion,
agreement, and unknown/disagreement coverage:

```bash
python -m scripts.experiment.thinkaloud_rl.paper.audit report \
  --key "$RUN_ROOT/label_review/private_key.jsonl" \
  --a "$RUN_ROOT/label_review/reviewer_a.jsonl" \
  --b "$RUN_ROOT/label_review/reviewer_b.jsonl" \
  --output "$RUN_ROOT/label_review/report.json"
```

Writing review, separately blinded to model and condition:

```bash
python -m scripts.experiment.thinkaloud_rl.paper.judge packet \
  --root "$RUN_ROOT" --output "$RUN_ROOT/writing_review"
```

Optional local judge generation (on a GPU; cache the chosen model first):

To run judging inside the **same allocation**, set `judge_model` to an independent
judge model ID in your config before preparation. Preparation caches its exact
revision; the last sequential stage generates the blinded packet, judges every
criterion, imports scores, and includes them in the report. The default is null,
so no judge weights or compute are spent without choosing one. Gated models
require access and an existing Hugging Face login during preparation.

Alternatively, run judging manually:

```bash
python -m scripts.experiment.thinkaloud_rl.paper.judge local \
  --packet "$RUN_ROOT/writing_review/packet.jsonl" \
  --output "$RUN_ROOT/writing_review/scores" \
  --model YOUR_INDEPENDENT_JUDGE_MODEL --revision EXACT_MODEL_COMMIT
python -m scripts.experiment.thinkaloud_rl.paper.judge import \
  --packet "$RUN_ROOT/writing_review/packet.jsonl" \
  --key "$RUN_ROOT/writing_review/private_key.jsonl" \
  --scores "$RUN_ROOT/writing_review/scores" \
  --output "$RUN_ROOT/writing_scores.jsonl"
```

Choose a judge family independent of the generators; validate judgments against
blinded human review before claiming writing-quality improvements. Parse failures
remain unscored. This local judge uses the released criteria, not WritingBench's
exact leaderboard model/protocol. Its scores are not directly leaderboard-comparable.

`report.json` and `results.csv` preserve task counts, failures, and truncation.
Pass@k includes all failed tasks and uses `1 - C(n-c,k)/C(n,k)`. Paired intervals
resample training seeds and task IDs, not individual rollouts. They are exploratory
95% intervals, not multiplicity-adjusted significance tests. Predeclare primary
comparisons and use validation only for tuning. Single-seed intervals cannot
measure variation across training seeds.

## External full-recipe reproductions

The native methods isolate individual mechanisms under matched settings. To
compare full published training recipes (e.g. distributed DAPO or Writing-Zero),
use `external_baselines` in the config with concrete `id`, `argv` (a list, never
a shell string), `source_url`, and `source_commit`. They execute sequentially
inside the same allocation. Their pinned environment must be installed before
submission; no unsupported full reproduction is silently approximated.

The command receives `TOPGRPO_PLAN`, `TOPGRPO_DATA`, and `TOPGRPO_OUTPUT`. It must
write `$TOPGRPO_OUTPUT/external_result.json` with `adapter`, `model`, `seed`,
`training_tokens`, `training_seconds`, and `train_task_ids`. The adapter must
load through PEFT. Test-task IDs are rejected, and the common evaluator evaluates
the resulting adapter. The external model must be in the prepared model list.

