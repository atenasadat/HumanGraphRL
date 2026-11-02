# Multi-domain cognitive shaping

Implemented data ingestion, four-move priors, event pooling, divergence analysis,
LoRA training, held-out evaluation, and local/SLURM execution plans. Full-model
experimental results and cross-domain generalization are **not established**.

## Data and descriptive findings

`Data_proposal_writing/` contains timestamped, source-indexed outputs for P4–P17,
with an ingestion audit. Thirteen participants contribute 3,935 segments; P8 has
an empty output with an explicit exclusion because both speakers were assigned
the same speaker label. Only `*.transcript.json` files are ingested, avoiding
full/part transcript duplication.

The exact start phrase is absent in most recordings. Reviewed source-turn
boundaries and reasons are recorded in `prepare_multidomain.py` and every output;
`proposal_source_hashes.json` prevents silently applying them to changed files.
Conditional references to a future interview are not treated as actual endings.
P14 stops at the first “I'm done with the draft,” per the requested cutoff; this
excludes later revision. Instructor speech, short acknowledgments, and detected
setup chatter are removed. Remaining ASR text, task relevance, and labels need
human validation; filtering is not a claim of perfect diarization or annotation.

The original `human_cognitive_blueprint.json` is preserved. For an additional
comparison using the same label method, programming source segments are split
and reclassified into 3,854 thought units. The new programming/writing/unified
blueprints use a common four-move schema with raw event and transition counts.
Unified priors pool 7,789 events; transitions never cross participant/domain
boundaries. Phases use within-session event index, with cutoffs at 33% and 66%.
Zero-count rows use a documented uniform fallback; potential log probabilities
use an epsilon floor.

| Estimate | Programming, shared classifier | Writing, shared classifier |
|---|---:|---:|
| Explore | 0.65% | 1.63% |
| Evaluate | 95.51% | 97.53% |
| Backtrack | 1.89% | 0.43% |
| Commit | 1.95% | 0.41% |

Transition JS divergence is **0.02873 bits** (equal-weight mean of four row-wise
JS divergences, log base 2). Against the requested original human blueprint,
after dropping `prompt-LLM` and renormalizing each row, it is **0.25740 bits**.
The latter comparison mixes annotation methods and is not a clean domain effect.
See `cross_domain_divergence.json` for every row and move-frequency difference.

All new labels are **unvalidated regex estimates**. The classifier defaults to
`evaluate`; its very high evaluate frequency on human text demonstrates that
these outputs cannot independently validate the earlier model “cognitive
collapse” claim. Writing has slightly more explore, but also more evaluate under
this classifier. These provisional priors need annotation review before use as
publication evidence. Task prompts do not reuse participant transcript text.

## Trainer and invariance

`train.py` uses PEFT LoRA rank 16, alpha 32 on q/k/v/o projections across all
layers. It saves adapters, tokenizer, seed/config, blueprint snapshots, rollout
token IDs, and per-step metrics. Existing run directories are protected from
overwrite. Generation and likelihood computation use the same policy: dropout
is disabled and sampling uses temperature 1 without top-k/top-p truncation.
One on-policy GRPO update is taken per fresh group; reference KL coefficient is
zero. Separate rollout backward passes avoid retaining the whole group's graphs.
The previous negative-advantage cross-entropy sign was reversed; positive
advantage now increases the probability of sampled actions.

For the episodic task objective, gamma is 1. Both initial and terminal potentials
are zero, including budget-truncated outputs. Thus `sum(F_t)=0` exactly up to
floating-point error. A scalar whole-response shaping reward cannot change GRPO
advantages. The previous undiscounted average at gamma=.99 did not preserve
that invariant objective.

The implementation instead applies the exact shaping return-to-go `-Phi(s_t)`
as a token-level control variate. Each potential depends only on the observed
prefix and a fixed token budget, never the eventual output length. The loss uses
a fixed budget divisor, rather than action-dependent response length. This is
an invariant reward transformation, **not a theorem about finite-sample GRPO
convergence or a source of additional expected task-learning signal**.

A lagged controller measures separate task and shaping gradient norms, targeting
35% of their summed norms in the *next* group of the same domain. The current
share and gradient cosine are logged. The target is not guaranteed and is not
the fraction of useful learning. The cap defaults to 10; `--fixed-scale` disables
adaptation. Zero observed task/shape norm sets the next scale to zero. A current
all-equal-outcome group can still receive finite-sample control-variate noise
from its previous scale; `control_variate_only_update` identifies it. There is
no claimed “safety net” that knows which failed answer to improve. Gating the
current scale on current outcomes would invalidate the prefix-only baseline
argument. Reward invariance follows the [potential-shaping result](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf);
LoRA follows the [PEFT configuration interface](https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md).

## Evaluation and execution

Six training prompts cover LCS, knapsack, topological sort, abstract synthesis,
grant introductions, and design tradeoffs. Six distinct held-out prompts cover
other algorithms and writing topics. Each trained adapter is evaluated on both
domains. This is a small diagnostic suite, not a sufficient NeurIPS benchmark.

Training uses domain-specific rubric compliance **proxies**, not executable
coding correctness or validated scientific writing quality. Evaluation reports
backtrack frequency, move entropy, and `H(next move | explore)` (null when no
explore transitions are observed). Task success remains null unless independently
validated boolean judgments are provided. Missing/unfinished reasoning traces
are reported through trace coverage, rather than counted as zero backtracking.

```bash
# Rebuild data and priors (run from repository root).
.venv/bin/python -m experiments.thinkaloud_rl.prepare_multidomain \
  --source /Users/athenasaghi/Desktop/thougthgraph_studyData/raw_thinkaloud_review

# Print/save 27 jobs: 3 training domains x 3 conditions x 3 seeds.
bash experiments/thinkaloud_rl/run_multidomain_local.sh

# Execute locally; use a fresh output directory for each suite.
bash experiments/thinkaloud_rl/run_multidomain_local.sh \
  --output-root runs/multidomain_local --execute

# On CARC after syncing code and installing requirements.txt:
bash experiments/thinkaloud_rl/carc_multidomain_job.sh

# Independently score evaluation samples: JSON {"task_id:sample_number": true/false}.
.venv/bin/python -m experiments.thinkaloud_rl.evaluate \
  --run-dir runs/multidomain_local/programming_unified_seed17 \
  --rescore-only --judgments /path/to/judgments.json
```

The replacement launcher targets b11-12 (A40), d13-08/d14-06/d14-07/d14-08/d14-16
(V100), and e21-12 (P100). It submits seven held arrays with two simultaneous
tasks each, then cancels this user’s previous `cognitive_domains` jobs and releases
the replacements. All 27 indices occur exactly once. Each replacement suite uses
a fresh output directory. Run this script with `bash` on the login node.

All these runs use NF4 quantization with FP16 computation and gradient checkpointing
to reduce memory use on P100. Install `bitsandbytes` in the active virtualenv before
launching. Evaluation reloads the same quantization configuration. Each GPU job
checks a small NF4 forward/backward operation before loading the model. The CUDA,
PyTorch, and bitsandbytes builds must support Pascal for P100; this GPU path has
not been validated locally. CPU regression tests and mocked scheduler success/
failure tests passed. Resource availability remains controlled by Slurm.
No cluster job has been submitted from this session. Local accelerator detection
returned CPU only. The CPU integration test uses a tiny randomly initialized
Qwen model with synthetic rollouts to verify nonzero updates, all-layer LoRA,
checkpoint reload, and real held-out generation for all three conditions; it is
not a training experiment or an accuracy measurement.

Existing tests use package-relative imports; invoke them by module name:

```bash
.venv/bin/python -m unittest \
  experiments.thinkaloud_rl.test_core \
  experiments.thinkaloud_rl.test_inventory \
  experiments.thinkaloud_rl.test_prepare \
  experiments.thinkaloud_rl.test_reasoning_shaper \
  experiments.thinkaloud_rl.test_multidomain \
  experiments.thinkaloud_rl.test_training
```
