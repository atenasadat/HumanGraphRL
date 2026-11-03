# Fixed-batch training budget — 2026-11-03 — Codex

Applied locally; no commit, push, GPU submission or research-data change.

| Setting | v4 pilot | v4 full experiment |
|---|---:|---:|
| Training batches per run | 150 | 250, provisional |
| Prompts per batch | 4 | 4 |
| Samples per prompt | 4 | 4 |
| Committed responses per completed run | 2,400 | 4,000 |
| Cumulative token cap | Disabled | Disabled |
| Seeds | 17 | 17, 29, 43 |
| Primary training runs, scheduled first | 6 | 18 |
| Total training runs | 21 | 63 |

`steps` counts processed batches. A batch whose rewards produce zero gradient
does not count as a nonzero `optimizer_update`. Fixing both successful updates
and response exposure is therefore impossible in general. The chosen contract
fixes batch exposure and records actual updates separately. Explicit
`training_stop_rule: fixed_batches` rejects a nonzero cumulative token cap during
planning and before model loading. Health failures and interruptions still stop
training; disabled token caps do not disable per-response truncation limits.

The full default was reduced from 500 to 250 batches. This is an engineering
starting point, not evidence of convergence. Freeze the final horizon using pilot
validation curves before inspecting final holdout results. Generic main/ablation
defaults also disable cumulative caps; historical JSON overrides remain explicit.

Primary GRPO/human-domain runs now precede the five additional methods, across
all seeds and domains. Prerequisites and parent/checkpoint audit dependencies are
preserved. This is scheduling order, not an automatic pause or a rule to discard
controls when results are unfavorable. All planned comparisons remain present.

Reports now explicitly identify the stopping rule, cumulative cap and completion
reason. Existing token/time ledgers and validation curves remain enabled. Tokens
and time are observations, not interchangeable compute units. Raw work also
includes retries, evaluation, calibration, audits and judges. Older dynamic
resampling methods can generate unequal response counts at equal batch counts.

## Verification

Three additional local software checks passed: variable response lengths and tied
rewards preserve batch/prompt exposure; invalid caps fail before model loading;
both v4 plans preserve primary ordering, counts and audit dependencies. The
length test produced 12 versus 48 tokens while both runs processed three batches
and 12 responses, with zero optimizer updates correctly reported.

All nine existing regression checks passed. Controlled-autograd checkpoint tests
passed exact-weight resume, reuse of completed runs, and final-step interruption
followed by validation without retraining. Static checks passed for 24 Python
modules, six JSON configurations, three shell scripts and embedded GPU-check
syntax. Temporary test code remains outside the repository in
`/private/tmp/hgrl_final_audit/`; these fixtures are software tests, not scientific
data or evidence of model improvement.

## Still required

The live judge still fails proposal/fiction semantic checks, so full training
remains blocked. This change does not repair that issue. See the earlier
[pipeline repair report](2026-11-03_pipeline_repairs_codex.md) for measured model
results and the writing/mixed memory limitation.

No target-GPU 7B runtime benchmark was run. Before committing compute, measure
five representative batches and validation with the intended policy, judges,
precision and hardware. Estimate each domain/method separately where costs
differ; include evaluation and auxiliary stages. Do not extrapolate the small
local programming test into a claimed 7B GPU-hour measurement.

Group size eight, grouped generation, 16-bit loading and parallel seed suites
are not enabled. The current two-stage sampler gives each reasoning sample its
own final-answer context; batching needs correct padding, masks, probabilities,
stop handling and memory tests. Policy weights alone do not establish capacity
for judges, gradients, activations and generation caches. Parallel suites also
need separate writable roots and a combined review/report plan. These changes
remain unverified, rather than being advertised as working optimizations.

Changed source/configuration requires a fresh immutable suite plan. Existing
download caches remain reusable. No CARC files or existing experiment artifacts
were modified.
