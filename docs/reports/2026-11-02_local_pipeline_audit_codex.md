# Local pipeline audit — candidate v4

Date: 2026-11-02. Author: Codex.

This is the earlier audit, before the two-stage and sampling-probability fixes.
See [the subsequent repair report](2026-11-03_pipeline_repairs_codex.md) for current status.

**Release status: not ready; changes remain unpushed.** Local software checks
passed, but real generation/training checks exposed an unresolved output-format
and reward-sparsity problem. No complete CARC pilot or publication result is
claimed. The v4 files are a local candidate, not an approved experiment release.

## Data actually checked

| Public source | Train | Validation | Test | Use |
|---|---:|---:|---:|---|
| Mostly Basic Python Problems (MBPP) | 374 | 90 | 0 | Programming training and executable validation |
| WritingBench, English subset | 313 | 69 | 173 | Writing training, validation and evaluation |
| LiveCodeBench, release_v6, contests from 2024-08-01 | 0 | 0 | 454 | External programming evaluation |
| MATH-500 | 0 | 0 | 500 | Math transfer evaluation |
| Total | 687 | 159 | 1,127 | 1,973 public tasks |

The source preparation completed locally. Hash verification, required provenance,
and normalized-prompt split checks passed. The diagnostic/final evaluation
partition contains 231/896 disjoint task IDs. Exact matching does not establish
absence of semantic overlap or pretraining contamination.

The previous local template writing prompts were removed from preparation.
[WritingBench](https://github.com/X-PLUG/WritingBench) uses model assistance and
human refinement; public provenance does not make it exclusively human-authored.
The original English prompts and criteria are retained. A deterministic,
normalized-prompt hash assigns a nominal 60/10/30 split, with the actual counts
above. This is a derived split and adapted judging protocol, **not the official
WritingBench evaluation score**. Its broad writing tasks are not exclusively
research-proposal writing. Prompt material is supplied evidence, not independently
verified factual truth.

Pinned source revisions:

- MBPP: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- MATH-500: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- LiveCodeBench data: `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.
- LiveCodeBench evaluator: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- WritingBench: `ae2d5176449b7b769815482641d35926f26793eb`.

All 27 included raw human-source file hashes were checked. All 571 processed
programming cells and 3,935 writing segments matched their raw source content
after applying the already documented researcher-prompt removal: **4,506 checked,
zero unmatched**. This verifies processing provenance, not the scientific validity
of a graph interpretation or human review that has not happened.

## Observed defects and changes

- Actual Qwen generation reproduced the calibration failure: four initial
  samples lacked extractable reasoning blocks. The candidate uses an explicit
  opening `<think>` prefix and clearer reasoning/final-answer instructions.
  Calibration signatures include this protocol. This improves graph coverage,
  but **does not solve reliable closing tags or final answers**, as shown below.
- One official MBPP reference failed because setup instantiated a class before
  its definition. The verifier now executes submitted code before test setup.
  All 464 train/validation reference solutions then passed.
- Training/preflight no longer materialize every multi-gigabyte hidden-test
  payload. Task selection streams records; preparation writes hashed train/val
  indexes. Untracked indexes are ignored. Tampered tracked indexes are rejected.
- Old manifests/local-synthetic task sources are rejected. New launcher defaults
  use `human_topology_pilot_v4` and `/scratch1/$USER/humangraphrl_public_data_v2`.
  Full v4 requires that pilot's readiness checks. Old outputs remain untouched.
- Nonquantized loading supports an explicit dtype. MPS defaults to float32:
  an actual float16 calibration attempt failed with nonfinite sampling probabilities.
  Judge dtype can be selected independently for local integration checks.
- Checkpoints now include MPS RNG state. Reports distinguish processed batches
  from actual optimizer updates; a zero-gradient batch is not reported as an update.
  Learning-curve time labels correctly say wall-clock hours.

## Executed local checks

Hardware: Apple Silicon Mac, 18 GiB RAM, PyTorch 2.14.1, CPU/MPS; no CUDA.
Models use pinned pretrained weights. Test programs and generated artifacts are
outside the repository under `/private/tmp/hgrl_full_audit`.

| Check | Observed result |
|---|---|
| Human processing provenance | All 4,506 included segments match source content |
| Full public data integrity | 1,973 tasks; hashes/provenance/exact split checks pass |
| Prepared subset indexes | Real-source subset build, index consistency, tamper rejection and no-redownload resume pass |
| Streaming selection | 12 comparisons match the previous stable hash ordering |
| MBPP verifier | 464/464 original reference solutions pass |
| Math verifier | 500/500 original gold-answer roundtrips pass |
| Official LiveCodeBench harness | Both published function examples pass all five example cases |
| Phi-3.5 reward judge | All four live sanity fixtures pass on MPS bfloat16 |
| Smaller Qwen judge candidate | Failed sanity checks; not substituted for either configured judge |
| Qwen2.5-1.5B structural calibration | 16 real samples; graph reference construction completes |
| Qwen2.5-1.5B programming training | Two batches, eight training responses; zero optimizer updates |
| Reduced Qwen2.5-0.5B mixed training | One balanced batch, four responses; zero optimizer updates |
| Checkpoint integration | Real adapter save/load, completed-run reuse and MPS RNG restoration pass |
| Failure/resume components | Judge failures, pending validation, calibration resume, zero-gradient stop and optimizer/RNG recovery pass |
| Reports | Real validation CSVs and curve PDF generated; zero updates retained explicitly |
| Launchers/plan | Defaults and explicit overrides pass; 35 stages, 21 training runs, one seed; missing human-review prerequisite blocks full training |
| Static checks | Python/JSON parsing, Bash syntax and Git whitespace checks pass |

Ten component scripts passed. Some intentionally use controlled fixtures,
mocked services or a tiny initialized model to test failure paths. Those are
software tests, **not experimental data, human observations or measured model
performance**. The judge's explicitly fabricated negative examples are sanity
fixtures only. No fixture, test program or test result is added to the paper data.

Token-budget checks used all 555 English WritingBench prompts and actual
tokenizers. Maximum policy input was 3,820 tokens; judge inputs with an empty
candidate were 10,865 for Phi and 10,590 for Mistral. These fit the configured
8,192/16,384 input limits. Candidate outputs consume additional tokens, so this
is not a guarantee that every later judgment fits; runtime budget checks remain.

## Failed readiness checks

The actual 1.5B calibration used 1,024 generated-token caps and two samples on
four distinct training prompts per domain. Programming produced 8/8 valid graphs;
writing produced 5/8. Three programming and four writing generations were
truncated. A valid reasoning graph is not a valid final answer: all eight cached
programming completions failed their final-code verifier. Calibration's minimum
prompt-coverage gate passed, but this is insufficient evidence to start training.

The actual 1.5B programming run used the production model revision, real public
tasks, real human references, two batches of two prompts/two samples, and a
reduced 512-token cap. It generated 3,542 training tokens in about 211 training
wall seconds. Both batches had zero outcome reward and zero optimizer updates.
Greedy validation also failed to produce usable final code. Some generated code
appeared inside the reasoning block, or the block never closed; extracting it as
a final answer would silently change the declared protocol and was not done.

The reduced mixed run used a 0.5B policy, MPS bfloat16, one balanced batch and a
384-token cap. It generated 1,332 training tokens, also with zero updates.
Writing outputs lacked final answers, so the runtime rejected them before
invoking the judge. This tests the empty-output gate and mixed path; it **does not
verify successful joint policy/judge training**. The separate actual Phi sanity
test did invoke the configured reward judge and passed.

A further temporary diagnostic removed the forced opening prefix and used more
explicit two-section instructions at the intended 1,024-token cap. Two of four
outputs still lacked complete reasoning/final sections; neither of the two
programming answers passed its verifier. The alternative prompt was not adopted.
This failure cannot be explained solely by the shorter training smoke-test cap.

These are small engineering checks, not representative performance estimates.
Shorter generation caps and different local precision limit comparison with
CARC. Nevertheless, they disprove a claim that successful calibration or absence
of exceptions alone establishes learning readiness. Reports preserve the failures;
no passing scores, human reviews or successful optimizer updates were invented.

## Still required before release/full training

1. Establish a reliable reasoning/final-answer protocol with the selected policy,
   then demonstrate nonzero, finite updates on real programming, writing and mixed
   batches at the intended pilot budgets. Increasing update count alone does not
   repair missing final answers or all-zero rewards.
2. Exercise actual writing reward integration and independent Mistral evaluation.
   Full Mistral inference was not run locally: model weights/memory exceed the
   remaining local resource budget. Tokenizer/schema tests are not equivalent.
3. Validate the exact V100 CUDA/NF4 environment, full worker sequence, interruption
   and resubmission on CARC. CPU/MPS checks cannot certify CUDA kernels or Slurm.
4. Complete blinded human graph/output reviews and judge calibration; inspect
   graph validity, truncation and sensitivity against declared limits. No reviewer
   scores are supplied by this audit.
5. Run the intended pilot horizon and inspect actual validation curves. One seed
   does not estimate cross-seed variability. No smoke test certifies NeurIPS
   readiness, human-like cognition or hallucination-free output.

The fresh v4 prepared-data directory is necessary because the former writing
training corpus was locally templated. Cached upstream/model downloads can be
reused, but old prepared tasks and calibration outputs cannot be relabeled as v4.
The launcher changes are unpushed while these release conditions remain unresolved.
