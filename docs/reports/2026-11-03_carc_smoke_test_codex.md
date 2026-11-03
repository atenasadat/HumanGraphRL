# CARC smoke-test report — 2026-11-03 — Codex

**The whole pipeline has not passed. CARC training is not cleared.**
Programming components completed a fresh real-model integration run after an
evaluator fix. The full suite remains blocked by the writing judge and excessive
reasoning truncation. No code was pushed and no cluster jobs were submitted.

## Fixed and verified

- **Evaluator path:** robust training/evaluation ignored a separately supplied
  LiveCodeBench checkout. The argument now reaches the reward runtime and is
  recorded in the configuration. Resuming the saved adapter completed the real
  LiveCodeBench evaluation without repeating training.
- **Preflight failure:** a failed judge now stops the suite before loading the
  second judge, while retaining all assessed fixtures and naming the failures.
- **Submission:** stale source code or a changed prepared plan now fails on the
  login node before `sbatch`, rather than wasting an allocation.
- **Failure reports:** plots and review packets can be generated when preflight
  fails before the first training directory exists.

## Actual checks

| Check | Result |
|---|---|
| Public MBPP reference solutions | 464/464 pass executable tests |
| Public MATH-500 reference answers | 500/500 pass symbolic verification |
| Official LiveCodeBench dataset loader | Two public records streamed with the required datasets 3.x loader |
| Official LiveCodeBench execution harness | Two intentionally incorrect negative controls rejected |
| Human graphs and structural controls | Rebuilt 316 human windows; reused 16 genuine cached calibration traces; all seven methods scored valid traces in both domains (14 checks) |
| Fresh Qwen2.5-1.5B programming training | Two batches, eight responses, two nonzero optimizer updates |
| Validation and checkpoints | Three validation points; saved adapter reused without retraining |
| Generated LiveCodeBench answer | One public problem passed its official tests; this is not a performance estimate |
| Reports | Training JSON, aggregate JSON/CSV and learning-curve PDF generated |
| Software checks | Nine regressions, three fixed-batch checks and four new failure checks passed |
| Resume and launchers | Exact-weight resume, final-validation recovery, environment defaults and submission guards passed |

The live training run used all 374 public MBPP training prompts as its sampling
pool, one validation task, two prompts per batch and two samples per prompt.
It generated 2,334 training tokens in 84.1 training seconds; validation took
26.7 seconds. These are Mac measurements, not CARC GPU-hour estimates. The
configured pilot's four-by-four batch memory footprint was not tested here.

## Blocking findings

1. **Judge semantics:** the production protocol's cached live assessments reject
   a valid hypothetical proposal and explicitly requested fiction. A fresh
   six-case live trial using a shorter, context-first prompt also failed both.
   That unsuccessful prompt was not adopted. JSON validity alone does not make
   the judge accurate; the independent 7B judge was not validated in this run.
2. **Reasoning truncation:** three of eight fresh training traces hit the
   reasoning limit (37.5%, above the configured 25% threshold). Final answers
   were not truncated. This does not pass the learning-preflight health gate.
3. **Incomplete platform/context coverage:** writing and mixed training have not
   passed end to end. This Mac has Apple MPS and no NVIDIA CUDA; V100 NF4,
   CUDA attention/backward, production batch memory and Slurm execution still
   require a target-GPU smoke test.

The local environment contained datasets 5.1; the repository already requires
3.x for the official LiveCodeBench script. Compatible dependencies were installed
in a temporary directory, leaving the project environment unchanged. The final
programming evaluation ran with that temporary dependency set.

No failure thresholds were relaxed, no human reviews were filled in, and no
synthetic tasks were inserted into research data. Software fixtures and negative
controls are separate from actual model generations and public benchmark records.

Evidence is preserved in
[`pilots/results/carc_smoke_2026-11-03/verification.json`](../../pilots/results/carc_smoke_2026-11-03/verification.json),
with logs, real generated samples and reports in the same ignored directory.
Temporary test programs remain outside Git in `/private/tmp/hgrl_final_audit/`.
The experiment should remain blocked until the judge and truncation checks pass,
followed by the complete smoke run on CARC.
