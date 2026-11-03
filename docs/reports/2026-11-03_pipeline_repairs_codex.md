# Pipeline repairs and local verification

Date: November 3, 2026. Author: Codex.

**Fixes are local and unpushed. The full experiment is not cleared.**
Software and programming checks pass. The reward judge still rejects a valid
hypothetical proposal and explicit fiction; writing/mixed training could not be
completed within this Mac’s practical memory budget.

## What the pipeline does

Your goal is to teach a model to organize its written reasoning more like your
human think-aloud transcripts. The pipeline groups similar reasoning steps into
graph nodes and connects successive steps. It compares these patterns with human
references while also checking the final answer.

Programming answers face executable tests. A second language model, the
“judge,” rates writing. In the main direct-topology condition, training combines
the answer score with a structure bonus weighted by 0.2. Invalid graphs or risky
answers receive no structure bonus. These are measurable proxies, not proof of
human cognition or factual truth.

## Problems and repairs

| Problem | Cause and repair |
|---|---|
| Empty final answers and zero learning | Models sometimes never closed their reasoning section. Generation now has separate reasoning and answer stages; both sampled sections are trained. |
| Inconsistent training mathematics | Hidden sampling penalties changed generation probabilities. These are disabled, and training uses the declared sampling temperature. |
| Invalid judge responses | Enforce JSON structure, exact candidate quotations, bounded retries and recorded failures. Invalid judgments cannot become rewards. Literal tags inside quoted JSON remain intact. |
| Weak answers passed without flags | Any quality dimension below 2/4 now blocks the output reward, even when the judge omits a risk flag. |
| Repeated work or missing validation | Save before validation, replay pending validation, preserve completed stages, and require missing final validation after a last-step interruption. |
| Successful batches mistaken for learning | Report actual optimizer updates separately. Early checks require updates, finite metrics, sufficient graph coverage and acceptable truncation before the 21 main pilot runs. |
| Correct programming reference rejected | MBPP setup instantiated a class before its definition. The implementation now runs before setup and assertions. |
| Memory overhead and opaque failures | Add configurable SDPA attention, retain partial preflight records and calibration traces, and record judge inputs before long generations. |

The writing rubric now distinguishes factual assertions from proposed methods,
fiction and introductory phrases. **This clarification did not resolve the
judge’s semantic failures.** The expanded live checks catch them and stop the
pilot instead of treating them as acceptable scoring.

CARC startup now checks aligned NF4 and SDPA forward/backward operations.
Those CUDA checks still require execution on CARC.

## What was actually tested

Hardware: 18 GiB Apple Silicon Mac; PyTorch 2.14.1, Transformers 5.20.2.
Local model checks use MPS/bfloat16, not CARC’s CUDA/NF4 configuration.

| Check | Result |
|---|---|
| Mostly Basic Python Problems reference solutions | **464/464 pass** |
| Public-source integrity | 1,019 records checked: 464 MBPP and 555 English WritingBench tasks; provenance and exact split checks pass |
| Human sources | 27 raw-source hashes verified; 571 programming and 3,935 writing segments loaded; 316 graph windows built |
| Sampling/training probabilities | Maximum log-probability difference 0.0000191; nonzero gradients; interrupted generation rejected |
| Qwen2.5-1.5B programming training | **Two actual updates**, 2,416 generated training tokens, 112 changed LoRA B matrices; about 98.7 training seconds |
| Programming validation | Base and trained adapter both scored **0/2** on two validation tasks; no improvement demonstrated. Cached reuse and CSV/PDF reporting pass |
| Structural calibration | All 16 samples produced valid graphs across four training prompts per domain |
| Phi-3.5 reward judge, final rubric | **4/6 pass**. Grounded text and negative cases pass; valid hypothetical proposal and fiction fail |
| Mistral formatting constraints | Real tokenizer permits valid JSON and blocks the earlier invalid verdict/missing-field forms; full Mistral inference untested |
| Recovery and software checks | Nine regression groups pass; forced-crash recovery gives identical tiny-model weights; final validation resumes without retraining |
| Launchers/configuration | Defaults, overrides and pre-submission review blocking pass; 38 pilot stages/21 training runs; static checks pass |
| Writing/mixed training | **Unverified**: local integration did not complete; no nonzero writing update established |

Calibration coverage alone is insufficient: **13/16 calibration samples hit a
generation limit**. In programming training, 62.5% hit the reasoning cap and none
hit the final-answer cap. The early truncation check now exposes this before
expensive main runs. Two updates are an implementation check, not a useful-training
or convergence result.

## Data and evidence

Local template writing tasks were replaced with original public WritingBench
prompts and criteria. WritingBench is model-assisted and human-refined, not
exclusively human-authored; its tasks extend beyond proposal writing. The derived
English split and adapted judging are not its official benchmark score.

Malformed responses, negative judge examples and tiny-model fixtures are
**software tests**, not experimental or participant data. No passing judgments,
human reviews or research results were invented.

See the [verification record](2026-11-03_pipeline_verification_codex.json) and
[local evidence](../../pilots/results/local_validation_2026-11-03/README.md).
Test programs remain outside Git; evidence is saved under the ignored pilots
directory. The [earlier audit](2026-11-02_local_pipeline_audit_codex.md) retains
previous failures and the larger dataset audit.

## Remaining requirements

1. Validate or replace the reward judge using human-reviewed examples, especially
   proposals. Clearer formatting alone cannot fix incorrect assessments.
2. Calibrate generation budgets and verify full-sized writing/mixed training,
   independent Mistral evaluation and checkpoint/resubmission on CARC. A measured
   15 GiB local process footprint prevented reliable combined-model testing.
3. Run the intended pilot horizon, inspect validation trends, and complete human
   reviews before full training. A single seed does not measure seed variability.

The new v4 run/data roots separate changed data and generation protocols from
older results. Existing downloads remain reusable. No code was pushed.
