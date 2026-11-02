# Dataset inventory and domain pilot

Date: 2026-11-02  
Author: Codex  
Status: configured locally; GPU pilot not run.

## Human programming

Fourteen participants provide 580 extracted transcript segments, yielding 3,854 heuristic cognitive events. These build programming priors and controls; participant transcripts do not become benchmark training questions or verified correctness labels.

## Human writing

Fourteen participant files contain thirteen included sessions and 3,935 extracted writing segments; P8 is excluded. These build writing priors and controls, with heuristic cognitive labels requiring independent human validation later.

## MBPP

MBPP contains 374 training and 90 validation programming problems. We train using executable assertions, monitor held-out validation performance, and reserve separate LiveCodeBench problems for final coding evaluation across all conditions.

## Synthetic constrained writing

Synthetic writing contains 72 training prompts and 18 validation prompts covering separate topics. Training rewards enforce length and structure; they measure constraint compliance, not independently judged writing quality or generalization.

## LiveCodeBench

LiveCodeBench release_v6 supplies coding tests dated August 2024 onward. Exact filtered size is recorded during preparation; pilots select 50 problems, using public and private executable tests, without training on them.

## WritingBench

WritingBench currently provides 1,000 writing queries with evaluation rubrics. Pilots select 50; full experiments use all prepared queries. Independent judgments assess quality; absent judgments remain unscored rather than fabricated successes.

## MATH-500

MATH-500 contains 500 mathematics problems used exclusively for transfer evaluation. Pilots select 50 deterministic problems; full experiments evaluate all 500. Symbolic answer verification measures correctness without introducing mathematics training data.

## Sources and frozen sizes

- [Official MBPP dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/main/README.md): full split; its 500-task test and 10-task prompt splits are not used here.
- [WritingBench official repository](https://github.com/X-PLUG/WritingBench): current benchmark is 1,000 queries; the original release had 1,239. Preparation records the exact downloaded revision and task count.
- [Official MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).
- [Official LiveCodeBench dataset](https://huggingface.co/datasets/livecodebench/code_generation_lite): release_v6, date filter >= 2024-08-01. Filtered count is not known locally until preparation; never substitute the full-release count.
- Human counts come from `processed_data/cognitive_priors_and_audits_from_transcripts/verbalizations_extracted_segments.json`, `cross_domain_divergence.json`, and `processed_data/writing_task_segments_extracted_from_transcripts/*_proposal_verbalizations.json`.
- Synthetic writing counts come directly from `writing_training()` in the benchmark-preparation code.

## Pilot workload and interpretation

Qwen2.5-1.5B-Instruct; seed 17; four methods (GRPO, human, shuffled, uniform);
programming, writing, and mixed training: **12 training runs**. Each gets **150
updates**, four distinct prompts/update, four completions/prompt, at most 1,024
generated tokens/completion, and validation every 15 updates (10 checkpoints of
validation). Validation uses up to 24 tasks/domain, hence 24 coding and all 18
writing validation tasks. Checkpoints are saved every five updates. The initial
frozen-gradient audit uses four validation tasks and 16 groups/task in each of
programming and writing. Base and trained models each evaluate the same 50 test
tasks/domain with four samples/task, generation cap 2,048.

The three training pools contain 374 programming tasks, 72 writing tasks, and
446 mixed tasks. Mixed sampling is uniform over tasks: expected 83.9% coding,
16.1% writing. This preserves the current pooled-data interpretation. A balanced
50:50 experiment is a separate sampling choice, not silently imposed here.
Human priors remain pooled (`prior_source=unified`) across conditions; changing
training-data context does not simultaneously change the prior source.

150 updates provide 600 prompt selections and up to 2,400 completions/run.
This is an initial calibration horizon, not proof of convergence. Inspect validation
trajectories through updates 120/135/150; if still changing materially or noisy,
extend with a fresh root and a declared larger step count. Use validation to choose
learning rate, potential scale, generation length, and the full-training horizon;
do not tune against the final benchmark subset. The 50-task pilot test subsets
remain diagnostic, with final held-out evaluation fixed before paper claims.
A single seed cannot establish cross-seed reliability. Full comparisons use three
seeds and actual token/time accounting.

Compare each trained context on each evaluation domain in `results.csv` and
`report.json`; mixed-training results retain their context label rather than being
averaged into the separate-training results. Learning curves show coding correctness
and writing constraint compliance separately. No numerical average across writing
scores and coding/math correctness is scientifically defined by this suite.

## Before full training

Review split-integrity and evaluator tests; verify finite training values, nonzero
updates, failure/truncation rates, checkpoint recovery, and observed token/time
costs. Inspect frozen-gradient variance ratios rather than assuming a useful prior.
Validate cognitive labels and writing judgments with independent blinded review.
Generated code runs with process limits; use an isolated account/container for
stronger containment. Freeze primary comparisons and resource budgets before
launching the full matrix. Nothing here automatically certifies scientific success.

Use `bash scripts/pilots/run_domain_pilot.sh prepare`, then `submit` on CARC.
The default root is `pilots/results/domain_pilot_v2`; use a fresh root for changed
code/configuration. At this date no GPU run has been launched by this change.

## Final layout update

Development tests and old compatibility links were removed at the user's request.
The environment is now `scripts/experiment/.venv`; full experiment outputs belong
in `processed_data/full_training_and_benchmark_results/`. Python and shell syntax, prior loading,
and dry-run pilot planning were checked after the cleanup. No GPU job was submitted.
