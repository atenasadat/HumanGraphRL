# Specialist writing judges — 2026-11-04 — Codex

The v5 pilot and full-training configurations replace the general instruction
judges with open-weight models trained specifically for evaluation. Changes are
local; nothing was committed or pushed. Live GPU validation is still pending.

| Role | Model | Pinned revision |
|---|---|---|
| Training reward | `AtlaAI/Selene-1-Mini-Llama-3.1-8B` | `427792f1c3e2073cb7da216924fd884b1ba496e0` |
| Independent evaluation | `prometheus-eval/prometheus-8x7b-v2.0` | `2db013b60e3e91f7a06113e436410899768e8228` |

Update (same day): roles swapped. Selene Mini reports stronger judge-benchmark
results than Prometheus 2–7B, so it scores the reward; evaluation uses the larger
Mixtral-based Prometheus 2–8x7B so the evaluator is stronger than, and from a
different family than, the reward judge.

Both checkpoints are publicly downloadable without a paid judging API. GPU
compute and storage still cost resources. Model cards list Apache 2.0; Selene is
derived from Llama 3.1, whose applicable base-model terms also matter.

## Literature and selection

- [Prometheus, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/803485352e61e3ebf41221e4776c9fd4-Abstract-Conference.html)
  motivates evaluation-specific training and explicit rubrics. We select its
  successor [Prometheus 2](https://arxiv.org/abs/2405.01535), an **EMNLP 2024**
  model, not an ICLR or NeurIPS publication. Its official code supports
  reference-free absolute grading, needed because WritingBench supplies criteria
  but no single gold response for each writing task.
- [Atla Selene Mini](https://arxiv.org/abs/2501.17195) supports rubric-based absolute
  scoring and reports evaluation across multiple judge benchmarks. Its Llama
  backbone differs from the Prometheus Mistral backbone, reducing shared-family
  dependence without guaranteeing independent errors. This is a **technical
  report**, not a verified NeurIPS/ICLR main-conference acceptance.
- [MT-Bench, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
  and [JudgeLM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7f8f73134e253845a8f82983219a8452-Abstract-Conference.html)
  document judge biases and motivate validation against human judgments.

These sources support the design choices; they do not establish that either
selected checkpoint is highly reliable on our proposals or fiction. No claim of
conference endorsement or guaranteed factual accuracy is made.

## Implemented protocol

Each judge sees the final answer, task, supplied evidence and criteria. The
policy's reasoning trace is excluded. Graph similarity remains a separate reward;
the writing judge does not assign it. Executable programming checks are unchanged.

Four native, reference-free grading calls score coherence, relevance, factual
grounding and feasibility from 1–5. Prometheus returns feedback followed by
`[RESULT] n`; Selene returns `**Reasoning:** ... **Result:** n`. Templates are
vendored from pinned official sources with attribution and license. No fabricated
reference answers or human labels are introduced.

The quality proxy is the mean of `(score - 1) / 4`. If any dimension is below 3,
the output reward is zero and the existing runtime blocks the structural bonus.
This threshold and aggregation are **our experimental choices**, not validated
universal rules. Unlike the old backend, this protocol does not extract and
verify individual claim quotations. Grounding is a rubric judgment, not a factual
certificate. The rubric distinguishes proposed methods and fiction from claims
of completed experiments; non-factual content without unsupported real-world
assertions receives grounding 5, meaning no grounding violation was detected.

Native chat templates are used without the policy's reasoning wrapper or duplicate
BOS tokens. Parsing rejects missing, ambiguous, out-of-range or truncated scores.
Each criterion permits one retry per invocation with a formatting reminder and
larger output budget. Persistent failure raises an error rather than supplying a
reward. Completed criteria survive interruption in the cache. Model revisions,
prompt identities, raw feedback, scores and token/time usage are recorded.

Four calls per answer increase judging cost; bounded retries can increase this
to eight. The training judge is also larger than the previous Phi model. Measure
CARC throughput and memory before committing to the full run.

## Verification and remaining work

- **17 local software tests passed:** native score parsing, retries, partial-cache
  recovery, interruption, failed generation, reward/structure gating, legacy
  dispatch, fixed-batch schedules, plan construction, source mismatch protection,
  and preflight failure handling. Generated fixture strings are software tests,
  not research observations. Test programs remain outside the repository.
- **Both pinned official tokenizers passed native-template checks**, including
  exact tokenization and one BOS token. Only tokenizer/configuration files were
  downloaded; model weights were not loaded.
- Preparation checks confirmed pinned revisions, cached manifest reuse and
  correct dispatch for historical v4 and current v5 CLI configurations. Python,
  JSON, launcher syntax and Git whitespace checks passed.
- Live CUDA/NF4 inference, memory use, judge accuracy and end-to-end writing
  training with these replacements remain untested. This Mac has no CUDA and
  insufficient free disk for both unquantized checkpoints.

The mandatory live preflight retains seven cases per judge: grounded summary,
fabricated result, nonsense, judge injection, hypothetical proposal, fiction and
off-topic output. Passing these 14 cases is necessary, not sufficient. Held-out
blinded human reviews and configured judge agreement/risk checks remain required.
No existing readiness gate was relaxed.

## CARC configuration

Launchers now select `human_topology_pilot_v5.json` and
`human_topology_full_v5.json`, using fresh v5 result roots. Old configurations and
results retain their original meanings. Fixed scheduled batches, token cap zero,
policy generation budgets, datasets and structural rewards are unchanged.

After these local changes are deliberately transferred, run `prepare` for the
new root to cache the new judge weights and freeze the manifest. Existing valid
public data and other model downloads remain reusable. Explicit environment
overrides of `RUN_ROOT` or `PAPER_CONFIG` still take precedence over launcher
defaults. Previous judge scores cannot be reused under the changed protocol.
