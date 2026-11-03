# Judge claim kinds and two-stage token budgets — 2026-11-03

Fixes the two blockers in the
[CARC smoke-test report](2026-11-03_carc_smoke_test_codex.md): the writing judge
rejected valid proposals and fiction, and reasoning truncation (37.5%) exceeded
the 25% gate. Nothing was pushed or submitted. CUDA/NF4, the Mistral evaluation
judge and writing/mixed training are still unverified.

## 1. Judge: scoring rule, not judge quality

The judge had rated the proposal 4/4/4/4. The veto came from `result()`: every
listed sentence needed a verdict, `supported` required a verbatim quote from the
evidence, and any non-supported claim or any dimension below 2 set reward to 0.
Plans and requested fiction can never satisfy that. Most WritingBench answers
would score 0, which removes the writing reward signal.

Changes in `paper/output_quality.py` (protocol `evidence_aware_constrained_v5_claim_kinds`):

| Rule | Before | After |
|---|---|---|
| Claim types | none | `kind`: `empirical_result`, `citation`, `factual_background` (checked); `plan_or_hypothesis`, `fiction`, `opinion_or_recommendation`, `other_non_factual` (never need evidence) |
| Veto | any flag, any non-supported claim, any score < 2 | severe flags (`nonsense`, `invented_result`, `invented_citation`, `contradiction`, `prompt_injection`); unsupported results/citations; any contradicted claim; coherence or relevance < 2 |
| Unsupported background facts | veto | reward × (1 − 0.5 × unsupported fraction) |
| Grounding when nothing checkable is asserted | counted (fiction scored 0) | excluded from the quality average |
| Misquoted claim | judgment fails | fails, unless the answer is already vetoed by coherence, relevance or a severe flag; then kept as `discarded_claims` for audit |
| Judge cache key | protocol + schema | also a hash of the prompt template, so prompt edits never reuse old judgments |

The claim is quoted before its `kind` is chosen, and kinds are defined by tense.
Phi otherwise labelled "we would…" as a completed result.
`paper/preflight.py` adds an `off_topic` fixture: code returned for an investment
analysis, modelled on WritingBench #92, must be rejected on relevance.

**Live check:** Phi-3.5-mini (pinned `2fe1924`), Mac MPS bf16, greedy,
production preflight fixtures, training phase. **7/7 pass**:

| Fixture | Expected | Reward | Decided by |
|---|---|---:|---|
| grounded | pass | 1.0 | supported background fact |
| fabricated_result | reject | 0 | unsupported `empirical_result` (coherence/relevance were 4) |
| nonsense | reject | 0 | `nonsense` flag, coherence 0 |
| judge_injection | reject | 0 | relevance 0, `nonsense` |
| hypothetical_proposal | pass | 1.0 | claims typed `plan_or_hypothesis` |
| explicit_fiction | pass | 1.0 | claims typed `fiction` |
| off_topic | reject | 0 | relevance 0 |

Unit checks of the rule (8 synthetic judgments, misquote handling, rejection of
outputs without `kind`) and the constrained decoder (only valid `kind` values are
reachable) also pass. **Not checked:** the Mistral-7B evaluation judge (too large
for this Mac), CUDA numerics, real WritingBench outputs, and human calibration,
which remains a pilot gate.

## 2. Token budgets: the old data was capped

Earlier traces were cut at 256 reasoning / 768 answer tokens: 8/8 writing
reasoning traces and 3/8 writing answers hit the cap. Those lengths can't size a
budget. Remeasured with Qwen2.5-1.5B (pilot policy, `989aa79`), two-stage, T=0.7,
caps 2048/4096, 8 MBPP and 8 WritingBench training prompts (including the 8 calibration
prompts). No response was truncated:

| Domain | Reasoning tokens (sorted) | Answer tokens (sorted) |
|---|---|---|
| Programming | 73, 86, 149, 151, 152, 209, 215, 234 | 19, 61, 66, 85, 86, 90, 103, 126 |
| Writing | 146, 178, 395, 398, 465, 783, 1038, 1512 | 146, 219, 245, 313, 406, 507, 662, 663 |

**Configs:** `reasoning_fraction` 0.25 → 0.5. Pilot `max_new_tokens` 1024 → 2048,
giving 1024 reasoning + 1024 answer tokens; one of the 16 measured traces (6%)
exceeds this. Full stays at 2048 (1024/1024 training; 2048/2048 at the 4096-token
evaluation). Generation stops at EOS, so larger caps cost only the long tail.
Sixteen samples from a 1.5B model don't establish 7B lengths. Structural
calibration now records p50/p95/max and truncation counts per stage in
`llm_*.coverage.json`. Check the 7B values there before the full run; the 25%
learning-preflight truncation gate is unchanged.

## Consequences

The code fingerprint and the pilot config changed, so use a fresh pilot run
folder. Earlier judge-cache entries won't be reused. Test scripts and outputs are
outside Git in the session scratchpad.
