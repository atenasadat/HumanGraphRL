# Constrained writing-judge generation

Date: 2026-11-02. Author: Codex.

The CARC preflight exposed free-form judge responses with missing fields or
`invented_result` used as a claim verdict. Prompt retries alone were insufficient.
Both training and evaluation writing judges now constrain allowed next tokens
using one explicit JSON Schema: all four fields are required, scores are integer
choices 0–4, flag and verdict choices are distinct, and extra fields are forbidden.
Policy sampling remains unconstrained.

The implementation uses LM Format Enforcer 0.11.3's public core token API. Its
released Transformers adapter imports an obsolete symbol under Transformers 5,
so the local bridge avoids that adapter. Tokenizer preprocessing follows the
upstream word-boundary convention. Source:
https://github.com/noamgat/lm-format-enforcer

Completed judge responses undergo independent JSON Schema validation plus the
existing evidence checks. Duplicate keys, nonfinite numbers, missing fields,
unknown labels, and empty evidence on a supported claim are rejected. No missing
judgment is filled in or turned into a zero reward. Prior ad hoc risk-label
normalization is superseded by exact constrained verdict choices.

Each assessment has at most two generation attempts. Reaching the token limit
invalidates that response and increases the retry allowance from 1,536 to 3,072
tokens. Semantic validation errors can also trigger the bounded retry. Every
completed attempt saves raw output, token counts, budget, truncation status and
precise validation errors. Failed-attempt history persists across re-submissions.
Generation exceptions are recorded and stop the stage. Protocol
`evidence_aware_constrained_v2` and schema hashes separate these cache entries
from previous unconstrained judgments.

Training catches persistent judge failures, restores the incomplete batch's RNG,
and checkpoints the last committed weights and optimizer state. Pending validation
is recorded and replayed before training resumes. Completed evaluation samples,
judge cache entries and suite stages retain their existing resume behavior.
An incomplete training batch may be regenerated; it is never partially optimized.

## Verification

Temporary verification scripts live under `/private/tmp`; no repository test
suite was added.

- Actual tiny CPU Qwen2 `generate()` with the production sampling function and
  token filter produced schema-valid JSON. The filter blocked premature EOS,
  missing required fields, a score of 9, and an invalid `invented_result` verdict.
- Both actual public judge tokenizers accepted a complete valid assessment and
  disallowed premature EOS: Mistral revision
  `c170c708c41dac9275d15a8fff4eca08d52bab71` and Phi revision
  `2fe192450127e6a83f7441aef6e3ca586c338b77`. Only tokenizer files were downloaded.
- Mocked judge orchestration verified truncation retry budgets, strict validation,
  cache reuse, attempt diagnostics and persistent failure instead of a reward.
- Actual tiny Qwen2 LoRA/Adam training with an injected persistent judge failure
  resumed to bit-identical adapter weights versus uninterrupted training, even
  with periodic checkpointing set to every ten updates.
- Injected periodic and final validation failures saved pending validation and
  resumed it successfully, again retaining identical final adapter weights.

Full Phi/Mistral inference and CUDA execution were not performed locally. CARC's
mandatory live judge preflight remains required. Format constraints do not prove
factual accuracy, adequate claim coverage or resistance to evaluator manipulation.
Token limits, generation infrastructure errors, evidence-validation failures and
incorrect judgments can still stop the pilot or fail its scientific checks.

## Update CARC

Use a fresh root because the source fingerprint and pilot config changed. This
does not redownload already cached data or model weights. Stop any old allocation
before replacing its source files.

```bash
cd ~/HumanGraphRL
git pull --ff-only
source humangraphrl/bin/activate
export TRAIN_VENV="$PWD/humangraphrl"
export PAPER_VENV="$PWD/humangraphrl"
export RUN_ROOT="$PWD/pilots/results/human_topology_pilot_v3_constrained"
bash scripts/pilots/run_domain_pilot.sh prepare
bash scripts/pilots/run_domain_pilot.sh submit
```

Preparation installs `lm-format-enforcer==0.11.3` and `jsonschema==4.23.0` and
checks their imports/schema before GPU allocation. Subsequent retries use the
same root and only `submit`.
