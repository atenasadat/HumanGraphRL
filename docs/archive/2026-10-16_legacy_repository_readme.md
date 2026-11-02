# Thought-grounded writing: GRPO on think-aloud reasoning graphs

Plan: `~/.claude/plans/purring-tinkering-biscuit.md`. Literature: `literature/00_synthesis.md`.

**Method.** Each writer gets their own LoRA adapter on a base model, trained with GRPO. The prompt is graph-free:
only the task description and which section to write. The writer's brainstorm reasoning graph is used **only** as
the reward (which ideas they endorsed, ruled out, or left as side-thoughts), so their thinking reaches the
model through the weights, never through the prompt.

**Baseline (later).** The base model given the same task prompt, no adapter. It is run only at the end, against
the trained results: `python -m tw.train.grpo.evaluate --config ... --base --split test`.

**Scope.**
- Writers: the 11 that pass the audit gate (P4–P10, P12, P15–P17). Excluded: P11, P14 (no ruled-out ideas), P13 (sparse).
- Data: one state per writer, the final brainstorm-phase reasoning-graph snapshot, copied verbatim. Source data is never modified.
- Splits are within a writer, by proposal section: train = full, objective, motivation, tasks, data
  (15 prompts with 3 instruction variants); val = significance (3); test = novelty, results (6).

## Pipeline (run from this folder with `PYTHONPATH=.`)
| Step | Command | Output |
|---|---|---|
| Cache brainstorm snapshots | `python -m tw.data.rg_sessions [P4 ...]` | `data/rg_cache/<session>/` |
| Idea fates + prompts | `python -m tw.data.dataset` | `data/ideas/`, `data/prompts.jsonl` |
| Audit + go/no-go gate | `python -m tw.audit` | `runs/audit/report.md` |
| 1. Base-model reference essays | `python -m tw.train.grpo.reference --config configs/grpo/train.yaml` | `data/reference/<model>.jsonl` |
| 2. Rarity weights | `python -m tw.ideas.rarity --config configs/grpo/train.yaml` | `data/reference/<model>.rarity.json` |
| 2b. Per-writer matching thresholds (P0 gate) | `python -m tw.ideas.calibrate --config configs/grpo/train.yaml` | `data/reference/<model>.thresholds.json` |
| 3. GRPO, one writer | `python -m tw.train.grpo.train --config configs/grpo/train.yaml --writer P7` | `runs/grpo/P7/<run>/` |
| 4. Test-set evaluation | `python -m tw.train.grpo.evaluate --config configs/grpo/train.yaml --run-name default` | `runs/eval/<run>/` |
| (verl export, optional) | `python -m tw.train.grpo.prepare_data --config configs/grpo/train.yaml` | `data/verl/<W>/` |

Any config key can be overridden: `--set grpo.steps=30 lora.r=16`. `configs/grpo/smoke.yaml` is a tiny
Qwen2.5-0.5B version of `train.yaml` for checking that the code runs.

**On CARC** (from `/home1/saghi/TopGRPO`):
```
bash carc/setup_env.sh          # once, on the login node: .venv (python/3.11.9 module) + model downloads
SMOKE=1 bash carc/submit.sh     # first: prep + 30 GRPO steps on P7 + eval
bash carc/submit.sh             # full: prep -> 11 writers in parallel -> eval + plots
SKIP_PREP=1 RUN_NAME=lr5e-5 EXTRA="grpo.lr=5e-5" bash carc/submit.sh   # another run, reusing the reference essays
```
Terminal output goes to `topgrpo_<stage>_<jobid>.log` in the project folder; the same lines go to `runs/grpo/<W>/<run>/train.log`.

**Training run output** (`runs/grpo/<W>/<run>/`): `config.yaml`; `train_log.jsonl` (per step: reward and
components, loss, KL, entropy, clip fraction, grad norm); `val_log.jsonl` (step 0 = base model, then every
`validation.every` steps); `val_samples.jsonl` (the validation essays); `adapter_best/` (best mean val reward);
`adapter_final/`; `plots/training.png` (reward, completion length, KL, entropy) and `plots/details.png`
(reward components, loss, grad norm, clip fraction). After eval, `runs/grpo/summary_<run>.png` overlays all writers.

**Environments.**
- Local data work: conda `base`. Local model code: conda `stepaudio2` (torch 2.10, transformers 4.49, peft 0.14).
- CARC: `.venv` from `carc/setup_env.sh` (torch 2.8, transformers 4.51-4.55 for Qwen3, peft, matplotlib).

## Modules
- `tw/data/phases.py`: brainstorm window from `CHI2027 Analysis/annotations/P*.json`, aligned via snapshot `saved_clock`.
- `tw/data/clean.py`: duplicate-ID repair across snapshots; read-aloud tags, used as scoring metadata only.
- `tw/ideas/fates.py`: endorsed / ruled_out / neutral from the last brainstorm snapshot (`configs/fates.json`).
- `tw/gen/task_prompt.py`: graph-free prompts and per-section word budgets.
- `tw/rewards/composite.py`: R = (selection + creativity + ownership + synthesis) × (1 − copy) − ruled_out − padding − length (`configs/rewards.json`).
  Creativity is measured against the base model's essays for the same task and section, and selection is weighted by idea rarity.
- `tw/ideas/entail.py`: `embed` backend (smoke) and `minicheck` backend (real runs; it uses an encoder for sentence similarity).
- `tw/train/grpo/`: `policy.py` (model, LoRA, rollouts), `data.py` (configs, splits), `reward_fn.py` (scorer; also verl's `compute_score`),
  `reference.py`, `train.py`, `evaluate.py`, `plot.py`, `prepare_data.py` (verl).
- `tw/eval/metrics.py`: held-out selection F1 and cross-writer distinctness (computed with a different encoder from the reward's).


## Idea two: Top-GRPO-Form (`tw/form/`, `configs/form/`, `carc/form_*.sh`)

Trains the **form** of brainstorming, not only the essay. The policy writes a numbered brainstorm trace
(`[n7] <- n3 | revisits n2: ...`, see `tw/form/trace.py`) and then the essay. A deterministic parser turns the trace
into a graph with the human schema (edges come from node references, never from labels), and the reward compares its
topology with the 14 human graphs. Design and literature: `literature/06_form_of_reasoning.md`.

Reward (`tw/form/reward.py`): form (each topology feature inside the human IQR band; non-monotone) + memory
(reach-back histogram matches the human one; fitted access model d = 1.08) + content (idea one's essay reward)
− rumination (short loops, duplicate thoughts, references to forgotten/unknown thoughts) − format, plus a batch
diversity bonus so the 8 rollouts do not collapse to one shape. Forgetting environment (`memory.env: true`,
`tw/form/env.py`): the brainstorm is generated in chunks; between chunks, thoughts outside the last 4 whose ACT-R
activation has decayed are removed from the context, so reach-backs emerge from bounded memory.

| Step | Command | Output |
|---|---|---|
| P0 human profile + memory fit (Mac, no GPU) | `python -m tw.form.profile` | `data/form/human_profile.json` |
| prep (shared with idea one) | `sbatch carc/job.sh prep` if `data/reference/qwen3-1.7b.jsonl` is missing | |
| P1 base-model form vs human | `python -m tw.form.evaluate --config configs/form/train.yaml --base` | `runs/form_eval/base/` |
| P2 GRPO, reward only | `python -m tw.form.train --config configs/form/train.yaml --writer P7` | `runs/form/P7/<run>/` |
| P3 GRPO + forgetting environment | `... --run-name forget --set memory.env=true` | `runs/form/P7/forget/` |
| eval + plots | `python -m tw.form.evaluate --config configs/form/train.yaml --run-name <run>`; `python -m tw.form.plot --run-name <run>` | `runs/form_eval/<run>/`, `plots/` |

CARC: `SMOKE=1 bash carc/form_submit.sh` (base eval + 30 steps on P7 + eval), `bash carc/form_submit.sh` (all
writers), `RUN_NAME=forget EXTRA="memory.env=true" bash carc/form_submit.sh` (P3). Logs: `topgrpo_form_*.log` and
`runs/form/<W>/<run>/train.log`. Plots per run: `training.png` (reward, length, KL, entropy), `topology.png` (each
feature over training with the human IQR band), `rewards.png`; all writers: `runs/form/summary_<run>.png`.

**Gate (P1).** `runs/form_eval/base/test.summary.json` reports the base model's parse rate and a human-vs-model
classifier AUC on the topology vector (0.5 = indistinguishable). GRPO only amplifies behaviour that already occurs,
so if the parse rate is below ~0.3 the next step is SFT priming on synthetic traces with human statistics, not more RL.
The prompt prefills `<brainstorm>\n[n1] goal:` so the model starts inside the format (`memory.prefill`).

Files to copy to CARC for idea two: `tw/`, `configs/`, `carc/`, `data/prompts.jsonl`, `data/ideas/`, `data/form/`.

## Training algorithm (`tw/train/grpo/train.py`)
Per step: 4 train prompts × G=8 rollouts. Advantages are group-relative (std-normalized; `adv_norm: none` gives
Dr. GRPO), and groups with zero reward variance are skipped. The loss is a clipped surrogate with clip-higher
(0.2/0.28) plus a k3 KL term (0.005) to the base model, obtained by disabling the adapter, so there is no second
copy of the weights. Aggregation is Dr. GRPO: token sum over a constant normalizer. The run aborts if mean token
entropy falls below `min_entropy`.

## Run 1 post-mortem (2026-10-16, `runs/eval/default`)
- Ownership 0.57, selection 0.04, creativity 0.13, ruled_out 0.01, copy 0, length penalty 0.53 (essays 3-5x the
  section budget). P4/P5/P8 had **zero** content signal from step 0: at the global 0.55 similarity threshold none of
  their (disfluent) utterances could match any sentence, so GRPO optimized the only non-zero term (length) and the
  essays lost all connection to the writer. The null similarity level differs per writer (95th pct 0.38-0.65).
- Fixes now in the code: per-writer thresholds at a fixed 5% false-positive rate (`tw/ideas/calibrate.py`, run in
  prep); the word budget is in every prompt ask (`prompts.jsonl` regenerated: re-copy it to CARC); the trainer aborts
  when selection and ownership are both 0 at step 0 (`reward.require_signal`). Base-model baseline: `sbatch carc/job.sh base`.

## Status (2026-10-14)
- **Audit:** 11 writers pass the gate (needs 10) → GO. Ruled-out counts are low (1–6 per writer).
- **Verified locally:** prompt splits (15/3/6 per writer for all 11 writers), config inheritance and overrides,
  reference generation, rarity, reward sanity checks (endorsed ideas score high, ruled-out negative, verbatim copying
  scores below rephrasing, think blocks stripped), verl export, validation at step 0, and test evaluation.
- **Code check on a tiny random Qwen3 with the CARC package versions:** prep, training (KL, PPO epochs, gradient
  checkpointing, both advantage modes), validation, adapter saving, evaluation and all plots run end to end.
  The real 1.7B model has not been trained yet. Run `SMOKE=1` first.
- **Still open:**
  - fate-rule validation (κ vs 2 annotators);
  - read-aloud tagger precision;
  - the essay-quality model (validated on PERSUADE 2.0), currently only a length guard;
  - checking the MiniCheck API and the verl config keys on CARC.
