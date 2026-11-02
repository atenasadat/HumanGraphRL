# Think-aloud writing RL data tools

These tools implement prerequisites for `rl_ideas_thinkaloud.md`. They do not train an LLM or assign validated process-quality rewards.

For the versioned training-data preparation, source checks, exclusion rules and blocking preflight, see [TRAINING_READINESS.md](TRAINING_READINESS.md). The current preparation output is `runs/thinkaloud_training_v2`; `v1` was the initial diagnostic build and is superseded. Neither is approved reward supervision.

The all-writer audit is [REPORT.md](../../runs/thinkaloud_rl_inventory/REPORT.md). Original study folders were relocated by the user to `~/Desktop/thougthgraph_studyData`; `paths.py` resolves those paths without editing the original manifests. Older derivative files on the Desktop retain historical provenance paths.

```sh
python3 -m experiments.thinkaloud_rl.inventory
python3 -m unittest experiments.thinkaloud_rl.test_core experiments.thinkaloud_rl.test_inventory -v
```

The inventory scans locally available logs, including P4's 12.96 GB log, and skips cloud-only event-log contents. For this run, the four compact P5–P8 `session_state.json` files were selectively read and cached under `runs/thinkaloud_rl_inventory/selective_state_cache` (9,348,310 bytes total). The cache is not a new human dataset; each source path and content hash is retained in the inventory. No large cloud log was downloaded. The default inventory reuses these caches.

To refresh the compact session-state/curated-response checks without repeating the completed event scans:

```sh
python3 -m experiments.thinkaloud_rl.inventory --refresh-state-sources
```

Outputs:

- `inventory.json`: all 14 writers; verified transcript copies, task-boundary flags, process annotation coverage, source paths, media/graph file availability, event metadata, document-state statistics and endpoint comparisons.
- `physical_action_states/P*.snapshots.jsonl`: all retained physical-action text snapshots in their original node order, including repeated values. Preserve this order as source provenance, not as validated chronology. Node timestamp is not necessarily text-capture time.
- `local/*.states.jsonl`: distinct document text values recovered from event logs, retaining the first occurrence of each value. These files are an inventory, not a complete sequence of transitions.
- `endpoint_differences.json`: the P10/P12 curated-endpoint discrepancies. Original text is never overwritten with the curated variant.
- `tests.log`: execution of the mathematical and extraction checks.

The source `nodes.physical_action[].full_code_snapshot` contains document text. Generated labels such as `move_type`, confidence and Toulmin analysis are not used as quality labels. No source node is assigned an expert/good-step label simply because it came from a participant.

The original three-writer probe remains available as `python3 -m experiments.thinkaloud_rl.audit`; it now resolves relocated inputs and writes new derivatives to `runs/thinkaloud_rl_audit` instead of recreating the moved source-result folder. Use the all-writer inventory for current conclusions.

Current limitations: time alignment is unvalidated; saved node order includes reversals and duplicate timestamps; later merges can attach newer document text to an older node; transcripts mix participants, researchers and AI quotations; proposal endpoints can be incomplete; existing Goal Analysis coverage is sparse and uneven. These checks establish data availability, not training efficacy.

## Multi-domain extension

See [MULTIDOMAIN.md](MULTIDOMAIN.md) for writing ingestion, provisional cognitive
priors, corrected LoRA GRPO training, invariance limits, cross-domain evaluation,
and local/SLURM commands. Earlier pilot accuracy, convergence, and safety-net
claims are not validated by this extension.
