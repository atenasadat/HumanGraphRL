# TopGRPO

Public harness benchmark: **QuixBugs: A Multi-Lingual Program Repair Benchmark Set
Based on the Quixey Challenge** ([official source](https://github.com/jkoppel/QuixBugs)).
It validates execution infrastructure; the main training/evaluation datasets are
listed in the dataset report.

| Folder | Contents |
|---|---|
| `docs/` | Dated ideas, literature, reports, and historical documentation |
| `data/` | Raw participant data and external benchmark sources |
| `processed_data/` | Derived transcripts, priors, prompts, caches, and full-training outputs |
| `scripts/pilots/` | Pilot launcher, initial analyses, and legacy scripts |
| `scripts/experiment/` | Training, benchmark preparation, evaluation, dependencies, and Python environment |
| `pilots/` | Initial reports, figures, and pilot results |

Old top-level paths and compatibility symlinks have been removed. Development
tests and their launcher steps are removed. Executable benchmark test cases remain
part of the benchmark data and scoring pipeline.

The Python environment is `scripts/experiment/.venv`; dependencies are in
`scripts/experiment/requirements.txt` and
`scripts/experiment/thinkaloud_rl/requirements-paper.txt`. Set `TRAIN_VENV` when
using another CUDA environment on CARC. Preparation creates
`scripts/experiment/.venv-paper` by default.

From the repository root on CARC:

```bash
bash scripts/pilots/run_domain_pilot.sh prepare
bash scripts/pilots/run_domain_pilot.sh submit
bash scripts/pilots/run_domain_pilot.sh status
```

The current v3 pilot tests direct human-topology rewards with a separate final-output
reward: seven conditions, three contexts, one seed (21 runs), 150 updates each,
and balanced mixed-domain batches. It includes frozen-LLM and shuffled controls,
independent writing judges, diagnostic/final data separation, human review packets,
and automatic token/time/validation reports. See the
[implemented protocol and remaining evidence](reports/2026-11-02_human_topology_pilot_v3_codex.md).
Outputs go to `pilots/results/human_topology_pilot_v3/`. The live CUDA/judge preflight
and independent reviews remain required; no GPU training has been launched here.

For full training after inspecting the pilot:

```bash
bash scripts/experiment/run_paper_experiment.sh prepare
bash scripts/experiment/run_paper_experiment.sh submit
```

The full launcher uses `human_topology_full_v3.json` and refuses to start until
the pilot's automated and human-review checks are complete. Its training caps
remain provisional until the pilot is inspected. Full-training outputs go to
`processed_data/full_training_and_benchmark_results/`.

See [dataset details](reports/2026-11-02_datasets_and_pilot_codex.md),
[initial analysis inventory](../pilots/reports/2026-11-02_initial_analysis_inventory_codex.md),
and the [suite guide](../scripts/experiment/thinkaloud_rl/paper/README.md).
Historical reports retain their original paths and describe the layout at their dates.
