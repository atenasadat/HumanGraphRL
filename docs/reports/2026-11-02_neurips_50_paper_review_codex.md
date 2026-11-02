# Human reasoning topology: code audit and 50-paper NeurIPS review

Date: 2026-11-02  
Author: Codex

## Scope and source access

This is a focused full-text methodological review of **50 unique accepted NeurIPS papers**:
48 from the official 2025 proceedings (including the datasets/benchmarks track) and
two listed as accepted in 2026. The table below identifies every paper, source PDF,
page count, and its relevance. Complete PDFs and extracted text were obtained for all
50 papers, totaling 1,623 PDF pages. The reading focused on methods, training setup,
evaluation, ablations, and relevant appendix passages. This is **not a claim of
cover-to-cover reading of 1,623 pages**, proof verification, experiment reproduction,
or an exhaustive survey. Downloading and extracting a PDF alone was not counted as
a substantive review; each included paper also has a specific methodological note.

The 2025 PDFs are publisher versions. The tested 2026 OpenReview PDF endpoints
returned HTTP 403. The two 2026 papers were matched by exact title to the official
accepted-paper list, then read as public arXiv versions; camera-ready equivalence
has not been established. One unsuccessful 2025 extraction (ThinkBench) is excluded.
The official acceptance listing is https://neurips.cc/Downloads/2026.

Sources are reproducibly identified in the accompanying JSON manifest, including
PDF URLs and SHA-256 hashes. Downloaded PDFs/text are in the temporary review
workspace, outside the repository. Text extraction can omit diagram content; no
conclusions here depend on measuring graphs from extracted figure text.

## Direct answers from the current code

### Writing quality is not the training reward

In `scripts/experiment/thinkaloud_rl/paper/scoring.py`, constrained-writing training
returns a weighted sum of four keyword-group matches (70%) and length compliance
(30%). These are task-format proxies. The WritingBench evaluator returns no reward
until separate judgments exist. In `paper/judge.py`, a judge can grade held-out final
answers against supplied criteria, but its scores are not fed into `paper/train.py`.
The default `judge_model` is null. Thus an independent writing-quality score is
currently neither measured automatically nor used to train the policy.

Using a quality judge in training is possible, but would be an explicit objective
change. Use training prompts for training rewards and independent judges/human
ratings on held-out prompts for evaluation. Reusing one judge score for both
optimization and final assessment creates a reward-hacking risk. For the user's
stated goal, topology can be the main training signal while quality is separately
assessed as a guardrail/outcome. Quality need not be optimized through keywords.

### The current code does not implement direct human-topology learning

Two different uses of keywords exist. The writing outcome proxy rewards topic and
format words. Separately, `reasoning_shaper.py` uses regex patterns to assign four
labels: explore, evaluate, commit and backtrack; unmatched steps default to evaluate.
Those labels produce phase distributions and adjacent-move transition tables.
This is a coarse sequence-style representation. It does not represent semantic
step dependencies, explicit branching, revisits to specific ideas, or a full
reasoning graph; labels remain unvalidated.

More fundamentally, the human term is applied as a **causal prefix baseline**:

`advantage_at_token = outcome_advantage - scale * Phi(prefix_before_token)`

For on-policy sampling, a fixed prior, detached causal prefix scores and the
implemented fixed token divisor, its added raw score-function gradient has
conditional expectation zero:

`E[Phi(s) * grad log pi(a | s) | s] = Phi(s) * grad sum_a pi(a | s) = 0`.

Consequently it changes the gradient estimator/noise rather than adding a direct
expected policy-gradient incentive to match human structure. Actual finite-sample
trajectories and Adam/clipping can still change behavior; this is not a statement
that every trained model must be identical. Variance reduction itself also needs
measurement and is not guaranteed. Zero-terminal potential shaping preserves the
trajectory's total return; its name alone does not establish structural imitation.
The same design intent is stated in the suite's existing control-variate documentation.

That makes the current pipeline suitable for a hypothesis about a human-informed
control variate, but insufficient to establish the requested learning of human
reasoning topology. Earlier descriptions that implied it already accomplished
that goal were too strong.

### A goal-aligned candidate objective, not yet implemented

A direct structure-learning experiment would give a generated trace a score based
on the similarity between its independently extracted reasoning graph and a human
reference distribution, then optimize that score as a rollout-level reward:

`J(theta) = E[S(G(generated_reasoning), human_reference_distribution)]`.

This deliberately changes the objective. It must not be presented as policy-invariant
reward shaping. Graph extraction and its human reference need independent validation.
Dependencies, branching, revisits and temporal organization need operational definitions;
embedding similarity alone does not prove functional equivalence to human thought.
Quality/correctness should be assessed independently and can be a constraint or an
additional declared objective. This is a proposed experiment, not a code change or
an efficacy claim.

## What the papers imply about 150 updates

There is no defensible universal update-count threshold for a NeurIPS-quality
contribution. Published recipes differ in prompt batch size, rollouts, response
length, mini-batch updates, objectives, supervision and model size. Specific examples:

- Open-source DAPO (paper 44) generates 512 x 16 = 8,192 responses per rollout batch,
  then performs 16 gradient updates. A rollout step and a gradient update are different.
- ProRL (paper 6) studies prolonged training beyond 2,000 steps on 136K examples.
- GURU (paper 17) includes a 200-RL-step cross-domain comparison with a 512-prompt
  training batch and 64-prompt gradient-update batch.
- Entropy Minimization (paper 38) reports a 40-step EM-FT setup and 100-step RL setups
  with much larger batches than our pilot. Its 5–15-step figure refers to inference-time
  token-logit optimization, not to RL training of model parameters.

Our pilot: 150 updates x 4 prompts x 4 responses = **2,400 generated responses/run**,
600 prompt selections, and an upper generation cap of **2,457,600 tokens/run**
(excluding validation, testing, audits and prompt tokens). Actual generated tokens
will usually be lower. This is a diagnostic workload, not evidence of convergence
or publishability. It is also one seed on a 1.5B model. Increasing only the update
count cannot fix a misaligned objective or an unvalidated structure measure.

The immediate priority is to freeze the scientific hypothesis. For direct topology
learning, correct the objective and representation first. Then calibrate training
horizons using validation learning curves and matched generated-token/compute
budgets. A 150/300/600-update sweep is a possible engineering calibration, **not**
a literature-mandated standard or an instruction to launch runs now. If continuing
improvement or instability remains at a cap, extend the horizon; if curves plateau,
report the evidence and actual cost. Keep final benchmark tests out of tuning.

A persuasive full study would test topology transfer and outcome quality separately,
use repeated training seeds, hold out human participants as well as benchmark tasks,
validate graph extraction, compare base/outcome-only/human-structure controls under
matched budgets, and include shuffled/degree- or count-matched controls. Add a
label-free graph-reward baseline such as SARL. Check whether effects survive
reasoning-length matching, cue-word paraphrases, and independent structure labeling.
These are recommendations inferred from the combined review, not guarantees of acceptance.

## Closest structural work

SARL (2026, paper 51) directly uses a graph-level structure reward for RL, rather than
an action-independent prefix baseline. Its steps are embedded, clustered within each
response, connected by transitions, and scored for graph organization. This is close
to the proposed structural-learning direction, so novelty cannot rest on saying
"reward reasoning topology" alone. The user's collected human trajectories could
support a distinct hypothesis about human-derived, domain-specific topology and
transfer, but that distinction must be demonstrated, not assumed.

Topology of Reasoning (2025, paper 1) also motivates graph-level analysis, while
studying graph properties does not by itself establish a successful training objective.

## Paper-by-paper review

The notes below distinguish what each paper contributes to this methodological audit.
They do not assert that every work uses the same RL algorithm or is a full-recipe
baseline suitable for direct numerical comparison.

| # | Year | Full paper | PDF pages | Relevance and caution |
|---|---|---|---:|---|
| 1 | 2025 | [Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties ](https://proceedings.neurips.cc/paper_files/paper/2025/file/147521db4d049c75cf18d9ff96bf68f5-Paper-Conference.pdf) | 32 | Uses hidden-state reasoning graphs to study cyclicity, diameter and small-world structure. Evidence about graph properties does not establish that matching four surface-language labels teaches human reasoning. |
| 2 | 2025 | [S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models ](https://proceedings.neurips.cc/paper_files/paper/2025/file/44f09d1973a88529b40029e6c3693ded-Paper-Conference.pdf) | 27 | Early-exit reward design controls redundant reasoning; on-policy batches contain 128 prompts times eight responses. Report both correctness and generation length. |
| 3 | 2025 | [Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/565f995643da6329cec701f26f8579f5-Paper-Conference.pdf) | 39 | Uses consistency and volatility of intermediate-state representations as intrinsic rewards. Compare structural signals against representation-based, label-free alternatives. |
| 4 | 2025 | [Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/82a34882f560db982d1257b5af461605-Paper-Conference.pdf) | 37 | Gradient-space data diversity is studied across hundreds of runs. Dataset diversity and generalization need their own evidence beyond prompt count. |
| 5 | 2025 | [Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/e01d556690c08fed8bd52d786dcef19f-Paper-Conference.pdf) | 22 | Adaptive fast/slow reasoning uses explicit tags, a cold-start stage and RL. Isolate explicit process supervision from the effect of subsequent RL. |
| 6 | 2025 | [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models ](https://proceedings.neurips.cc/paper_files/paper/2025/file/1a22b912945fb7c0bdd079e792b31b6f-Paper-Conference.pdf) | 34 | Prolonged RL uses 136K examples and more than 2,000 training steps, with training-stability interventions. This is one recipe, not a universal required update count. |
| 7 | 2025 | [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? ](https://proceedings.neurips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf) | 36 | Uses broad pass@k evaluations to distinguish probability concentration from capability expansion. Include base-model sampling baselines before claiming new reasoning capacity. |
| 8 | 2025 | [Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/6058d0c628a03fd95dfe5c72cbdf9e64-Paper-Conference.pdf) | 38 | SPARKLE separates plan following, subproblem solving and other reasoning components. Final-answer accuracy alone does not establish a change in reasoning process. |
| 9 | 2025 | [Beyond Verifiable Rewards: Scaling Reinforcement Learning in Language Models to Unverifiable Data ](https://proceedings.neurips.cc/paper_files/paper/2025/file/6bd67a424dc59481e1e5a5061ffc8dfe-Paper-Conference.pdf) | 28 | JEPO addresses unverifiable, long-form targets; uses 64 prompts and four generations per update. Difficulty of verification does not imply keywords are an adequate quality reward. |
| 10 | 2025 | [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution ](https://proceedings.neurips.cc/paper_files/paper/2025/file/7107d4d2e837bde2171c6b71b5bde954-Paper-Conference.pdf) | 26 | SWE-RL trains 1,600 steps with 32 problems and 16 rollouts per batch. Real software engineering requires execution-based, external evaluation even when training rewards are cheaper. |
| 11 | 2025 | [Progress Reward Model for Reinforcement Learning via Large Language Models ](https://proceedings.neurips.cc/paper_files/paper/2025/file/91b047c5f5bd41ef56bfaf4ad0bd19e3-Paper-Conference.pdf) | 28 | Progress-based dense reward combines LLM task planning and RL; reports five seeds. Reward construction, planning supervision and evaluation should be separated. |
| 12 | 2025 | [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Paper-Conference.pdf) | 35 | Studies high-entropy tokens and token-selective policy gradients, including substantial within-batch optimization. State the unit of every reported training step. |
| 13 | 2025 | [Reinforcement Learning for Reasoning in Large Language Models with One Training Example ](https://proceedings.neurips.cc/paper_files/paper/2025/file/b1ea3f93167c55f3fc2999b68a170ff3-Paper-Conference.pdf) | 44 | One-example RLVR can still involve approximately 1,400 or more training steps. Few unique training questions do not imply few generated responses or little optimization. |
| 14 | 2025 | [Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts ](https://proceedings.neurips.cc/paper_files/paper/2025/file/b42fb82fc3ffb88b872c4714f093875d-Paper-Conference.pdf) | 26 | Selective rollout sampling addresses uninformative zero-variance groups. Track rejected samples and rollout cost; select checkpoints using a separate validation split. |
| 15 | 2025 | [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/b8026fe01f5f8f7a80a1a9bf01d08cad-Paper-Conference.pdf) | 28 | Separates positive and negative reinforcement and examines pass@k. A single aggregate reward improvement cannot explain which gradient component produced a change. |
| 16 | 2025 | [Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/be91eb86eb74efc055cff83e953f86ce-Paper-Conference.pdf) | 26 | Shows process-reward exploitation and compares reward-credit aggregation strategies. A plausible-looking intermediate reasoning score is not automatically a reliable training signal. |
| 17 | 2025 | [Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective ](https://proceedings.neurips.cc/paper_files/paper/2025/file/c2f71567cd53464161cab3336e8fc865-Paper-Datasets_and_Benchmarks_Track.pdf) | 21 | GURU explicitly compares single-domain and mixed-domain RL, including a 200-step comparison with 512-prompt training batches. Transfer effects depend on source domain and difficulty. |
| 18 | 2025 | [GVPO: Group Variance Policy Optimization for Large Language Model Post-Training ](https://proceedings.neurips.cc/paper_files/paper/2025/file/f21a76d688be0553c943a6b6c1d4bb1f-Paper-Conference.pdf) | 23 | GVPO changes the policy-optimization objective to address stability. Name estimator, normalization, KL and optimization choices rather than grouping all variants under GRPO. |
| 19 | 2025 | [GPO: Learning from Critical Steps to Improve LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/01cdd2dd02d9d2db9349bd3c019a88da-Paper-Conference.pdf) | 35 | Uses step-aware signals to emphasize critical reasoning steps and reports averages over three seeds. A structure claim needs evidence at the process level. |
| 20 | 2025 | [Enigmata:  Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles ](https://proceedings.neurips.cc/paper_files/paper/2025/file/058460c719f190500568b341bd06875f-Paper-Conference.pdf) | 49 | Verifiable synthetic puzzles compare staged and mixed training under matched total steps. The paper includes a 370-step mixed-training setup and separate matched-budget comparisons. |
| 21 | 2025 | [Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector ](https://proceedings.neurips.cc/paper_files/paper/2025/file/095c7a06f229943c46bd0b519edf20f9-Paper-Conference.pdf) | 45 | Judge debiasing addresses systematic evaluation errors. An independent model judge needs calibration; it is not automatically ground truth for writing quality. |
| 22 | 2025 | [Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/12867c8f1888d97d4e9f95f03c0f3ec9-Paper-Conference.pdf) | 40 | Studies information patterns over reasoning trajectories. Reasoning-associated words alone are insufficient evidence about the functional role of a step. |
| 23 | 2025 | [ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs ](https://proceedings.neurips.cc/paper_files/paper/2025/file/26618fb384d3873b8ef6ab292a69095b-Paper-Conference.pdf) | 39 | Trajectory-aware PRMs combine step- and trajectory-level supervision. Distinguish learned process-quality assessment from handcrafted move-frequency matching. |
| 24 | 2025 | [AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play ](https://proceedings.neurips.cc/paper_files/paper/2025/file/30b00503cffe6acf25876fb1690a7357-Paper-Conference.pdf) | 42 | Self-play decomposes reasoning and searching roles and controls training-data overlap. Multi-stage methods need stage-specific budgets and contamination checks. |
| 25 | 2025 | [Eliciting Reasoning in Language Models with Cognitive Tools ](https://proceedings.neurips.cc/paper_files/paper/2025/file/3b911f3e7fc3586e5a5fff70c0da5b72-Paper-Conference.pdf) | 32 | Uses explicit cognitive operations such as understanding, recalling and backtracking through inference-time tools. A cognitive-inspired intervention need not be an RL training method. |
| 26 | 2025 | [Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/3e22bea3b170f4c2aebb9c48d98ae64d-Paper-Conference.pdf) | 27 | Studies thinking-budget allocation and self-improvement. Generation budget is an experimental factor; more tokens cannot silently substitute for better reasoning organization. |
| 27 | 2025 | [General-Reasoner: Advancing LLM Reasoning Across All Domains ](https://proceedings.neurips.cc/paper_files/paper/2025/file/51c52cbb4b3faa891ca0621d13efe1be-Paper-Conference.pdf) | 23 | General-domain RL uses a learned verifier and reports verifier exploitation that required mitigation. Outcome judging can support training, but needs hacking checks and held-out evaluation. |
| 28 | 2025 | [Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/54e1381d0c0598127b90af4c940fd3d9-Paper-Conference.pdf) | 29 | Directly contrasts process and outcome rewards for agentic retrieval. It is relevant to distinguishing process learning from merely measuring final-answer success. |
| 29 | 2025 | [ReDit: Reward Dithering for Improved LLM Policy Optimization ](https://proceedings.neurips.cc/paper_files/paper/2025/file/58caefe277fc1d48173bc59f56c9b86a-Paper-Conference.pdf) | 34 | Reward dithering changes optimization behavior under discrete outcomes. Gradient stability and reward statistics should be monitored, not assumed from update count. |
| 30 | 2025 | [Inference-Time Reward Hacking in Large Language Models ](https://proceedings.neurips.cc/paper_files/paper/2025/file/590a0cc0306c1c63e2d66a51a407718f-Paper-Conference.pdf) | 41 | Investigates exploitation of imperfect reward models during inference-time optimization. Keep an independent outcome assessment outside the reward being optimized. |
| 31 | 2025 | [Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization ](https://proceedings.neurips.cc/paper_files/paper/2025/file/614f8eba720cfc7ff00274bd64fb0a3f-Paper-Conference.pdf) | 28 | EMPO supplies an explicit entropy-minimization objective without external labels; a reported math setup uses 300 steps. A label-free method still requires an actual objective. |
| 32 | 2025 | [DAPO : Improving Multi-Step Reasoning Abilities of Large Language Models with Direct Advantage-Based Policy Optimization ](https://proceedings.neurips.cc/paper_files/paper/2025/file/6789f033ebde742552e5db84fb5d414a-Paper-Conference.pdf) | 33 | This DAPO means Direct Advantage-Based Policy Optimization, a different paper from the open-source DAPO system. It uses step-level offline RL and trained critics; do not merge the two recipes. |
| 33 | 2025 | [Ask a Strong LLM Judge when Your Reward Model is Uncertain ](https://proceedings.neurips.cc/paper_files/paper/2025/file/6c312cf14321a99ae3788a41593cf2bd-Paper-Conference.pdf) | 26 | Routes uncertain reward-model judgments to a stronger judge. Judge-based training is a design option with uncertainty and cost considerations, rather than something already connected in our code. |
| 34 | 2025 | [When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs ](https://proceedings.neurips.cc/paper_files/paper/2025/file/706338a08f9378b708f21cbf5686e617-Paper-Conference.pdf) | 38 | Reasoning can interfere with instruction following. Evaluate task compliance as a separate outcome instead of presuming every additional reasoning behavior helps. |
| 35 | 2025 | [Diversity-Aware Policy Optimization for Large Language Model Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/8808b4c72d5ade9bbcf9270ac6411314-Paper-Conference.pdf) | 26 | Explicitly optimizes diversity and analyzes length-related confounding. Compare structural changes at matched length and account for entropy effects. |
| 36 | 2025 | [SCOUT: Teaching Pre-trained Language Models to Enhance Reasoning via Flow Chain-of-Thought ](https://proceedings.neurips.cc/paper_files/paper/2025/file/8904a83578314d92558b8a49803b68aa-Paper-Conference.pdf) | 25 | Flow chain-of-thought changes the reasoning representation and training setup. Separate representation learning or distillation from RL contributions. |
| 37 | 2025 | [The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/9a14cf2ee36cf2524f54d8f196a6734c-Paper-Conference.pdf) | 31 | Entropy-minimization training includes a reported 40-gradient-step EM-FT setup and 100-step RL setups with large batches. The separate 5–15-step figure is token-logit inference optimization, not model RL training. |
| 38 | 2025 | [Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models ](https://proceedings.neurips.cc/paper_files/paper/2025/file/9baf31febefde7bd76023c2d2f13cbd7-Paper-Conference.pdf) | 65 | Studies structured instructions and rule-based compliance rewards. Instruction compliance is a valid narrowly defined metric, but does not establish substantive proposal quality. |
| 39 | 2025 | [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data ](https://proceedings.neurips.cc/paper_files/paper/2025/file/9d7a6dc57a3216dd1a2e80a8c4a55ed6-Paper-Conference.pdf) | 47 | Treats domain-uncertain data diversity as a learning problem. A pooled mixture has its own data-selection assumptions and should be compared with declared sampling ratios. |
| 40 | 2025 | [Checklists Are Better Than Reward Models For Aligning Language Models ](https://proceedings.neurips.cc/paper_files/paper/2025/file/a6837c1dd021f76f1b4098e3722052a8-Paper-Conference.pdf) | 27 | Instruction-specific checklists produce rewards using judges and verifiers. This supports richer task assessment, while the paper also documents substantial judging cost. |
| 41 | 2025 | [Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/a9a3c5df70368a2f8c3fd2ed12620203-Paper-Conference.pdf) | 39 | Uses trajectory-level Bellman-residual optimization and 128-prompt batches. Alternative RL estimators differ in sample use; compare actual tokens, samples and optimization work. |
| 42 | 2025 | [Reasoning Planning for Language Models ](https://proceedings.neurips.cc/paper_files/paper/2025/file/b03289925195cbdb80970003cd1c168a-Paper-Conference.pdf) | 29 | Selects among reasoning strategies using reward information. Inference-time planning and additional sampling should be distinguished from model-weight training. |
| 43 | 2025 | [DAPO: An Open-Source LLM Reinforcement Learning System at Scale ](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf) | 23 | Open-source DAPO uses 512 prompts times 16 responses per rollout batch and 16 gradient updates for that batch. Report rollout batches and optimizer updates separately. |
| 44 | 2025 | [SeRL: Self-play Reinforcement Learning for Large Language Models with Limited Data ](https://proceedings.neurips.cc/paper_files/paper/2025/file/95c2cbe23fb6d28a4ae908aa7f3de5bf-Paper-Conference.pdf) | 33 | Self-play can increase its reward while external accuracy decreases. Explicitly inspect reward hacking and maintain independent outcome validation. |
| 45 | 2025 | [DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization ](https://proceedings.neurips.cc/paper_files/paper/2025/file/52d6c85793b3dd47636eb430710da915-Paper-Conference.pdf) | 28 | DisCO analyzes difficulty bias and reports 1,400-step 1.5B and 1,000-step 7B/8B setups. These are method-specific budgets, not evidence that a smaller pilot cannot be useful. |
| 46 | 2025 | [Efficiently Scaling LLM Reasoning Programs with Certaindex ](https://proceedings.neurips.cc/paper_files/paper/2025/file/d037fd021c9aace128b8ce25001cdb6c-Paper-Conference.pdf) | 39 | Certaindex adapts inference-time compute after answer stabilization. Report compute savings and correctness together; reasoning-program steps are not RL updates. |
| 47 | 2025 | [SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/cb7baa005c239c1c7c4098c2a9e00450-Paper-Conference.pdf) | 27 | Self-play critics learn to identify faulty reasoning steps. Process evaluation needs independently informative criticism rather than treating a heuristic label as validated truth. |
| 48 | 2025 | [Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning ](https://proceedings.neurips.cc/paper_files/paper/2025/file/cc7611bdba09a35c9db78dc4419734a6-Paper-Conference.pdf) | 32 | Reasoning path compression evaluates inference memory and throughput with accuracy retention. Structural or length changes can arise through inference modifications rather than training. |
| 49 | 2026 | [SARL: Label-Free Reinforcement Learning by Rewarding Reasoning Topology ](https://arxiv.org/pdf/2603.27977) | 15 | SARL directly optimizes a rollout-level graph-structure reward derived from clustering and path length. It is a close structural-RL baseline, but does not learn an individual human reference topology. |
| 50 | 2026 | [Unveiling Implicit Advantage Symmetry: Why GRPO Struggles with Exploration and Difficulty Adaptation ](https://arxiv.org/pdf/2602.05548) | 26 | A-GRAE analyzes group-relative advantage symmetry and difficulty adaptation; reports 1,024 queries with eight rollouts. Advantage construction and exploration are substantive experimental factors. |

## Code references

- `scripts/experiment/thinkaloud_rl/paper/scoring.py`: current writing proxy and unscored judge tasks.
- `scripts/experiment/thinkaloud_rl/paper/judge.py`: blinded criterion-wise writing evaluation.
- `scripts/experiment/thinkaloud_rl/reasoning_shaper.py`: regex labels and phase/transition potential.
- `scripts/experiment/thinkaloud_rl/paper/priors.py`: pooled, domain-specific, shuffled and uniform priors.
- `scripts/experiment/thinkaloud_rl/paper/train.py`: detached causal-prefix contribution and fixed budget normalization.
- `scripts/experiment/thinkaloud_rl/paper/suite.py`: pilot model, seed, groups and 150-update configuration.

No training objective, model configuration or dataset was changed by this review.
No GPU jobs were launched. The report does not certify NeurIPS acceptance.
