# Training the *form* of reasoning: making an LLM's brainstorm graph topologically human

Written 2026-10-14. Companion to `TOPOLOGY_RESEARCH_REPORT.md`. Sources were verified by web search against
arXiv/publisher pages unless marked [memory]. Two reports (human cognition; LLM reasoning structure) are condensed here.

## 1. What our own data says (14 human brainstorms, final snapshot; `data/rg_cache`)

| | V | E/V | β1/V | trees | leaf frac | branch frac | revisit reach-back (nodes) |
|---|---|---|---|---|---|---|---|
| Human (n=14) | 74 ± 40 | 1.04 ± 0.10 | 0.077 ± 0.096 | 21% | 0.15 | 0.08 | median 16, IQR 6–30, tail to 100 (68 edges) |
| Llama-3.1-8B graphs (n=18) | 20 | 0.90 | 0.005 | 89% | 0.21 | 0.00 | ≈ 0 |
| Qwen3-8B graphs (n=18) | 9 | 0.89 | 0.006 | 94% | 0.28 | 0.02 | 0 |

Readings that matter for reward design:
1. Humans are **not** densely connected. The backbone is a chain (most nodes degree 2, 8% branch points) and cycles come
   from a *few* long-range `synthesizes` / `revisits` edges. "Human-like" = chain + sparse long-range closure.
2. LLM graphs are small, **fragmented** (E/V < 1: several disconnected pieces), and strictly forward (no reach-back).
   Llama emits `revisits`/`synthesizes` *labels* that do not close loops: the vocabulary of the form without the form
   ("topological mimicry", Chen & Wang 2026). Any structural reward must be computed from the parsed structure, never from labels.
3. Reach-back of human revisits is heavy-tailed (40% within 10 nodes, tail to 100). This is an empirical memory-access
   curve: the natural target for a forgetting model.
4. Writers differ enormously in form: P16 is web-like (β1/V 0.37, E/V 1.36), P12/P13/P17 are pure trees. Human form is a
   **distribution**, and form itself is personalizable.

## 2. Human idea formation: structure and forgetting (cognitive science)

- **Abbott, Austerweil & Griffiths 2015** (Psych Review 122:558): human fluency sequences ≈ random walk on a semantic network
  where **revisits are silent** (a node is emitted only on first visit). Reproduces Hills et al.'s patchy "foraging"
  signature without an explicit switch rule. → Human graphs show latent cycles; the visible trace has few repeats.
- **Hills, Jones & Todd 2012** (Psych Review 119:431): optimal foraging in memory; switch clusters when local yield drops
  below the long-run average (marginal value theorem). **Nijstad & Stroebe 2006** (SIAM): idea generation = search of
  associative memory with local depletion → impasse → category switch. → Cluster size / switch timing statistics.
- **ACT-R base-level activation** (Anderson & Schooler 1991; Anderson & Lebiere 1998): A_j = ln Σ_k t_k^(−d), d = 0.5;
  retrieval probability logistic in A_j − τ. Power-law, not exponential (Murre & Dros 2015 Ebbinghaus replication).
  **Cowan 2001**: ~4 items live in focus of attention. → Parameter-light model of *which* earlier nodes are reachable.
- **Anderson, Bjork & Bjork 1994** retrieval-induced forgetting; **Storm, Angello & Bjork 2011**: forgetting competitors
  helps overcome fixation. **Richards & Frankland 2017** (Neuron): transience is regularization. → Forgetting is a
  feature to engineer, not a defect to patch.
- **Kenett et al. 2014, 2018** (creative people: shorter ASPL, lower modularity, higher small-worldness at matched degree);
  **Wang et al. 2025** (CHB:AH): LLM fluency networks have longer ASPL, lower clustering, higher modularity than humans;
  arXiv 2510.04391: LLM networks collapse to one community. Raw *density* is not consistently higher for LLMs; the robust
  gap is community structure and long-range shortcuts.
- **Goldschmidt linkography** (1990; 2014 book) and **Kan & Gero 2008** (Design Studies 29:315): design moves with
  backlinks/forelinks; link index, critical moves, chunk / web / sawtooth patterns; link-distance entropy is low for both
  empty and saturated linkographs. → A natural non-monotone (band) structural reward.
- **Mota et al. 2012, 2017, 2018** (PLOS ONE; npj Schizophrenia): speech graphs; L1–L3 short loops vs LCC/LSC long-range
  connectivity, normalized against a shuffled null. Mature adult signature: **low short-range recurrence, high long-range
  re-entry**. → Exactly the LLM pathology (re-verifying explored paths) vs the human pattern (sparse long reach-backs).
- **Gray et al. 2019** forward flow (mean distance from all prior items); **Toubia, Berger & Eliashberg 2021** (PNAS)
  speed / volume / circuitousness of a text trajectory. → Cheap embedding-space shape metrics.
- **Padmakumar & He 2024** (ICLR), **Wenger & Kenett 2025**, **Doshi & Hauser 2024**: RLHF-style training itself
  homogenizes; LLM outputs are far more similar to each other than human outputs are. → Any form reward needs a
  between-rollout diversity term or it will homogenize the form too.
- **Binz et al. 2025 Centaur** (Nature): fine-tuning on human behavioural traces yields a human-likeness predictor.
- Human↔LLM reasoning comparisons: **Kargupta et al. 2025** (arXiv 2511.16660; 54 human think-alouds + 192K model
  traces): models collapse to rigid sequential processing and *repeat verification on already-explored paths*.
  **Chen & Wang 2026** (arXiv 2606.07410): 10K annotated steps, humans vs R1 on AIME; R1 loops in local verification.
  **Wurgaft et al. 2025** (arXiv 2505.23931): 4,947 Game-of-24 think-alouds coded as search graphs by an LLM — the same
  coder can graph human and model traces identically.

## 3. LLM reasoning structure and structure-shaping RL

- **Topology of Reasoning** (Minegishi et al., NeurIPS 2025, arXiv 2506.05744): CoT steps → hidden-state clusters →
  graph; cycle count, diameter, small-world index. Reasoning models have *more* cycles and higher S than base models,
  mostly rumination. **Mapping the Minds** (Xiong et al., EMNLP 2025, 2505.13890): exploration density, branching ratio,
  convergence ratio, linearity. **LCoT2Tree** (Jiang et al., EMNLP 2025, 2505.22148): explicit revisit counter per node;
  both over- and under-branching predict failure → targets must be bands. **ReasoningFlow** (2506.02532 / 2606.05402),
  **ThinkProbe** (2606.29067): trace-graph schemas and 19-metric structural fingerprints.
- **Gandhi et al. 2025** (2503.01307): RL only amplifies structural behaviours already present; SFT-prime them first.
  **Yeo et al. ICML 2025** (2502.03373): outcome RL does not increase branching/backtracking; dense per-token n-gram
  repetition penalty (N=40, −0.05/token) is the effective anti-rumination term.
- **APR** (2602.00760): penalize only the pathological region (answer-stable tail) with
  R = correct · max(0, 1 − β·L_region); needs DAPO dynamic sampling so all-equal groups still carry signal.
- **Distribution matching**: GFlowNet fine-tuning (Hu et al. ICLR 2024, 2310.04363), **Flow of Reasoning** (2406.05673),
  **FlowRL** (2509.15207): sample traces ∝ exp(R) rather than maximize → preserves a spread of shapes.
  **Outcome-based exploration** (Song, Kempe, Munos, NeurIPS 2025, 2509.06941) and **SetPO** (2602.01062): batch-level
  diversity bonuses, transferable from answers to graph motifs.
- **Forgetting as an environment**: **Markovian Thinker / Delethink** (ICLR 2026, 2510.06557): context hard-reset every
  chunk, policy learns what to carry; **PENCIL** (ICML 2025, 2503.14337): erase consumed thoughts; **InftyThink**,
  **LightThinker**: summarize/compress. **Forgetting Transformer** (ICLR 2025, 2503.02130): per-head forget gate (needs
  architecture access). **Forgetting Curve** (EMNLP 2024, 2410.04727) and **PI-LLM** (2506.08184): LLMs fail to
  *suppress* stale information → measure the induced curve.
- Agent memories with decay: Generative Agents (0.995/hour recency), MemoryBank (Ebbinghaus exp(−t/S), S grows on recall),
  CoALA. No published RL reward derived from human think-aloud distributions was found: that slot is open.

## 4. Design: Top-GRPO-Form

**Policy output** = a brainstorm trace in a line-per-thought format, then the essay:
```
[n1] goal: ...
[n2] <- n1: ...                       (derives_from)
[n7] <- n2 | revisits n3: ...         (closes a loop to an earlier node)
[n9] <- n4 + n7: ...                  (synthesis of two lines)
[n5] <- n4 dead-end: ...
```
A deterministic parser builds a graph with the human schema; node references, not labels, define edges. Malformed
lines are dropped and the parse rate is a format reward (DeepSeek-R1 style).

**Reward** R = R_form + R_memory + R_content − R_rumination, with batch diversity.
- R_form: per-feature band score inside the human envelope (IQR → 0 penalty, quadratic outside) for E/V, β1/V, leaf
  fraction, branch fraction, synthesis rate, connectedness (one component), Mota short-loop rate vs shuffled null,
  linkography link-distance entropy. Non-monotone by design (LCoT2Tree; Kan & Gero).
- R_memory: log-likelihood of the trace's revisit targets under an ACT-R access model fitted to the 68 human backward
  edges: P(revisit j at step i) ∝ σ(A_j − τ), A_j = ln Σ t^(−d). Rewards reach-backs that follow the human decay
  (mostly recent, power-law tail), penalizes uniform access to the whole context.
- R_rumination: dense per-token repetition penalty (Yeo et al.) + penalty on cycles of length ≤ 3 returning to an
  already-resolved node (Mota L2/L3; Kargupta re-verification).
- R_content: the existing essay reward (endorsed / ruled-out / synthesis pairs). Cycles in the trace that close around
  endorsed ideas get the existing `w_cycle_boost`; this ties form to content.
- Diversity: outcome-exploration-style batch bonus on the motif vector so the 8 rollouts do not converge to one shape;
  optionally FlowRL for true distribution matching.

**Forgetting environment (the mechanism, not just a reward).** Generate the trace in chunks (e.g., every 12 nodes).
Between chunks, rewrite the context: keep the last ~4 nodes verbatim (Cowan), keep older nodes whose activation A_j is
above threshold as one-line stubs, and *remove* the rest. A `recall nX` action restores a removed node at a cost. The
policy can only revisit what it can still see, so the reach-back distribution emerges from bounded memory, and RL learns
what to carry (Delethink). The forgetting schedule is fitted to human data, not chosen by hand.

**Evaluation.** Same graph coder for human and model traces (Wurgaft); Mota recurrence profile vs shuffled null;
a human-vs-model classifier on the motif vector (target: chance); per-writer form transfer (does P16's adapter make
webs and P12's make trees?); essay content metrics unchanged; between-rollout distinctness.

**Pilots.** P0 fit the ACT-R access model on human edges (local, no GPU). P1 zero-shot trace format with Qwen3-1.7B:
parse rate, motif vector vs human. P2 GRPO 30 steps with R_form + R_rumination only. P3 add the forgetting environment.
P4 full run, 3 seeds, ablations (no R_memory; no forgetting environment; no diversity term).

**Risks.** Topological mimicry (labels without structure) → structure from references only. Reward hacking by adding
trivial nodes → length-normalized features and the content reward. Collapse to one shape → band targets + diversity.
Trace format vs Qwen3 thinking mode → keep `enable_thinking: false` and put the trace in the visible output.
