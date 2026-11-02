# Thinking like the writer: literature synthesis

**Working idea.** Take a person's think-aloud brainstorming trajectory: ideas proposed, alternatives weighed, dead-ends and revisits over time, from the ThoughtGraph reasoning graphs. Inject it into an LLM's reasoning process (prefilled `<think>` block, or trained with SFT + GRPO to "think like the user"). The model then writes in that person's voice, with their idiosyncratic ideas, instead of producing homogenized text. Venue: ICLR. No new user study.

Detailed reports (all citations web-verified by search agents; unverifiable items marked [UNVERIFIED] in each file):
- `01_homogenization_diversity.md` — evidence, causes, metrics, training-free fixes
- `02_rl_creative_writing.md` — GRPO variants, writing rewards, diversity-aware RL, pitfalls
- `03_personalization.md` — benchmarks, per-user methods, voice metrics, author datasets
- `04_human_thought_traces.md` — thinking interventions, human traces, graph encoders, process datasets
- `05_evaluation.md` — validated automatic evaluation without a user study

Before submission, re-check every citation you use against arXiv or the venue page, especially 2026 preprints whose venue may have changed.

---

## 1. The problem is well established and getting worse

- **Individual gain, collective loss.** AI help makes each writer's output rate higher but makes writers converge: Doshi & Hauser (*Sci. Adv.* 2024); Padmakumar & He (ICLR 2024); Anderson et al. (C&C 2024). The loss persists after assistance ends (Kumar et al., CHI 2025), erases demographic and cultural signal (Agarwal et al., CHI 2025; Sourati et al., *Nat. Hum. Behav.* 2026), and flattens first-person voice (van Nuenen 2026).
- **Models converge with each other too**, so switching models doesn't help: Artificial Hivemind (NeurIPS 2025 D&B Best Paper); Wenger & Kenett (*PNAS Nexus* 2026).
- **Post-training is the cause.** RLHF cuts diversity (Kirk et al., ICLR 2024); preference data carries a typicality bias (Verbalized Sampling, 2025); the loss happens at the SFT stage for thinking models and at DPO for instruct models, and lives in the weights (Karouzos et al. 2026); reasoning models are *less* uncertain in creative writing than base models (Sui 2026); RLVR narrows what models can produce (Yue et al., NeurIPS 2025).
- **Implication:** "homogenization exists" is no longer a contribution. **The contribution must be the fix**, and the fix must survive GRPO, which is itself a known collapse risk.

## 2. Existing fixes add randomness, not a person

| Family | Examples | What's missing |
|---|---|---|
| Sampling / decoding | min-p, contrastive decoding, BACo, conformative decoding | Diversity is random, not grounded in any individual |
| Prompting | Verbalized Sampling, String Seed of Thought, Intent Factored Generation | Synthetic "intents", not a real human's reasoning |
| Diversity-aware post-training | DivPO, DDPO/DORPO (COLM 2025), DARLING, DQO (ICLR 2026), CrPO, DPWriter | Diversity across samples; no fidelity to a specific author |
| Writing RL | Writing-Zero, RLMR, Rubrics-as-Rewards, LongWriter-Zero, RLCS, UniCreative | Optimizes generic quality, which pulls toward AI tone (Antislop, Rubicon) |
| Personalization | LaMP/LongLaMP, REST-PG, OPPU, P2P hypernetworks, StyleVector, VPL | Conditions on finished past texts; inference-time methods stay below the human authorship floor on informal writing (PersonalBench 2026; Catch Me If You Can, EMNLP-F 2025) |

**The gap:** no line of work combines (a) *process-level* human signal, (b) fidelity to *this* author, and (c) distinctness from *other* authors.

## 3. Closest prior work and novelty risks

| Work | Overlap | How we differ |
|---|---|---|
| **REER / DeepWriting-20K** (ICLR 2026) | Trains a writer on reasoning traces paired with good human texts | Traces are **synthetic**, reverse-engineered by perplexity search, and generic. Ours are **observed, individual human** think-aloud. Strongest baseline and must-cite. |
| **Thinking Intervention** (2025), **ThoughtMani**, **ThinkPilot** | Put outside content inside `<think>` | Short designer guidance or a smaller model's CoT, aimed at accuracy, efficiency or safety. Not a person's trajectory, not writing. |
| **Think-aloud persona fine-tuning** (Ishikura et al., PALS@EMNLP 2025) | `<thinking>{think-aloud}</thinking>` SFT for individual modeling | Dialog personality, not long-form writing; no RL, no homogenization measurement. Cite and contrast. |
| **Hippo / Interactive Reasoning** (IUI 2026), Co-CoT | Humans edit the reasoning tree at inference | Live editing vs a naturally captured trajectory used for *training*; no voice/diversity objective. |
| **TagPR, PRIME, REST-PG** | Personalized reasoning via SFT/RL | Reasoning is *inferred by the model* from user history, not recorded from the user. |
| **Thought Cloning** (NeurIPS 2023) | Imitate human thoughts plus actions | Gridworld with synthetic thoughts; conceptual ancestor. |
| **ScholaWrite** (ACL 2026) | Trains on human writing-process logs | Edit and intention level; no upstream brainstorming, no think block. |
| **Capturing Classic Authorial Style with GRPO** (CoNLL 2026) | Authorship-verification reward in GRPO | Voice of famous authors from their corpus; no process conditioning, no ideas. |

**No paper was found that prefills or trains a reasoning model on a person's own brainstorming or think-aloud trajectory to generate that person's writing.** That's the novelty claim. Do a final arXiv sweep before submission.

**Evidence that supports the premise:**
- Human think-aloud traces are more varied and less rigid than model traces (Kargupta et al. 2025, 54 human vs 192K model traces).
- LLM-simulated think-alouds are over-coherent (Borchers et al. 2026).
- The *behaviors* in a trace (backtracking, verification) matter more for RL than correctness (Gandhi et al., COLM 2025), which supports keeping dead-ends.

**Evidence to pre-empt:**
- Human-written CoT can underperform model CoT in distillation on math (Findings ACL 2025). Our objective is fidelity and diversity, not accuracy.
- SFT alone can collapse diversity (Karouzos et al. 2026), so SFT on 17 people's traces needs a diversity-preserving RL stage.

## 4. Datasets

**Primary (ours):** ThoughtGraph study, ~17 participants. Brainstorm-phase reasoning graph snapshots (~600 per person) with raw utterances, node types, dead-end and revisit edges, timestamps, plus keystroke-logged writing with AI-paste events (so self-typed text can be isolated).
- **No public corpus pairs think-aloud brainstorming with keystroke-logged writing.** Releasing it (if consent allows) is a contribution.
- Use raw `utterance` fields, not the LLM-written `distilled_claim` paraphrases, to avoid re-injecting model voice.

**Public, for scale and voice metrics:**

| Dataset | Use |
|---|---|
| Blog Authorship Corpus (19K authors, ~35 posts each; non-commercial) | Held-out voice evaluation; synthetic-trajectory scale experiment |
| Catch-Me-If-You-Can release (CCAT50, Enron, Blog, Reddit) | Ready-made author-imitation protocol and baselines |
| LongLaMP | Standard personalized long-form benchmark (REST-PG baseline) |
| DeepWriting-20K (REER) | Synthetic-trace baseline data |
| Cognitive Foundations traces (54 human think-alouds) | Evidence human traces differ; possible extra human traces |
| CoAuthor, ScholaWrite, KLiCKe | Writing-process data (no brainstorming); related-work contrast |
| LitBench, HANNA, TTCW, LAMP, RoSE, Ocsai AUT | Validating judges and metrics (Section 6) |

## 5. Method design, informed by the literature

**Main method: SFT + GRPO, "think like the writer."**
1. **SFT:** user trajectory rendered as first-person thinking → the user's self-typed text for the same section.
2. **GRPO:** given the first part of a trajectory, the model continues the thinking and writes. Rewards:
   - **Idea fidelity:** atomic idea units from the trajectory, entailment recall with MiniCheck/AlignScore, **weighted by rarity across participants** (MuseScorer-style rarity).
   - **Ruled-out reintroduction penalty:** draft entails an idea the user marked as a dead-end (RLMR-style constraint weighting). **New metric.**
   - **Voice:** LUAR similarity to held-out self-typed text (as in the authorial-style GRPO paper), cross-checked with StyleDistance so topic overlap doesn't inflate it.
   - **Anti-homogenization:** group-relative diversity gated by quality (DARLING/DQO style), plus distance from other users' outputs on the same prompt.
   - **Craft:** WQRM or LitBench reward model, absolute scores only.
3. **Stability:** Dr. GRPO (no length/std normalization), DAPO Clip-Higher and overlong shaping, Clip-Cov/KL-Cov entropy control.

**Injection-site ablation (analysis contribution).** Same information, different entry point:
- A: final outline in the prompt
- B: trajectory in the prompt
- C: trajectory prefilled into `<think>` (Thinking Intervention style)
- D: trained (SFT+GRPO)
- E (optional): graph encoder soft tokens (GraphToken) or per-user cartridge/LoRA

**Baselines:**
- Verbalized Sampling
- Few-shot style imitation with a frontier model
- StyleVector steering
- REST-PG
- OPPU / P2P
- REER / DeepWriter-8B
- DDPO/DORPO
- DARLING
- Quality-only GRPO
- Authorship-reward GRPO

## 6. Evaluation without a user study

| Claim | Metric | Validated against |
|---|---|---|
| Keeps the user's ideas | Idea-unit recall/precision (FActScore-style + MiniCheck) | RoSE ACU labels, LLM-AggreFact |
| Respects what they ruled out | Ruled-out reintroduction rate | Perturbation tests (inject dead-end ideas → metric must rise) |
| Sounds like them | LUAR vs held-out self-typed text, human cross-author floor/ceiling (PersonalBench protocol); attribution among participants | Catch-Me protocol; STEL-or-Content |
| Not homogenized | Cross-user similarity (Padmakumar & He), Vendi, NoveltyBench distinct@k, Creativity Index, Antislop frequency | Known-groups: TTCW professional > LLM; LAMP edited > unedited |
| Still good writing | WQRM + rubric judge from a *different model family*, length-controlled | LitBench test, HANNA, LAMP/WQ; report κ; PPI for confidence intervals |

Rules from the judge-reliability literature: never judge with the generator's model family (self-preference); use absolute rather than pairwise scores for anything RL can game; report κ, not percent agreement; report metrics that are also rewards only alongside held-out judges.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| ~17 participants | Leave-participants-out evaluation, per-participant results, synthetic-trajectory scale experiment (clearly secondary), dataset release |
| Leakage (trajectory wording copied into draft) | Cap n-gram overlap in reward, report overlap, StyleDistance check |
| Reward hacking (length, rubric, diversity via off-topic text) | Dr. GRPO, length control, quality-gated diversity, held-out judges (Mahmoud et al. 2026) |
| Circularity (LLM rewards + LLM judges) | Different model families; real human trajectories as inputs; validation on human-rated sets |
| SFT collapses diversity | Keep the SFT stage short; diversity reward in RL; report diversity after each stage |
| "Human-in-think-block exists" (Hippo, PALS persona) | Cite; emphasize trained use, writing domain, homogenization objective |
| Consent for releasing transcripts | Check IRB before promising the dataset |

## 8. Suggested paper outline

1. **Motivation:** homogenization + mode collapse; the missing ingredient is the individual's reasoning process.
2. **Task and data:** thought-trajectory → personal writing; ThoughtGraph dataset; public voice sets.
3. **Method:** injection-site framework (A–E) + SFT/GRPO "think like the writer" with process-consistency rewards (rarity-weighted fidelity, ruled-out penalty, voice, group diversity, craft).
4. **Evaluation:** validated automatic protocol (Section 6).
5. **Results:** process > product conditioning; trained > prefill > prompt; diversity–fidelity frontier vs DARLING/DDPO/REER.
6. **Analysis:** which trajectory elements matter (dead-ends, branches, timing); failure modes; limits of a small-N dataset.

## 9. Immediate next steps

1. Data audit (read-only): brainstorm-phase utterances, dead-ends and self-typed text per participant; draft one rendered "human thinking" sample.
2. Pilot A/B/C with an open reasoning model (no training) and measure cross-user homogenization and voice. This gives the motivating result.
3. Final arXiv novelty sweep on "think-aloud" / "human reasoning trace" / "personalized writing" before investing in GRPO.
