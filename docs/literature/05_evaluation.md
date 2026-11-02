# Automatic evaluation of creative and personal writing, validated against existing human ratings

Constraint: no new user study. Every item was checked against arXiv, ACL Anthology, ACM, PMLR, GitHub or Hugging Face except where marked [UNVERIFIED]. Web-search quota ran out near the end, so a few details (OpenMEVA size, some licenses and venues) are flagged.

## 1. Creative writing evaluation frameworks

- **Art or Artifice? LLMs and the False Promise of Creativity** — Chakrabarty et al., 2024, CHI — https://arxiv.org/abs/2309.14556 — TTCW: 14 binary tests grouped under Fluency, Flexibility, Originality and Elaboration. 3 experts × 48 stories = 2,016 verdicts with rationales. LLM stories pass 3–10× fewer tests than professional ones, and no LLM judge correlated positively with the experts. *Relevance:* the clearest negative result on LLM judges. Data: HF `salesforce/ttcw_creativity_eval` (BSD-3); New Yorker texts linked, not included.
- **Can AI writing be salvaged? Mitigating Idiosyncrasies… through Edits** — Chakrabarty, Laban, Wu, 2025, CHI — https://arxiv.org/abs/2409.14509 — Seven-category taxonomy of LLM writing flaws (cliché, redundant exposition, purple prose, poor sentence structure, lack of specificity, awkward word choice, tense inconsistency). LAMP corpus: 1,057 LLM paragraphs edited by professional writers. *Relevance:* span-level ground truth for a slop/idiosyncrasy detector. github.com/salesforce/creativity_eval (BSD-3).
- **AI-Slop to AI-Polish? Edit-Based Writing Rewards and Test-time Computation** — Chakrabarty et al., 2025, arXiv (venue [UNVERIFIED]) — https://arxiv.org/abs/2504.07532 — Combines 5 preference datasets into the Writing Quality (WQ) benchmark; the WQRM reward model reaches 74%, and experts agreed with its picks 66% of the time. *Relevance:* ready-made validated quality scorer.
- **LitBench** — Fein et al., 2025/26, EACL 2026 — https://arxiv.org/abs/2507.00769 — 2,480 debiased human preference test pairs plus 43,827 training pairs from r/WritingPrompts. Claude-3.7-Sonnet agrees with humans 73% of the time; trained reward models reach 78%.
- **Of Human Criteria and Automatic Metrics (HANNA)** — Chhun et al., 2022, COLING — https://aclanthology.org/2022.coling-1.509/ — 1,056 stories rated on 6 criteria; 72 automatic metrics tested against them.
- **Do Language Models Enjoy Their Own Stories?** — Chhun et al., 2024, TACL — https://aclanthology.org/2024.tacl-1.62/ — LLM ratings beat older metrics at the system level, but their explanations are weak.
- **StoryER** — Chen et al., 2022, EMNLP — https://aclanthology.org/2022.emnlp-main.114/ — ~100k ranked story pairs (by upvotes) and 46k aspect ratings with comments.
- **OpenMEVA** — Guan et al., 2021, ACL — https://aclanthology.org/2021.acl-long.500/ — Test suite for story metrics: human correlation, generalization, perturbation robustness. 1–5 annotations; annotated-set size [UNVERIFIED].
- **Pron vs Prompt** — Marco et al., 2024, EMNLP — https://aclanthology.org/2024.emnlp-main.1096/ — Novelist Patricio Pron vs GPT-4; 5,400 critic ratings on a rubric based on Boden's definition of creativity.
- **Evaluating Creative Short Story Generation in Humans and LLMs** — Ismayilzada et al., 2024/25, ICCC 2025 — https://arxiv.org/abs/2411.02316 — "Complexity illusion": LLM judges and non-experts rate LLM stories as more creative; experts and automatic novelty metrics favor human stories.
- **EQ-Bench Creative Writing v3** — Paech, 2024–25, leaderboard — https://eqbench.com/creative_writing.html — Claude judge with rubric plus pairwise Elo; not validated against human ratings, so cite as common practice only.

## 2. How reliable LLM judges are

- **Judging LLM-as-a-Judge (MT-Bench)** — Zheng et al., 2023, NeurIPS D&B — https://arxiv.org/abs/2306.05685 — GPT-4 agrees with humans over 80% on chat; identifies position, verbosity and self-enhancement biases.
- **LLM Evaluators Recognize and Favor Their Own Generations** — Panickssery et al., 2024, NeurIPS — https://arxiv.org/abs/2404.13076 — Self-preference grows with a model's ability to recognize its own text. *Relevance:* never use the generator's model family as judge.
- **Length-Controlled AlpacaEval** — Dubois et al., 2024, arXiv — https://arxiv.org/abs/2404.04475 — Removes length bias with a regression on length difference.
- **Style Outweighs Substance** — Feuer et al., 2025, ICLR — https://arxiv.org/abs/2409.15268 — LLM judges put style ahead of substance.
- **Pairwise or Pointwise?** — Tripathi et al., 2025, arXiv — https://arxiv.org/abs/2504.14716 — A distractor feature flips 35% of pairwise verdicts but only 9% of absolute scores. *Relevance:* use absolute scores for anything an RL policy could game.
- **LLMs instead of Human Judges? (JUDGE-BENCH)** — Bavaresco et al., 2025, ACL (short) — https://aclanthology.org/2025.acl-short.20/ — 20 human-annotated datasets, 11 LLMs; judges must be validated per task.
- **Judging the Judges** — Thakur et al., 2025, GEM workshop — https://aclanthology.org/2025.gem-1.33/ — Report Cohen's κ, not raw percent agreement.
- **AutoEval Done Right** — Boyeau et al., 2024, arXiv — https://arxiv.org/abs/2403.07008 — Prediction-powered inference (PPI) corrects judge bias with a small human-labeled set and gives valid confidence intervals. *Relevance:* calibrate with existing human labels.

## 3. Idea fidelity and content faithfulness

- **FActScore** — Min et al., 2023, EMNLP — https://aclanthology.org/2023.emnlp-main.741/ — Splits text into atomic facts and checks each against a source. *Relevance:* apply both ways: source→output for idea recall, output→source for unsupported additions.
- **MiniCheck + LLM-AggreFact** — Tang, Laban, Durrett, 2024, EMNLP — https://aclanthology.org/2024.emnlp-main.499/ — 770M-parameter checker at GPT-4 accuracy on a human-labeled grounding benchmark. *Relevance:* cheap per-idea entailment scorer.
- **AlignScore** — Zha et al., 2023, ACL — https://arxiv.org/abs/2305.16739 — One consistency model covering NLI, QA and paraphrase.
- **QAFactEval** — Fabbri et al., 2022, NAACL — https://arxiv.org/abs/2112.08542 — QA-based consistency checking; complements NLI.
- **RoSE / Atomic Content Units (ACU)** — Liu et al., 2023, ACL — https://aclanthology.org/2023.acl-long.228/ — 22k human coverage annotations at the ACU level (BSD-3). *Relevance:* best human benchmark for validating idea-unit coverage.
- **FineSurE** — Song et al., 2024, ACL — https://aclanthology.org/2024.acl-long.51/ — LLM checking of faithfulness, completeness (key-fact coverage) and conciseness.

## 4. AI-typical style, homogenization and authorship

- **Measuring AI "Slop" in Text** — Shaib et al., 2025, arXiv 2509.19163 (reported for ICLR 2026, unconfirmed) — Expert taxonomy with span-level annotations.
- **Antislop** — Paech et al., 2025 — https://arxiv.org/abs/2510.15061 — Some phrase patterns are over 1,000× more frequent in LLM text than human text. *Relevance:* ready slop-frequency metric.
- **Idiosyncrasies in LLMs** — Sun et al., 2025, ICML — https://arxiv.org/abs/2502.12150 — 97.1% accuracy identifying the source model; survives rewriting.
- **Binoculars** — Hans et al., 2024, ICML — https://arxiv.org/abs/2401.12070
- **Fast-DetectGPT** — Bao et al., 2024, ICLR — https://arxiv.org/abs/2310.05130
- **RAID** — Dugan et al., 2024, ACL — https://aclanthology.org/2024.acl-long.674/ — 6M-generation benchmark showing detectors are brittle.
- **People who frequently use ChatGPT… are accurate detectors** — Russell et al., 2025, ACL — https://aclanthology.org/2025.acl-long.267/ — 300 articles with expert labels and written explanations.
- **LUAR** — Rivera-Soto et al., 2021, EMNLP — https://aclanthology.org/2021.emnlp-main.70/ — Authorship embeddings. *Relevance:* voice preservation as LUAR similarity to the user's own writing.
- **PersonalBench** — 2026, arXiv 2608.19746 — https://arxiv.org/abs/2608.19746 — LUAR + LLM judge + stylometrics; LLM output is further from any human author (0.48–0.51) than humans are from each other (0.626 floor). The LLM judge separated methods LUAR couldn't.
- **Catch Me If You Can? Not Yet** — Wang et al., 2025, Findings of EMNLP — https://arxiv.org/abs/2509.14543 — Style imitation across 400+ authors fails on informal writing.
- **Padmakumar & He** (ICLR 2024), **Agarwal et al.** (CHI 2025), **Echoes in AI / Sui Generis** (PNAS 2025; moderate correlation with human surprise), **Creativity Index** (ICLR 2025 oral; human authors +66.2% vs LLMs), **NoveltyBench** (venue [UNVERIFIED]) — see `01_homogenization_diversity.md`.

## 5. Divergent thinking and originality scoring

- **Beyond Semantic Distance (Ocsai)** — Organisciak et al., 2023, Thinking Skills and Creativity — https://doi.org/10.1016/j.tsc.2023.101356 — 27,217 human-judged Alternate Uses Task (AUT) responses; fine-tuned LLMs reach r=0.81 with human raters, semantic distance only 0.12–0.26.
- **MuseScorer** — Bangash et al., 2025, arXiv — https://arxiv.org/abs/2505.16232 — Originality scored by idea rarity; r=0.89 with humans across 16,294 ideas.
- **Doshi & Hauser** — Science Advances 2024 — https://www.science.org/doi/10.1126/sciadv.adn5290
- **The Language of Creativity** — Orwig et al., 2024, Journal of Creative Behavior — https://pmc.ncbi.nlm.nih.gov/articles/PMC11065427/ — GPT-4 creativity ratings correlate r=.65 with humans; data on request only.

## Human-rated datasets

| Name | Content | Rating dimensions | Size | License | URL |
|---|---|---|---|---|---|
| TTCW | Short stories: New Yorker vs LLM | 14 binary tests + expert rationales | 48 stories × 3 experts = 2,016 verdicts | BSD-3 (New Yorker texts linked only) | huggingface.co/datasets/salesforce/ttcw_creativity_eval |
| LAMP | LLM paragraphs with professional edits | 7 flaw categories (span), expert preferences | 1,057 paragraphs | BSD-3 | github.com/salesforce/creativity_eval |
| WQ benchmark | 5 writing-preference sets combined | Pairwise quality | [UNVERIFIED size] | BSD-3 | same repo, WritingRewards/ |
| LitBench | r/WritingPrompts story pairs | Human (upvote) preference | 2,480 test / 43,827 train | Test: Reddit IDs only (rehydrate via API); train source MIT | huggingface.co/SAA-Lab |
| HANNA | 1,056 generated stories | Relevance, Coherence, Empathy, Surprise, Engagement, Complexity | 19,008 ratings (3 per story) + LLM ratings | MIT | github.com/dig-team/hanna-benchmark-asg |
| StoryER | WritingPrompts stories | Upvote ranking; 1–5 aspect ratings + comments | ~100k pairs, 46k ratings | [UNVERIFIED] | aclanthology.org/2022.emnlp-main.114 |
| OpenMEVA | ROCStories/WritingPrompts generations | Overall quality 1–5 | [UNVERIFIED] | [UNVERIFIED] | github.com/thu-coai/OpenMEVA |
| Ismayilzada creative stories | 5-sentence cue-word stories, human and LLM | Creativity, originality, surprise, value (expert and non-expert) | 479 stories | [UNVERIFIED] (code MIT) | huggingface.co/datasets/mismayil/creative_story_generation_dataset |
| Doshi & Hauser | 8-sentence stories with/without AI ideas | Novelty, usefulness, enjoyment | 293 writers, 3,519 ratings | Dryad (reuse with attribution) | datadryad.org/dataset/doi:10.5061/dryad.qfttdz0pm |
| Pron vs Prompt | Pron vs GPT-4 stories (EN/ES) | Boden-based rubric | 5,400 ratings | [UNVERIFIED] | aclanthology.org/2024.emnlp-main.1096 |
| Ocsai AUT federated set | Alternate Uses Task responses | Originality | 27,217 responses | [UNVERIFIED] | openscoring.du.edu |
| RoSE | News and dialogue summaries | ACU coverage + 3 other protocols | 22k summaries | BSD-3 | huggingface.co/datasets/Salesforce/rose |
| LLM-AggreFact | Grounded claims from 11 datasets | Supported / unsupported | [UNVERIFIED size] | Mixed per source | huggingface.co/datasets/lytang/LLM-AggreFact |
| Russell et al. | 300 non-fiction articles | Human vs AI label + rationale | 300 × 5 annotators | [UNVERIFIED] | aclanthology.org/2025.acl-long.267 |

## Recommended evaluation protocol (automatic, validated)

1. **Idea fidelity.** Split source thoughts into atomic idea units (FActScore/ACU style).
   - Recall: share of units entailed by the output (MiniCheck or AlignScore).
   - Precision: share of output claims supported by the source.
   - Validate on RoSE ACU labels and LLM-AggreFact; report balanced accuracy and Kendall τ.
2. **Writing quality.** WQRM plus one absolute-score rubric judge from a different model family than the generator.
   - Validate on LitBench-test, LAMP/WQ and HANNA; report κ or Spearman ρ with human agreement as the ceiling.
   - Control length with a regression like Length-Controlled AlpacaEval.
3. **Creativity and novelty.** Creativity Index, Sui Generis, plus an LLM originality rater.
   - Validate on originality/surprise ratings (Ismayilzada data, HANNA-Surprise, Doshi & Hauser).
   - Check that metrics rank TTCW's professional stories above the LLM stories.
4. **Homogenized voice.**
   - Antislop pattern frequency and LAMP-category rates.
   - Binoculars as a secondary signal only.
   - LUAR similarity to the user's own prior text, against a human cross-author floor (as in PersonalBench).
   - Similarity between outputs from different users (Padmakumar & He).
   - Validate detectors against the Russell et al. expert labels.
5. **Statistics.**
   - Panel of several judges; randomized positions; absolute scores for any reward signal.
   - Report inter-judge κ and bootstrap CIs; apply PPI (Boyeau et al.) where a human-labeled subset exists.
   - Known-groups checks (human > LLM, edited > unedited on LAMP) and perturbation tests (removing ideas must lower recall).

## Reviewer objections to anticipate

- **"LLM judges don't track experts on creativity"** (TTCW, Ismayilzada). Answer with per-metric validation numbers and human ceilings; avoid claims resting on the judge alone.
- **Circularity and reward hacking.** If a metric is an RL reward, report results only on held-out metrics or judges; pairwise verdicts are easy to game.
- **Self-preference and model-family leakage.** Use judges from other families and report results per judge.
- **Domain shift.** Validation sets are fiction, news or Reddit, not reflective or academic writing. Admit it and add perturbation checks.
- **Upvotes aren't quality.** LitBench and StoryER labels reflect popularity with Reddit's demographic skew.
- **Novelty vs incoherence.** Rare-idea metrics reward nonsense; pair novelty with coherence and fidelity.
- **AI-text detectors are brittle** (RAID) and can penalize non-native writers; treat as secondary.
- **Fidelity vs elaboration.** Separate harmless from harmful unsupported additions, and show the atomic-unit extraction is reliable.
- **Licensing and reproducibility.** LitBench needs Reddit rehydration; New Yorker texts can't be redistributed. Release code and IDs only.
- **No new user study.** Cite JUDGE-BENCH on validating with existing task-specific annotations, and use PPI to bound error.
