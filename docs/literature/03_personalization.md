# Personalized text generation and voice/style preservation (2023–2026)

Every entry was checked against arXiv, ACL Anthology, the conference page or HF/GitHub. "arXiv" as the venue means no peer-reviewed venue was found.

## 1. Personalization benchmarks

- **LaMP: When Large Language Models Meet Personalization** — Salemi et al., 2024, ACL — https://arxiv.org/abs/2304.11406 — Seven tasks (3 classification, 4 short generation) with user profiles and retrieval-augmented baselines; code and data creation CC BY-NC-SA 4.0. *Relevance:* the standard baseline, but outputs are short and scored with ROUGE, so voice isn't measured.
- **LongLaMP: A Benchmark for Personalized Long-form Text Generation** — Kumar et al., 2024, arXiv — https://arxiv.org/abs/2407.11016 — Four long-form tasks: email (Avocado, private, 279 users), abstracts (authors with ≥70 papers), Amazon reviews (≥50 per user), Reddit topic writing. User and temporal splits; ROUGE/METEOR. *Relevance:* the closest public long-form setup, and shows the field still scores style with n-gram overlap.
- **REST-PG: Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation** — Salemi et al., 2025, arXiv — https://arxiv.org/abs/2501.04167 — The model reasons over the user profile before writing, then improves by self-training; +14.5% on LongLaMP. *Relevance:* a reasoning-first baseline, the closest analogue to conditioning on a brainstorming trajectory.
- **PersonalLLM: Tailoring LLMs to Individual Preferences** — Zollo et al., 2025, ICLR — https://arxiv.org/abs/2409.20296 — 10.4k prompts × 8 responses; users simulated by mixing reward models; CC BY 4.0. *Relevance:* preference personalization, not authorial voice; useful as a contrast.
- **The PRISM Alignment Dataset** — Kirk et al., 2024, NeurIPS D&B (oral) — https://arxiv.org/abs/2404.16019 — 1,500 participants from 75 countries, 8,011 conversations with 21 LLMs, ratings linked to participant profiles; prompts CC BY 4.0, responses CC BY-NC 4.0. *Relevance:* individual feedback, but participants write no long text of their own.
- **Evaluating Style-Personalized Text Generation: Challenges and Directions** — Jangra et al., 2025 (arXiv v1), EMNLP 2026 main — https://arxiv.org/abs/2508.06374 — Style-discrimination benchmark over 8 writing tasks; BLEU/ROUGE are unreliable and an ensemble of varied metrics beats any single one. *Relevance:* supports an ensemble of voice metrics.
- **PersonalBench: Measuring the Authorship Gap in LLM Personalization** — Sawant, 2026, arXiv — https://arxiv.org/abs/2608.19746 — Companion ICML 2026 workshop (CTB) paper: https://arxiv.org/abs/2604.26460 — 50 authors, 1,000 generations, scored with LUAR, an LLM judge and function-word stylometry. Methods reach 0.484–0.508 against a human same-author floor of 0.626, and the metrics barely correlate. *Relevance:* the closest prior evaluation; inference-time methods only, no process data.
- **APM: Evaluating Style Personalization in LLMs with Arbitrary Preference Mappings** — Spohn, Girrbach, Akata, 2026, arXiv — https://arxiv.org/abs/2605.21063 — Hidden random mappings from attributes to style block stereotype shortcuts. Routing was most reliable, RAG was inconsistent, and soft prompts didn't beat the non-personalized baseline. *Relevance:* evidence that RAG alone doesn't reliably personalize.

## 2. Personalization methods

- **OPPU: Democratizing LLMs via Personalized PEFT** — Tan et al., 2024, EMNLP — https://arxiv.org/abs/2402.04401 — One LoRA per user plus retrieval; beats prompt-only methods on LaMP. *Relevance:* the natural per-participant baseline, though about 17 users with little data each is a hard fit.
- **Profile-to-PEFT (P2P): Instant Personalized LLM Adaptation via Hypernetwork** — Tan et al., 2025 (arXiv), ACL 2026 — https://arxiv.org/abs/2510.16282 — A hypernetwork turns an encoded profile into LoRA weights with no per-user training, and generalizes to unseen users. *Relevance:* suits a small participant pool; the profile could be a think-aloud trajectory.
- **Text-to-LoRA** — Charakorn et al., 2025, ICML — https://arxiv.org/abs/2506.06105 — A hypernetwork builds a LoRA from a natural-language task description in one forward pass. *Relevance:* could be adapted to go from a person's text or trajectory to a voice adapter.
- **Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights** — Liang et al., 2025, NeurIPS — https://arxiv.org/abs/2506.16406 — Turns unlabeled prompt batches into LoRA weights; up to 12,000× cheaper than fine-tuning. *Relevance:* an alternative hypernetwork baseline.
- **HYDRA: Model Factorization for Black-Box LLM Personalization** — Zhuang et al., 2024, NeurIPS — https://arxiv.org/abs/2406.02888 — Shared base model with per-user heads for reranker and adapter. *Relevance:* shares knowledge across users, which helps with few users.
- **Personalized Pieces (Per-Pcs)** — Tan et al., 2024 — https://arxiv.org/abs/2406.10471 — Users share PEFT pieces. *Relevance:* another sharing baseline. [Venue unconfirmed; seen only in search results.]
- **PRELUDE/CIPHER: Aligning LLM Agents by Learning Latent Preference from User Edits** — Gao et al., 2024, NeurIPS — https://arxiv.org/abs/2404.15269 — Infers readable preference descriptions from user edits and retrieves them for similar contexts; tested on summarization and email with a simulated GPT-4 user. *Relevance:* participants' edits could supervise the model the same way.
- **VPL: Personalizing RLHF with Variational Preference Learning** — Poddar et al., 2024, NeurIPS — https://arxiv.org/abs/2408.10075 — A latent variable per user conditions the reward model and policy. *Relevance:* a per-user reward option if pairwise voice judgments are collected.
- **P-RLHF: Personalized Language Modeling from Personalized Human Feedback** — Li et al., 2024, arXiv / NeurIPS 2024 workshop — https://arxiv.org/abs/2402.05133 — A lightweight user model trained jointly with the policy; handles explicit and implicit preferences. *Relevance:* same role as VPL.
- **StyleVector: Personalized Text Generation with Contrastive Activation Steering** — Zhang et al., 2025, ACL — https://arxiv.org/abs/2503.05213 — The style vector is the activation difference between a user's real text and style-free model output. No training; +8% quality and 1,700× less storage than PEFT. *Relevance:* a cheap per-participant voice baseline.
- **Style Vectors for Steering Generative LLMs** — Konen et al., 2024, EACL Findings — https://arxiv.org/abs/2402.01618 — Adds activation-derived style vectors at inference. *Relevance:* earlier steering baseline.
- **Persona Vectors** — Chen et al., 2025, arXiv — https://arxiv.org/abs/2507.21509 — Activation directions for character traits, used to monitor and control them. *Relevance:* a way to detect voice drift during fine-tuning.

## 3. Style and voice evaluation

- **LUAR: Learning Universal Authorship Representations** — Rivera-Soto et al., 2021, EMNLP — https://aclanthology.org/2021.emnlp-main.70/ — Contrastive author embedding trained on Reddit; transfers across domains; checkpoints on HF. *Relevance:* the main automatic "same author?" metric (pre-2023 but standard).
- **Can Authorship Representation Learning Capture Stylistic Features?** — Wang et al., 2023, TACL — https://arxiv.org/abs/2308.11490 — LUAR-style embeddings respond to style, not just topic. *Relevance:* supports LUAR as a voice metric.
- **STEL** — Wegmann et al., 2021, EMNLP — https://arxiv.org/abs/2109.04817 — Style-vs-content probe, including a STEL-or-Content variant; style embeddings at https://github.com/nlpsoc/Style-Embeddings. *Relevance:* checks that a metric isn't rewarding shared topic.
- **StyleDistance** — Patel et al., 2025, NAACL — https://arxiv.org/abs/2410.12757 — Style embeddings trained contrastively on synthetic paraphrases across 40 style features, so they depend less on content. *Relevance:* a complement to LUAR that reduces topic leakage.
- **STEB: Style Text Embedding Benchmark** — Rivera Soto, Wegmann, Aggazzotti, 2026, arXiv — https://arxiv.org/abs/2606.31741 — 96 datasets in 7 languages; no style embedding wins everywhere, and semantic embeddings underperform. *Relevance:* justifies using several metrics.
- **AuthBench** — Huang, Zhang, Cardie, 2026, arXiv — https://arxiv.org/abs/2609.06771 — 428k documents by 153,825 authors, 10 languages, 9 genres; CC BY 4.0. *Relevance:* for checking our metric and for extra authors.
- **Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors** — Wang et al., 2025, EMNLP Findings — https://arxiv.org/abs/2509.14543 — Few-shot imitation of 400+ authors, scored with attribution, verification and AI detection. LLMs do well on news and email but fail on blogs and forums. Data and code: https://github.com/jaaack-wang/llms-implicit-writing-styles-imitation. *Relevance:* key motivation that prompting doesn't capture informal voice; a ready-made evaluation protocol.
- **Do LLMs Write Like Humans? Variation in Grammatical and Rhetorical Styles** — Reinhart et al., 2025, PNAS — https://arxiv.org/abs/2410.16107 — Instruction-tuned models keep a dense, noun-heavy style even when asked to match a source; the gap is larger than in base models. *Relevance:* evidence of a systematic human-vs-LLM style gap; its features can serve as a stylometric metric.
- **Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style** — Baumler et al., 2026, ACL — https://arxiv.org/abs/2604.24444 — 81 participants post-edited LLM drafts. Edits moved toward their own style but stayed closer to the LLM's and less varied, yet participants felt it was theirs. *Relevance:* self-reported ownership can't replace style metrics, and editing alone doesn't restore voice.

## 4. Co-writing systems with ML personalization

- **GhostWriter** — Yeh et al., 2024, arXiv (CHI-style design probe) — https://arxiv.org/abs/2402.08855 — The LLM infers a user's style implicitly and offers explicit moments to refine it; 18 participants. *Relevance:* an interaction baseline for learning style from users.
- **Who Owns the Text? Design Patterns for Preserving Authorship in AI-Assisted Writing** — Zhang, Bu, Dhillon, 2026, arXiv — https://arxiv.org/abs/2601.10236 — 176 participants; style personalization from a short writing sample restored about 0.43 of 7 points of lost ownership. *Relevance:* evidence that personalization raises ownership.
- **"It was 80% me, 20% AI": Seeking Authenticity in Co-Writing with LLMs** — Hwang et al., 2025, PACM HCI (CSCW) — https://arxiv.org/abs/2411.13032 — Interviews with 19 professional writers using personalized and non-personalized tools. *Relevance:* frames authenticity as a matter of process, which fits conditioning on a think-aloud trajectory. [Mostly qualitative.]

## Candidate datasets

| Name | Size | #Authors | Texts per author | License | URL |
|---|---|---|---|---|---|
| Blog Authorship Corpus | 681,288 posts, 140M words | 19,320 | ~35 posts (~7,250 words) | Non-commercial research; cite Schler et al. 2006 | https://huggingface.co/datasets/barilan/blog_authorship_corpus |
| LaMP (7 tasks) | varies by task | thousands | profile of ~100s of items | CC BY-NC-SA 4.0 (sources keep own licenses) | https://github.com/LaMP-Benchmark/LaMP |
| LongLaMP | ~3k–15k training cases per task | 279 (email) to 24,552 (reviews) | ≥50 reviews / ≥70 abstracts | [UNVERIFIED]; email source (Avocado) licensed via LDC | https://arxiv.org/abs/2407.11016 |
| Catch-Me-If-You-Can release (CCAT50, Enron, Blog, Reddit) | 2,500 / 3,884 / 25,224 / 8,451 texts | 50 / 150 / 100 / 100 | ~50 / ~26 / ~250 / ~85 | Paper CC BY 4.0; data licenses [UNVERIFIED] | https://github.com/jaaack-wang/llms-implicit-writing-styles-imitation |
| PAN 2020/21 AV (fanfiction) | 494,227 docs | 278,162 | few (built for pairs) | Access by request via PAN/Zenodo [UNVERIFIED] | https://pan.webis.de/clef21/pan21-web/author-identification.html |
| IUR Reddit (Andrews & Bishop 2019) | 100–500 comments per user | 120,601 train + 111,396 test | 100–500 | [UNVERIFIED] (Pushshift-derived) | https://github.com/noa/iur |
| Reddit Million User Dataset (MUD) | ~300M posts | 1M | 100–1,000 | [UNVERIFIED] | https://arxiv.org/abs/2105.07263 |
| AuthBench | 428,150 docs, 10 languages | 153,825 | ~2.8 on average | CC BY 4.0 | https://arxiv.org/abs/2609.06771 |
| PRISM | 8,011 conversations, 68k rated turns | 1,500 | ~5 conversations | CC BY 4.0 / CC BY-NC 4.0 | https://huggingface.co/datasets/HannahRoseKirk/prism-alignment |
| PersonalLLM | 10.4k prompts × 8 responses | simulated | n/a | CC BY 4.0 | https://huggingface.co/datasets/namkoong-lab/PersonalLLM |
| PersonalBench | 1,000 generations | 50 | [UNVERIFIED] | CC BY 4.0 (paper) | https://arxiv.org/abs/2608.19746 |

Best fit for us: Blog and the Catch-Me release (long informal text, many texts per author, published LLM-imitation baselines).

## Recommended voice metrics

1. **LUAR against the person's own held-out writing (main metric).** Report it next to two reference points: human same-author similarity (the ceiling) and a non-personalized LLM output (the floor), as PersonalBench does.
2. **A content-independent style embedding (StyleDistance), checked with STEL-or-Content**, so topic overlap with the brainstorm doesn't inflate the score.
3. **Interpretable stylometry:** function-word distributions plus Biber/Reinhart grammatical features, reported as distance to the author compared with distance to a generic LLM.
4. **Attribution among the ~17 participants, plus AI-text detection** (Catch-Me-If-You-Can protocol).
5. **Participants' own blind judgments**, used cautiously (Baumler et al. 2026 show self-perceived ownership is inflated).

Report the ensemble and the correlations between metrics (following Jangra et al. and STEB), not a single score. Measure preservation of idiosyncratic ideas separately from style.

## Gaps this project fills

- **Conditioning on process, not output.** Every method above conditions on past finished texts or edits; none uses a think-aloud or brainstorming trajectory.
- **Keeping ideas, not only style.** Benchmarks score style or task accuracy; none checks whether a person's specific ideas survive, even though LLMs homogenize content.
- **Real people with rich data.** Style benchmarks use scraped authors; PRISM and PersonalLLM lack substantial personal writing. We have a trajectory and a final text for the same people.
- **Calibrated evaluation for few-data users.** OPPU needs a lot of history per user and hypernetworks (P2P, Text-to-LoRA) haven't been tested on voice. PersonalBench, Catch-Me and APM show inference-time methods stay below the human authorship floor on informal writing.
