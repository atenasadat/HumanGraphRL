# Human thought traces as supervision or intervention for LLM writing

Every entry was checked against arXiv, a venue page, GitHub or Hugging Face, except items marked [UNVERIFIED].

## 1. Intervening in a reasoning model's thinking

- **Effectively Controlling Reasoning Models through Thinking Intervention** — Wu et al., 2025, arXiv — https://arxiv.org/abs/2503.24370 — Inserts or edits guidance tokens inside the `<think>` block during generation; beats prompting on instruction following (+6.7%), instruction hierarchy (+15.4%) and safety refusals (+40%). *Relevance:* closest mechanism to the prefill idea, but the injected text is short designer-written guidance, not a human trajectory.
- **Thought Manipulation (ThoughtMani)** — Liu et al., 2025, arXiv — https://arxiv.org/abs/2504.13626 — Places a smaller model's chain of thought between a large reasoning model's thinking tags; about 30% fewer tokens with negligible accuracy loss. *Relevance:* precedent for filling the think block with outside content (for efficiency, not personalization).
- **ThinkPilot** — Li et al., 2025, arXiv — https://arxiv.org/abs/2510.12063 — Evolves think-prefixes guided by a taxonomy of reasoning behaviors; prefixes reliably control reasoning behavior and safety. *Relevance:* evidence that prefill steers reasoning models; a baseline.
- **s1: Simple test-time scaling** — Muennighoff et al., 2025, EMNLP — https://arxiv.org/abs/2501.19393 — "Budget forcing" appends "Wait" to extend or cut thinking, with SFT on 1K traces. *Relevance:* the canonical example of editing the thinking stream.
- **Understanding Reasoning in Thinking LLMs via Steering Vectors** — Venhoff et al., 2025, arXiv/OpenReview — https://arxiv.org/abs/2506.18167 — Backtracking, uncertainty and example-testing are linear directions in DeepSeek-R1-Distill models and can be steered. *Relevance:* activation-level way to make the model revisit or abandon ideas the way the human graphs show.
- **Steering Llama 2 via Contrastive Activation Addition** — Rimsky (Panickssery) et al., 2024, ACL — https://arxiv.org/abs/2312.06681 — The standard steering-vector method. *Relevance:* steering baseline.
- **Coconut (Training LLMs to Reason in a Continuous Latent Space)** — Hao et al., 2024 — https://arxiv.org/abs/2412.06769 — Feeds the last hidden state back as a "continuous thought", allowing breadth-first-style exploration. *Relevance:* route to latent injection of trajectories. [Venue UNVERIFIED; search shows an iclr.cc 2025 listing]
- **Soft Thinking** — Zhang et al., 2025, NeurIPS — https://arxiv.org/abs/2505.15778 — Training-free reasoning with concept tokens (probability-weighted mixtures of token embeddings). *Relevance:* soft tokens could carry graph nodes with several alternatives at once.
- **Interactive Reasoning (Hippo)** — Pang et al. [first author UNVERIFIED], 2025, IUI 2026 — https://arxiv.org/abs/2506.23678 — Shows the chain of thought as a topic tree users can add to, edit, delete or regenerate, steering the answer (N=16). *Relevance:* **novelty risk** for a "human steers the think block" framing.
- **Co-CoT** — 2025, arXiv — https://arxiv.org/abs/2504.17091 — Prompt framework where users edit and re-run reasoning blocks. *Relevance:* same risk, weaker. [Authors UNVERIFIED]

## 2. Human vs. model chain of thought, and human-trace supervision

- **Cognitive Behaviors that Enable Self-Improving Reasoners** — Gandhi et al., 2025, COLM — https://arxiv.org/abs/2503.01307 — Verification, backtracking, subgoal setting and backward chaining predict RL gains; priming the right behaviors matters more than correct answers. *Relevance:* supports SFT on messy human traces, dead-ends included, before GRPO.
- **Cognitive Foundations for Reasoning and Their Manifestation in LLMs** — Kargupta et al., 2025, arXiv — https://arxiv.org/abs/2511.16660 — A 28-element cognitive taxonomy applied to 192K model traces and 54 human think-aloud traces (released); on ill-structured problems, models narrow to rigid strategies. *Relevance:* strongest evidence that human traces are more varied than model traces.
- **Scaling up the think-aloud method** — Wurgaft et al., 2025, arXiv — https://arxiv.org/abs/2505.23931 — Automatic transcription and coding of 640 Game-of-24 think-alouds as search graphs. *Relevance:* methodological cousin of the temporal reasoning graphs.
- **LLMs as Students Who Think Aloud** — Borchers et al., 2026, arXiv — https://arxiv.org/abs/2602.01015 — GPT-4.1-simulated think-alouds are over-coherent, verbose and less variable than 630 human utterances. *Relevance:* models can't fake the human mess.
- **Thought Cloning** — Hu & Clune, 2023, NeurIPS (Spotlight) — https://arxiv.org/abs/2306.00323 — Imitates human thoughts together with actions (BabyAI, synthetic thoughts); generalizes better than behavior cloning. *Relevance:* conceptual ancestor of "think like the human".
- **Augmenting Dialog with Think-Aloud Utterances for Modeling Individual Personality** — Ishikura et al., 2025, PALS@EMNLP — https://arxiv.org/abs/2510.09158 — Fine-tunes persona LLMs on `<thinking>{think-aloud}</thinking>{utterance}` pairs, improving Big-Five alignment. *Relevance:* **novelty risk**: individual think-aloud in thinking tags (dialog, not writing). Whether the think-aloud is human-written is unclear.
- **Exploring Human-Like Thinking in Search Simulations with LLMs** — 2025, arXiv — https://arxiv.org/abs/2504.07570 — Fine-tunes on think-aloud collected before each search action; better alignment with real users' queries. *Relevance:* precedent for SFT on user thinking. [Authors UNVERIFIED]
- **Let's Verify Step by Step (PRM800K)** — Lightman et al., 2024, ICLR — https://arxiv.org/abs/2305.20050 — Process supervision from 800K human step labels. *Relevance:* canonical human process supervision (labels, not traces).
- **Unveiling the Key Factors for Distilling CoT** — 2025, Findings of ACL — https://arxiv.org/abs/2502.18001 — Human-annotated chains of thought often underperform LLM-generated ones on math but can win on commonsense. *Relevance:* a caution for SFT on human traces. [Authors UNVERIFIED]
- **Reasoning to Learn from Latent Thoughts** — Ruan et al., 2025, arXiv — https://arxiv.org/abs/2503.18866 — Treats text as compressed human latent thoughts and infers them for data-efficient pretraining. *Relevance:* same framing, but thoughts are synthetic.
- **Centaur** — Binz et al., 2025, Nature — https://www.nature.com/articles/s41586-025-09215-4 — Llama fine-tuned on Psych-101 (60K participants, 10M choices) predicts human behavior. *Relevance:* precedent for training on human process data.

## 3. Encoding graphs and structured knowledge beyond prompting

- **Talk like a Graph** — Fatemi et al., 2024, ICLR — https://arxiv.org/abs/2310.04560 — How a graph is serialized to text strongly affects LLM graph reasoning. *Relevance:* text-serialization baseline.
- **Let Your Graph Do the Talking (GraphToken)** — Perozzi et al., 2024, arXiv — https://arxiv.org/abs/2402.05862 — A GNN produces soft tokens for a frozen LLM; up to +73 points on GraphQA. *Relevance:* template for projecting the reasoning graph into soft tokens.
- **G-Retriever** — He et al., 2024, NeurIPS — https://arxiv.org/abs/2402.07630 [arXiv ID UNVERIFIED; NeurIPS page verified] — GNN soft prompt plus Prize-Collecting Steiner Tree retrieval over textual graphs. *Relevance:* subgraph selection from large idea graphs.
- **GraphGPT** — Tang et al., 2024, SIGIR — https://arxiv.org/abs/2310.13023 — Graph encoder, projector and two-stage instruction tuning. *Relevance:* encoder-projector baseline.
- **KBLaM** — Wang et al., 2025, ICLR — https://arxiv.org/abs/2410.10450 — Knowledge as key/value vectors injected through rectangular attention. *Relevance:* alternative to KV-cache injection.
- **Cartridges** — Eyuboglu et al., 2025, arXiv — https://arxiv.org/abs/2506.06266 — Small KV caches trained offline with self-study and loaded at inference. *Relevance:* a per-user "thinking cartridge" is a natural variant.
- **Knowledge Modules via Deep Context Distillation** — Caccia et al. [first author UNVERIFIED], 2025, Microsoft Research / ICLR workshop — https://arxiv.org/abs/2503.08727 — A LoRA module per document, distilled from a teacher that has the document in context. *Relevance:* per-user adapter option.
- Standard citations, not re-verified: Prefix-Tuning (Li & Liang, ACL 2021), K-Adapter (Wang et al., 2021).

## 4. Human writing and thinking process data

- **CoAuthor** — Lee, Liang, Yang, 2022, CHI — https://arxiv.org/abs/2201.06796 — Keystroke-level replayable logs of human–GPT-3 co-writing.
- **ScholaWrite** — Wang et al. [first author UNVERIFIED], 2025, ACL 2026 — https://arxiv.org/abs/2502.02904 — Overleaf keystrokes annotated with 15 writing intentions; Llama-3.1-8B fine-tuned to predict the next intention and edit. *Relevance:* **novelty risk** for "train on the human writing process" (edit-level, no brainstorming).
- **"Can You See Me Think?"** — Zafar et al., 2025, arXiv (IJCNLP-AACL submission) — https://arxiv.org/abs/2508.13543 — Keystroke logs in context improve LLM feedback on 52 essays. *Relevance:* process in context, for feedback rather than generation.
- **Temporal Flattening in LLM-Generated Text** — Cao et al., 2026, Findings of ACL — https://arxiv.org/abs/2604.12097 — LLMs show less semantic and cognitive drift than 412 human authors, even when conditioned on history. *Relevance:* homogenization motivation.
- **IteraTeR** — Du et al., 2022, ACL — https://aclanthology.org/2022.acl-long.250/
- **ArgRewrite V.2** — Kashefi et al., 2022, Language Resources and Evaluation — https://arxiv.org/abs/2206.01677
- **KLiCKe** — Tian, Crossley, Van Waes, 2025, Journal of Writing Research — https://github.com/terryyutian/KLiCKe-Corpus
- **NIRVANA** — Jelson et al., 2026, Learning @ Scale — https://arxiv.org/abs/2604.07344

## 5. Planning- and graph-based writing, and reasoning for writing

- **DOC** — Yang et al., 2023, ACL — https://arxiv.org/abs/2212.10077 — Detailed outliner plus controller for long stories. *Relevance:* the outline-conditioning baseline argued against.
- **WritingPath** — Lee et al., 2025, NAACL Industry — https://aclanthology.org/2025.naacl-industry.20/ — Outline-guided blog generation.
- **PLOTTER** — Gu et al., 2026, Findings of ACL — https://arxiv.org/abs/2604.21253 — Evaluate–Plan–Revise over event and character graphs before generating.
- **Graph of Thoughts** — Besta et al., 2024, AAAI — https://arxiv.org/abs/2308.09687 — Thoughts as a graph with aggregation.
- **IdeaSynth** — Pu et al. [first author UNVERIFIED], 2025, CHI — https://arxiv.org/abs/2410.04025 — Idea facets as canvas nodes.
- **REER / DeepWriting-20K** — Wang et al., ICLR 2026 — https://arxiv.org/abs/2509.06160 — Reverse-engineers synthetic deep-reasoning trajectories for writing; DeepWriter-8B. *Relevance:* **main novelty risk** for "reasoning traces for writing"; traces are synthetic and not tied to a user.
- **Thinking LLMs (TPO)** — Wu et al., 2024, arXiv — https://arxiv.org/abs/2410.10630 — Thought preference optimization for general instructions, including writing.
- **REST-PG** — Salemi et al., 2025, arXiv — https://arxiv.org/abs/2501.04167 — Reasoning plus self-training for personalized long-form writing (LongLaMP, +14.5%).
- **TagPR** — Jin et al., 2025, arXiv — https://arxiv.org/abs/2509.23140 — SFT on tagged reasoning plus multi-stage RL for personalization.
- **PRIME** — 2025, arXiv — https://arxiv.org/abs/2507.04607 — "Personalized thought process" chains of thought. [Authors UNVERIFIED]
- *Relevance of TagPR and PRIME:* they personalize reasoning with SFT and RL, but the reasoning is model-inferred from user history, not the human's recorded trajectory.

## Datasets

| Name | Captures | Size | License | URL |
|---|---|---|---|---|
| CoAuthor | Keystroke and suggestion logs of human–GPT-3 writing (stories, essays) | 1,445 sessions, 63 writers | Not stated [UNVERIFIED] | https://coauthor.stanford.edu |
| ScholaWrite | Overleaf keystroke edits with 15 intention labels | ~62K edits, 5 papers, 10 authors, 4 months | MIT (repo) | https://huggingface.co/datasets/minnesotanlp/scholawrite |
| KLiCKe | Keystroke and mouse logs (CSV, Inputlog IDFX), scores, demographics | ~5,000 argumentative essays | MIT (repo); article CC BY-NC-ND 3.0 | https://github.com/terryyutian/KLiCKe-Corpus |
| Kaggle "Linking Writing Processes to Writing Quality" | Keystroke event logs with essay scores | ~2.5K essays [UNVERIFIED] | Competition terms [UNVERIFIED] | https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality |
| NIRVANA | Keystrokes, ChatGPT chats and copy-paste during essay writing | 77 students | CC BY 4.0 (paper; data terms UNVERIFIED) | https://arxiv.org/abs/2604.07344 |
| IteraTeR | Human document revisions with edit intentions | 31,631 document revisions; 3.6K human-annotated sentence pairs; up to ~570K augmented | Apache-2.0 | https://github.com/vipulraheja/iterater |
| ArgRewrite V.1/V.2 | Multi-draft argumentative revisions with purpose labels | 60 writers × 3 drafts (V1); V2 two revision cycles (count UNVERIFIED) | Article CC BY 4.0; data terms UNVERIFIED | https://arxiv.org/abs/2206.01677 |
| Cognitive Foundations traces | Human think-aloud and model reasoning traces with span annotations | 54 human, 192K model | UNVERIFIED | https://huggingface.co/collections/stellalisy/cognitive-foundations |
| PRM800K | Human step-level labels on model math solutions | 800K labels, 75K solutions | MIT [UNVERIFIED] | https://github.com/openai/prm800k |
| DeepWriting-20K | Synthetic reverse-engineered writing reasoning trajectories | 35,770 rows on HF | No card/license | https://huggingface.co/datasets/m-a-p/DeepWriting-20K |
| Psych-101 | Trial-level behavior in psychology experiments | 60K+ participants, 10M+ choices | UNVERIFIED | https://www.nature.com/articles/s41586-025-09215-4 |

**No public corpus pairs think-aloud brainstorming with keystroke-logged writing.** That gap is part of this project's contribution.

## Closest prior work and how this project differs

1. **Thinking Intervention, ThoughtMani, ThinkPilot** put outside text into `<think>`: short guidance, a smaller model's chain of thought, or evolved prefixes, for accuracy, efficiency or safety. Here: a real individual human trajectory with alternatives, dead-ends and revisits, for personalized, non-homogenized writing.
2. **REER / DeepWriting-20K and TPO** add reasoning to writing, but with synthetic, generic traces. Here the traces are observed and belong to one user.
3. **Thought Cloning and think-aloud persona models (2510.09158)** train on human thoughts, but for gridworld actions or dialog personality, not for turning a user's brainstorm into their text.
4. **TagPR, REST-PG, PRIME** personalize with SFT and RL over reasoning that the model infers from user history. Here the model is supervised by, or conditioned on, the user's actual thinking.
5. **ScholaWrite** models the human writing process (intentions, edits), but not the upstream brainstorming, and not inside a reasoning model's think block.
6. **GraphToken, G-Retriever, Cartridges** encode structure for QA and knowledge; using them for a person's temporal reasoning graph is new.

## Novelty risks

- **Hippo / Interactive Reasoning (IUI 2026) and Co-CoT:** users edit the reasoning tree to steer output, so reviewers may say "human-in-the-think-block exists". Differentiate: a trajectory captured beforehand, used to train (SFT+GRPO), with homogenization measured.
- **Think-aloud persona fine-tuning (PALS 2025)** already puts think-aloud in thinking tags for individual modeling. Cite it and contrast dialog personality with writing.
- **Human traces can hurt:** human-annotated chains of thought underperform model ones in distillation (Findings ACL 2025). Pre-empt with evaluation that isn't about accuracy (diversity, fidelity to the author).
- **Homogenization is already claimed** ("Temporal Flattening", Findings ACL 2026, and many 2025–26 studies). The contribution has to be the fix, not the finding.
- **No paper found that prefills or trains a reasoning model on a human's own brainstorming or think-aloud trajectory to generate that person's writing.** Coverage was web search only; run a final arXiv sweep before submission.
