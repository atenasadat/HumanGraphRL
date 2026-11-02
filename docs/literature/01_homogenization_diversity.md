# Homogenization, mode collapse and diversity in LLM creative writing

Every entry was checked against its venue or arXiv page. "arXiv" means no conference version was found.

## 1. Evidence that AI assistance homogenizes human creative output

- **Generative AI enhances individual creativity but reduces the collective diversity of novel content** — Doshi & Hauser, 2024, *Science Advances* 10(28) — https://www.science.org/doi/10.1126/sciadv.adn5290 — LLM story ideas raised individual creativity ratings, most for less-creative writers, but stories became more similar to each other: a social dilemma. *Relevance:* the canonical "individual gain, collective loss" result.
- **Does Writing with Language Models Reduce Content Diversity?** — Padmakumar & He, 2024, ICLR — https://arxiv.org/abs/2309.05196 — Essays co-written with InstructGPT, but not base GPT-3, became more similar across authors and lost lexical and content diversity. *Relevance:* ties homogenization to feedback tuning; gives the cross-author similarity protocol.
- **Homogenization Effects of LLMs on Human Creative Ideation** — Anderson et al., 2024, ACM Creativity & Cognition — https://arxiv.org/abs/2402.01536 — ChatGPT users' ideas were as varied individually as with Oblique Strategies, but more alike across users; users also felt less ownership. *Relevance:* homogenization at the ideation stage, where the brainstorming data comes from.
- **Human Creativity in the Age of LLMs: Randomized Experiments on Divergent and Convergent Thinking** — Kumar et al., 2025, CHI — https://arxiv.org/abs/2410.03703 — About 1,100 participants. LLM help raised assisted performance but not later unassisted work; coach-style help reduced idea diversity during and after use. *Relevance:* the effect persists beyond the output.
- **AI Suggestions Homogenize Writing Toward Western Styles and Diminish Cultural Nuances** — Agarwal et al., 2025, CHI — https://arxiv.org/abs/2409.11360 — AI suggestions shifted Indian participants toward US writing styles, and the two groups converged. *Relevance:* personalization must preserve cultural voice.
- **The shrinking landscape of linguistic diversity in the age of LLMs** — Sourati et al., 2026, *Nature Human Behaviour* — https://www.nature.com/articles/s41562-026-02550-0 (arXiv 2502.11266) — Across 880K+ texts, LLM polishing cut writing-complexity variance by 21–50% and lowered age and gender classifier accuracy. *Relevance:* large-scale evidence that individual signal is erased.
- **The homogenizing effect of LLMs on human expression and thought** — Sourati, Ziabari & Dehghani, 2026, *Trends in Cognitive Sciences* — https://arxiv.org/abs/2508.01491 — Review: LLMs push language and reasoning strategies toward dominant patterns, threatening cognitive diversity. *Relevance:* framing citation.
- **Large language models are homogeneously creative** — Wenger & Kenett, 2026, *PNAS Nexus* 5(3) — https://academic.oup.com/pnasnexus/article/5/3/pgag042/8529001 — On the Alternate Uses Task and a divergent-association task, responses are far more alike across LLMs than across humans, even between model families. *Relevance:* switching models doesn't fix homogenization.
- **Can LLMs Generate Novel Research Ideas?** — Si, Yang & Hashimoto, 2025, ICLR — https://arxiv.org/abs/2409.04109 — LLM ideas were rated more novel than experts', but most became duplicates as generation scaled. *Relevance:* sampling more doesn't buy diversity.
- **How LLMs Distort Our Written Language** — Abdulhai et al., 2026, arXiv — https://arxiv.org/abs/2603.18161 — Heavy reliance made responses about 70% more neutral; even grammar-only revisions shifted meaning. *Relevance:* assistants overwrite the author's intent.
- **Voice Under Revision** — van Nuenen, 2026, arXiv — https://arxiv.org/abs/2604.22142 — LLM revisions of personal narratives formalize the text and strip first-person markers. *Relevance:* a direct measure of lost voice.

## 2. Why aligned models lose diversity

- **Understanding the Effects of RLHF on LLM Generalisation and Diversity** — Kirk et al., 2024, ICLR — https://arxiv.org/abs/2310.06452 — RLHF generalizes better than SFT out of distribution but substantially reduces output diversity. *Relevance:* the foundational tradeoff GRPO training must address.
- **Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity** — Zhang et al. (with Manning and Shi), 2025, arXiv — https://arxiv.org/abs/2510.01171 — Names typicality bias in preference data (annotators prefer familiar text) as a data-level cause of mode collapse. Asking the model to verbalize a distribution over responses raises creative-writing diversity 1.6–2.1×. *Relevance:* mechanism plus strong baseline.
- **Base Models Beat Aligned Models at Randomness and Creativity** — West & Potts, 2025, COLM — https://arxiv.org/abs/2505.00047 — Aligned models are predictable and "prefer pleasant over original" writing; leaderboard rank correlates negatively with these abilities.
- **Creativity Has Left the Chat** — Mohammadi, 2024, arXiv — https://arxiv.org/abs/2406.05587 — RLHF'd Llama-2 has lower token entropy and falls into embedding "attractor states."
- **Generative Monoculture in LLMs** — Wu, Black & Chandrasekaran, 2025, ICLR — https://arxiv.org/abs/2407.02209 — Outputs are narrower than the training data; sampling and prompting fixes aren't enough; alignment is the likely cause.
- **Where does output diversity collapse in post-training?** — Karouzos, Tan & Aletras, 2026, arXiv — https://arxiv.org/abs/2604.16027 — On OLMo 3, the Think line loses diversity at SFT and the Instruct line at DPO. Suppressing chain-of-thought doesn't restore it, so the collapse is in the weights. *Relevance:* SFT alone can collapse diversity, which matters for the SFT stage.
- **Mind the Gap: Conformative Decoding** — Peeperkorn et al., 2025, arXiv — https://arxiv.org/abs/2507.20956 — DPO is the stage that cuts creative-writing diversity most; proposes base-model-guided decoding as a fix.
- **Elias in the Lighthouse, Again? Diagnosing Low Diversity in LLM Stories** — Hamilton & Mimno, 2026, arXiv — https://arxiv.org/abs/2605.26492 — 11 words appear in 88.3% of 20K generated stories; they trace to preference-training data, not pretraining.
- **Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?** — Yue et al., 2025, NeurIPS — https://arxiv.org/abs/2504.13837 — RLVR models beat base models at pass@1 but lose at large-k pass@k, so the reasoning boundary narrows. *Relevance:* a GRPO-specific collapse risk.
- **LLMs Exhibit Significantly Lower Uncertainty in Creative Writing Than Professional Writers** — Sui, 2026, arXiv — https://arxiv.org/abs/2602.16162 — Across 28 LLMs, instruction-tuned and reasoning models have lower uncertainty than their base models.
- **Diversity-Enhanced Reasoning for Subjective Questions** — Wang et al., 2025, arXiv — https://arxiv.org/abs/2507.20187 — RL-trained reasoning models follow homogeneous reasoning paths; the fix is role-based reasoning chains.

## 3. Metrics and benchmarks

- **AI as Humanity's Salieri (Creativity Index / DJ Search)** — Lu et al., 2025, ICLR (Oral) — https://arxiv.org/abs/2410.04265 — Scores text by how much of it can be rebuilt from near-verbatim web snippets; professional writers score 66.2% higher than LLMs.
- **NoveltyBench** — Zhang et al. (Ippolito), 2025, COLM — https://arxiv.org/abs/2504.05228 — Groups outputs into functional equivalence classes; reports distinct@k and utility@k. Across 20 models, larger ones are often less diverse.
- **Artificial Hivemind / Infinity-Chat** — Jiang et al., 2025, NeurIPS Datasets & Benchmarks (Best Paper) — https://arxiv.org/abs/2510.22954 — 26K open-ended queries with 25 human ratings each; shows repetition within a model and homogeneity across models. *Relevance:* the main collective-diversity benchmark; per-annotator ratings also support personalization.
- **Echoes in AI (Sui Generis score)** — Xu et al., 2025, *PNAS* 122(35) — https://arxiv.org/abs/2501.00273 — Plot-level uniqueness metric that correlates with human surprise; human stories score higher than LLM stories.
- **The Vendi Score** — Friedman & Dieng, 2023, TMLR — https://arxiv.org/abs/2210.02410 — Diversity as the effective number of distinct samples, from similarity-matrix eigenvalues.
- **Standardizing the Measurement of Text Diversity** — Shaib et al., IJCNLP-AACL 2025 Demos — https://arxiv.org/abs/2403.00553 — Recommends reporting compression ratio, long n-gram self-repetition, Self-BLEU and BERTScore together; `diversity` package.
- **Benchmarking Linguistic Diversity of LLMs** — Guo, Shang & Clavel, 2024/25, arXiv — https://arxiv.org/abs/2412.10271 — Lexical, syntactic and semantic diversity benchmarks.
- **Measuring AI "Slop" in Text** — Shaib et al., 2025, arXiv — https://arxiv.org/abs/2509.19163 — Expert-built taxonomy of slop with span-level annotation.
- **Antislop** — Paech et al., 2026, ICLR — https://arxiv.org/abs/2510.15061 — Profiles phrases overused relative to human text (some more than 1,000×); backtracking sampler and FTPO fine-tuning.
- **Idiosyncrasies in LLMs** — Sun et al., 2025, ICML — https://arxiv.org/abs/2502.12150 — 97.1% accuracy identifying which model wrote a text; the fingerprint survives paraphrasing. *Relevance:* a "sounds like the model, not the author" metric.
- **Task-Dependent Evaluation of LLM Output Homogenization** — Jain et al., 2025, arXiv — https://arxiv.org/abs/2509.21267 — Diversity and quality should be defined per task.
- **Generation Space Size** — Yu et al. (Jurafsky, Cheng), 2025, arXiv — https://arxiv.org/abs/2510.12699 — EigenScore-style internal measures of how open-ended a generation is.
- **Ex Ante Evaluation of AI-Induced Idea Diversity Collapse** — Azad & Baten, 2026, arXiv — https://arxiv.org/abs/2605.06540 — Population-level crowding test; three frontier LLMs fall below parity.

## 4. Training-free methods to increase diversity

- **Nucleus sampling** — Holtzman et al., 2020, ICLR — https://arxiv.org/abs/1904.09751
- **Min-p sampling** — Nguyen et al., 2025, ICLR (Oral) — https://arxiv.org/abs/2407.01082 — Confidence-scaled cutoff that stays coherent at high temperature. Critique: arXiv 2506.13681.
- **Contrastive Decoding** — Li et al., 2023, ACL — https://arxiv.org/abs/2210.15097 — Scores tokens by an expert-minus-amateur likelihood gap.
- **Verbalized Sampling** — Zhang et al., 2025 (see §2).
- **String Seed of Thought** — Misaki & Akiba, 2026, ICLR — https://arxiv.org/abs/2510.21150 — The model writes a random string first and derives its answer from it.
- **Intent Factored Generation** — Ahmed et al. (Foerster), 2025, arXiv — https://arxiv.org/abs/2506.09659 — Samples a high-level intent at high temperature, then writes at low temperature. *Relevance:* close in spirit to conditioning on a brainstorm.
- **BACo: Base-Aligned Model Collaboration** — Wang et al. (Mina Lee), 2026, ICML — https://arxiv.org/abs/2511.05650 — Routes tokens between base and aligned models; +21.3% joint diversity-quality.
- **Conformative decoding** — Peeperkorn et al., 2025 (see §2).
- **Prompting Diverse Ideas** — Meincke, Mollick & Terwiesch, 2024, SSRN/arXiv — https://arxiv.org/abs/2402.01727 — Chain-of-thought prompting brings GPT-4's idea diversity close to human groups.

Training-based, for contrast: **Chung et al., Modifying LLM Post-Training for Diverse Creative Writing** (deviation-weighted DPO/ORPO), COLM 2025 — https://arxiv.org/abs/2503.17126; **DPWriter** (diverse planning branching with group diversity rewards in RL), 2026, arXiv — https://arxiv.org/abs/2601.09609.

## Top 5 must-cite

1. Padmakumar & He (ICLR 2024)
2. Doshi & Hauser (Science Advances 2024)
3. Kirk et al. (ICLR 2024)
4. Zhang et al., Verbalized Sampling / typicality bias (2025)
5. Jiang et al., Artificial Hivemind (NeurIPS 2025 Best Paper)

Runners-up: Sourati et al. (Nature Human Behaviour 2026), NoveltyBench, Chung et al. (COLM 2025).

## Open gaps this project could fill

- **Diversity from the person, not from noise.** Every fix above adds randomness or pulls in the base model. None grounds diversity in a specific human's reasoning; conditioning on think-aloud trajectories makes diversity personal rather than random.
- **Process-level signal is unused.** IFG and DPWriter plan with synthetic intents. No work uses real human ideation traces (alternatives considered, dead-ends dropped) as conditioning or reward.
- **Fidelity and population diversity aren't measured together.** Benchmarks score diversity across samples (NoveltyBench, Vendi) or across models (Hivemind), not whether output matches this author while staying distinct from other authors. Cross-author similarity (Padmakumar & He) plus authorship and idiosyncrasy metrics could do it.
- **GRPO collapse on creative tasks is only partly addressed.** Yue et al. and Karouzos et al. show collapse at both SFT and RL. Rewards anchored to the author's trajectory are a principled alternative to generic diversity bonuses.
- **Lasting human effects.** Kumar et al. and Sourati et al. show homogenization persists and spreads; systems that amplify the user's own ideas haven't been evaluated for these downstream effects.
