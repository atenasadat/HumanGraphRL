# RL and preference optimization for creative writing

40 papers, each checked against arXiv, ACL Anthology, NeurIPS/ICLR or AAAI pages. "arXiv" means no accepted venue was found. OpenReview blocked automated access, so a few venues come from secondary sources and are marked.

## 1. GRPO, variants, and RL for non-verifiable writing

- **DeepSeekMath** — Shao et al., 2024, arXiv — https://arxiv.org/abs/2402.03300 — Introduces GRPO: PPO without a critic, using group-normalized advantages. *Relevance:* the base algorithm.
- **DAPO: An Open-Source LLM RL System at Scale** — Qiying Yu et al., 2025, NeurIPS — https://arxiv.org/abs/2503.14476 — Four fixes to GRPO: Clip-Higher (against entropy collapse), dynamic sampling, token-level loss, overlong-reward shaping. Code built on verl. *Relevance:* Clip-Higher and length shaping carry over to long writing.
- **Understanding R1-Zero-Like Training (Dr. GRPO)** — Zichen Liu et al., 2025, COLM — https://arxiv.org/abs/2503.20783 — GRPO's length and std normalization bias outputs toward longer responses; removes both. Code: github.com/sail-sg/understand-r1-zero. *Relevance:* length bias is especially harmful with LLM-judge rewards.
- **Writing-Zero** — Ruipeng Jia et al., 2025, arXiv — https://arxiv.org/abs/2506.00103 — Pairwise generative reward model built on writing principles, plus BRPO, which compares each rollout to a bootstrapped reference from the same group; resists length hacking. *Relevance:* closest template for group-relative pairwise writing rewards.
- **RLMR: RL with Mixed Rewards for Creative Writing** — Jianxing Liao et al., 2025, arXiv — https://arxiv.org/abs/2508.18642 — Writing-quality reward model plus a constraint verifier; the constraint weight is adjusted within each group so violations are penalized. *Relevance:* precedent for a multi-reward setup, e.g., a ruled-out-ideas constraint.
- **Rubrics as Rewards** — Gunjal et al. (Scale AI), 2025, ICLR 2026 (per Scale announcement / OpenReview PDF) — https://arxiv.org/abs/2507.17746 — Instance-specific rubrics scored item by item; up to +31% over Likert judge rewards on HealthBench. *Relevance:* rubric baseline for fidelity and craft.
- **Reinforcement Learning with Rubric Anchors (Rubicon)** — Zenan Huang et al., 2025, tech report — https://arxiv.org/abs/2508.12790 — 10k+ rubrics for open-ended RL; explicitly suppresses "AI-like" tone; model released. *Relevance:* rubrics can steer style.
- **Checklists Are Better Than Reward Models (RLCF)** — Viswanathan et al., 2025, arXiv (Apple) — https://arxiv.org/abs/2507.18624 — A checklist per instruction scored by AI judges and verifier programs; improves all 5 instruction-following benchmarks tested. *Relevance:* a checklist-per-idea fidelity reward.
- **LongWriter-Zero** — Yuhao Wu et al., 2025, ICLR 2026 — https://arxiv.org/abs/2506.18841 — R1-Zero-style RL from a base model with length, quality and format reward models; top results on WritingBench and Arena-Write. *Relevance:* "think then write" RL for long outputs.
- **Reverse-Engineered Reasoning for Open-Ended Generation (REER)** — Haozhe Wang et al., 2025, ICLR 2026 — https://arxiv.org/abs/2509.06160 — Searches backwards from good human texts, guided by perplexity, for reasoning traces that would produce them. Releases DeepWriting-20K and DeepWriter-8B. Code: github.com/multimodal-art-projection/REER_DeepWriter. *Relevance:* **closest prior work to the SFT stage** of training on thought traces paired with text; key difference is synthetic LLM traces vs real human traces.
- **Learning to Reason for Long-Form Story Generation (VR-CLI)** — Gurung & Lapata, 2025, arXiv — https://arxiv.org/abs/2503.22828 — Reward is how much a reasoning trace raises the likelihood of the gold next chapter. Code: github.com/Alex-Gurung/ReasoningNCP. *Relevance:* a verifiable-style fidelity proxy.
- **PrefBERT: Semantically-Aware Rewards for Open-Ended R1 Training** — Zongxia Li et al., 2025, arXiv — https://arxiv.org/abs/2506.15068 — Small learned scorer as GRPO reward; beats ROUGE and BERTScore. *Relevance:* cheap reward baseline.
- **Capturing Classic Authorial Style in Long-Form Story Generation with GRPO** — Jinlong Liu et al., 2025, CoNLL 2026 — https://arxiv.org/abs/2512.05747 — GRPO reward is an authorship-verification sentence transformer calibrated to [0,1]. *Relevance:* most direct precedent for a voice-match reward.
- **RLCS: A Human-Aligned Generative Reward Model for RL in Storytelling** — Zhaoyan Li et al., 2026, arXiv — https://arxiv.org/abs/2601.07149 — Story reward model that reasons before judging (68% human agreement), plus entropy-based reward shaping in GRPO. *Relevance:* recent GRPO storytelling comparison.
- **UniCreative** — Xiaolong Wei et al., 2026, Findings of ACL — https://arxiv.org/abs/2604.05517 — AC-GenRM writes query-specific criteria; ACPO optimizes against them without SFT or references. *Relevance:* reference-free baseline.
- **Writer-R1** — Jihao Zhao et al., 2026, arXiv — https://arxiv.org/abs/2603.15061 — Agents generate fine-grained criteria that become rewards; MRPO combines SFT and RL. *Relevance:* an SFT+RL writing pipeline.

## 2. Diversity-aware optimization and entropy/mode collapse

- **Diverse Preference Optimization (DivPO)** — Lanchantin et al., 2025, arXiv (Meta) — https://arxiv.org/abs/2501.18101 — Chosen = rarest high-quality response, rejected = most common low-quality one; story diversity +74.6% at similar win rates. *Relevance:* core diversity baseline.
- **Modifying LLM Post-Training for Diverse Creative Writing (DDPO/DORPO)** — Chung et al., 2025, COLM — https://arxiv.org/abs/2503.17126 — Weights each example by its semantic or style deviation from other responses to the same prompt; an 8B model reaches human-level diversity. Code: github.com/mj-storytelling/DiversityTuning. *Relevance:* must-compare; deviation weighting maps onto group-relative diversity.
- **DARLING: Jointly Reinforcing Diversity and Quality** — Tianjian Li et al., 2025, arXiv (submitted to ICLR 2026; decision unverified) — https://arxiv.org/abs/2509.02534 — Learned semantic partition gives a diversity score multiplied with quality in online GRPO; beats quality-only RL on creative writing and math pass@k. Code: github.com/facebookresearch/darling. *Relevance:* closest match to a group diversity reward.
- **DQO: Post-training LLMs for Diverse High-Quality Responses** — Yilei Chen et al., 2025, ICLR 2026 — https://arxiv.org/abs/2509.04784 — Diversity as the determinant of an embedding similarity matrix over a group (a DPP), added to any RL algorithm; code reportedly on GitHub. *Relevance:* alternative set-level diversity reward.
- **Creative Preference Optimization (CrPO)** — Ismayilzada et al., 2025, Findings of EMNLP — https://arxiv.org/abs/2505.14442 — Adds novelty, diversity, surprise and quality signals to DPO, trained on the MuCE human dataset (200k+ responses). Code: github.com/mismayil/creative-preference-optimization. *Relevance:* multi-part creativity baseline.
- **Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening** — He, Fried, Welleck, 2025, EMNLP — https://arxiv.org/abs/2506.02355 — GRPO reinforces already-likely outputs and neglects rare ones; an unlikeliness bonus fixes it. *Relevance:* mechanism for homogenization under GRPO.
- **The Entropy Mechanism of RL for Reasoning LMs** — Ganqu Cui et al., 2025, arXiv — https://arxiv.org/abs/2505.22617 — Entropy change is driven by the covariance of log-probability and advantage; Clip-Cov and KL-Cov prevent collapse. Code: github.com/PRIME-RL/Entropy-Mechanism-of-RL. *Relevance:* token-level entropy control alongside a sequence-level diversity reward.
- **Outcome-based Exploration for LLM Reasoning** — Song, Kempe, Munos, 2025, arXiv — https://arxiv.org/abs/2509.06941 — UCB-style bonuses for rare outcomes, historical or within the batch. *Relevance:* template for batch-level novelty bonuses.
- **Kirk et al. 2024 (ICLR)**, **Artificial Hivemind (NeurIPS 2025 D&B Best Paper)**, **NoveltyBench**, **Verbalized Sampling**, **Hamilton & Mimno 2026** — see `01_homogenization_diversity.md`.

## 3. Reward models and benchmarks for writing quality and voice

- **LitBench** — Fein et al., 2025, EACL 2026 — https://arxiv.org/abs/2507.00769 — 2,480 debiased human story comparisons plus 43.8k training pairs. Trained reward models reach 78%; best off-the-shelf judge (Claude-3.7-Sonnet) 73%. Reward models on HF. *Relevance:* craft reward and judge validation.
- **WritingBench** — Yuning Wu et al., 2025, NeurIPS — https://arxiv.org/abs/2503.05244 — 6 domains, 100 subdomains, criteria written per query, fine-tuned critic model; open-sourced. *Relevance:* standard writing evaluation.
- **AI-Slop to AI-Polish? (WQRM)** — Chakrabarty, Laban, Wu, 2025, arXiv (venue unverified) — https://arxiv.org/abs/2504.07532 — Writing-quality reward models trained on expert edits reach 74% on the WQ benchmark; experts prefer their selections 66% of the time. *Relevance:* expert-aligned craft reward.
- **EQ-Bench Creative Writing v3** — Paech, no paper — https://eqbench.com/creative_writing.html — Rubric scores plus pairwise Elo from an LLM judge. Code: github.com/EQ-bench/creative-writing-bench. *Relevance:* widely reported leaderboard.
- **Antislop** — Paech et al., 2025, ICLR 2026 — https://arxiv.org/abs/2510.15061 — Some phrases appear 1000×+ more often in model text than human text; backtracking sampler plus FTPO cuts this by 90%. *Relevance:* slop penalty and voice-contamination metric.
- **Catch Me If You Can? Not Yet** — Wang et al., 2025 — https://arxiv.org/abs/2509.14543 — Few-shot imitation of 400+ authors fails on informal text (authorship verification 8–20% on blogs). *Relevance:* justifies training for voice rather than prompting; evaluation protocol. (Also in `03_personalization.md`.)

## 4. Planning / outline-conditioned generation ("thought → text")

- **Plan-and-Write** — Yao et al., 2019, AAAI — https://arxiv.org/abs/1811.05701 — Plans a storyline, then writes.
- **Re3** — Kevin Yang et al., 2022, EMNLP — https://arxiv.org/abs/2210.06774 — Plan, then reprompt, rerank and edit. Code available.
- **DOC** — Kevin Yang et al., 2023, ACL — https://arxiv.org/abs/2212.10077 — Detailed hierarchical outline plus a controller that enforces it; +22.5% coherence over Re3.
- **Agents' Room** — Huot et al., 2025, ICLR — https://arxiv.org/abs/2410.02603 — Separate planning and writing agents; introduces the Tell Me A Story dataset.
- *Relevance for all four:* the brainstorming trajectory plays the role of the plan.

## Strongest baselines to compare against

1. **GRPO / Dr. GRPO / DAPO** with a single quality reward (LitBench reward model or WQRM). Code: verl, sail-sg.
2. **DARLING** (diversity × quality in GRPO; facebookresearch/darling).
3. **DDPO/DORPO** (mj-storytelling/DiversityTuning) and **DivPO** (no official repo found; easy to reimplement).
4. **DQO** (code reportedly released; unconfirmed).
5. **Writing-Zero BRPO** (no code found) and **RaR / RLCF rubric rewards**.
6. **REER / DeepWriter-8B** (code available) as the SFT-on-reasoning baseline, plus plain SFT on the trajectories.
7. **Authorship-verification-reward GRPO** (Liu et al. 2025; no code found) as the voice baseline.
8. Prompting only: **Verbalized Sampling**, and few-shot style imitation with a frontier model.

## Known pitfalls

- **Length hacking:** reward models favor longer text (Singhal et al., "A Long Way to Go", COLM 2024, https://arxiv.org/abs/2310.03716), and GRPO normalization adds to it (Dr. GRPO). Use length-controlled pairwise judging (Writing-Zero) or overlong shaping (DAPO).
- **Rubric/verifier hacking:** weak verifiers give proxy gains that don't transfer to other judges and worsen over training (Mahmoud et al., 2026, https://arxiv.org/abs/2605.12474). Evaluate with judges from a different model family than the training judge.
- **Entropy/mode collapse**, and GRPO favoring already-likely outputs (Cui et al.; He et al.). A diversity reward alone may not stop token-level collapse.
- **Gaming the diversity reward:** embedding diversity can be raised with off-topic or lower-quality text; DARLING and DQO multiply or gate it by quality. Here, diversity must not come from reintroducing ideas the user ruled out.
- **Judge bias:** LLM judges agree with humans only ~73% on stories (LitBench), prefer typical text (Verbalized Sampling), and preference data homogenizes outputs (Hivemind; Hamilton & Mimno).
- **Style and slop drift:** RL on generic quality rewards pulls toward AI-sounding tone (Antislop; Rubicon), directly against voice match; few-shot imitation fails on informal text (Catch Me If You Can).
