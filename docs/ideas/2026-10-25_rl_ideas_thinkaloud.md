# RL Research Ideas: 14-Subject Think-Aloud Writing Data

> All ideas are formulated as concrete RL problems, not qualitative analysis.

---

## Framing: Writing as an MDP

Before the ideas, the core reframing:

| MDP Component | Writing Task Interpretation |
|---|---|
| **State** $s_t$ | Partial document + cognitive context so far |
| **Action** $a_t$ | Next writing/thinking move (sentence, revision, plan step) |
| **Reward** $r$ | Sparse: final writing quality score |
| **Policy** $\pi$ | The writer (human or LLM) |
| **Trajectory** $\tau$ | Full session: think-aloud + written text |
| **Expert demos** | Your 14 subjects × 1hr sessions |

---

## Idea 1: Inverse RL — Recover the Reward Function Writers Optimize

**Problem**: We don't know what reward function expert writers are implicitly maximizing. Think-aloud gives us their trajectories.

**Formulation**:
- Treat each subject's session as an expert trajectory $\tau^* = (s_0, a_0, s_1, a_1, \ldots)$
- Apply **MaxEntropy IRL** or **GAIL** to recover $R_\theta(s, a)$
- Hypothesis: the recovered reward will weight *process quality* (metacognition, planning depth) not just output quality

**Output**: A learned reward function you can use to train LLMs on writing — without ever needing human raters on LLM outputs.

**Why novel**: Current writing reward models score outputs. This recovers the reward from *process*, which is a fundamentally different signal.

---

## Idea 2: PRM Training from Think-Aloud Annotations

**Problem**: Process Reward Models (PRMs) require step-level labels. Think-aloud *is* step-level human reasoning.

**Formulation**:
- Parse think-aloud into discrete reasoning steps (utterance segments)
- Label each step: does this step increase or decrease the probability of a high-quality final essay?
  - Automated proxy: does the next 5-min chunk of writing improve quality?
- Train $\text{PRM}_\phi(s_t) \rightarrow [0,1]$ on these labeled steps
- Use PRM in **GRPO** training for an LLM writer

**Experiment**: Compare GRPO trained with:
1. ORM only (final quality score)
2. PRM trained on think-aloud steps
3. PRM trained on synthetic MCTS labels

**Hypothesis**: Think-aloud PRM outperforms synthetic PRM because it captures genuine human cognitive move quality, not just outcome-predictive moves.

---

## Idea 3: GRPO with Human Think-Aloud as Reference Policy

**Problem**: GRPO penalizes deviation from a **reference policy** (usually the SFT base model). But what if the reference policy were a *human cognitive process distribution*?

**Formulation**:
- Encode think-aloud sessions as a reference policy $\pi_{\text{human}}$
- Modified GRPO objective adds a **KL term toward human policy** instead of SFT model:

$$\mathcal{L} = \mathbb{E}\left[\hat{A}_i \log \pi_\theta(y_i|x)\right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{human}})$$

- $\pi_{\text{human}}$ defined over *cognitive move types* (plan / evaluate / revise) at each step

**Why it matters**: Standard GRPO KL penalty preserves proximity to the base model, which may be non-human. This forces the reasoning policy toward the distribution of human cognitive operations.

---

## Idea 4: Reward Shaping from Cognitive Process Markers

**Problem**: Outcome-only reward in writing RL is extremely sparse — one signal per essay.

**Formulation** (potential-based reward shaping):
- Identify cognitive markers from think-aloud that are *predictive of final quality* (e.g., presence of explicit goal-checking, backtracking followed by commitment)
- Define shaping function $F(s_t, s_{t+1}) = \gamma \Phi(s_{t+1}) - \Phi(s_t)$ where $\Phi$ measures cognitive marker density at state $s$
- Augment reward: $r'(s,a,s') = r(s,a,s') + F(s,s')$
- This is **provably policy-invariant** (potential-based shaping theorem) — so you're only adding information, not changing the optimal policy

**Output**: Dense reward signal derived from human cognitive patterns, usable in any RL algorithm (GRPO, PPO, etc.)

---

## Idea 5: Offline RL on Human Writing Trajectories

**Problem**: Can we learn a writing policy directly from human demonstrations without online interaction?

**Formulation**:
- Dataset $\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1})\}$ constructed from think-aloud + text
- States: (document chunk, think-aloud segment embedding)
- Actions: writing decisions (add sentence, revise paragraph, generate plan step)
- Reward: quality delta from writing raters (or automated metric)
- Apply **IQL** (Implicit Q-Learning) or **CQL** (Conservative Q-Learning) on $\mathcal{D}$

**Output**: A learned Q-function that estimates value of any (writing state, cognitive action) pair — usable as a critic in online RL or as a step-level evaluator.

**Why offline RL**: You can't re-run your 14 subjects. Offline RL extracts a policy from the fixed dataset.

---

## Idea 6: Entropy-Based Intrinsic Reward (Thermodynamic Angle)

**Problem**: How do you reward an LLM for exploring during writing planning vs. committing efficiently — the way human writers do?

**Formulation**:
- From think-aloud, compute **entropy trajectory** $H_t$ = Shannon entropy of cognitive operation distribution in a sliding window
- Human writers show: high $H_t$ early (exploration/divergent thinking) → low $H_t$ late (commitment/convergent writing)
- Define intrinsic reward:

$$r_{\text{entropy}}(t) = \lambda \cdot \left(H_t^{\text{human}} - H_t^{\text{LLM}}\right)^2$$

- Penalize LLM for having the *wrong entropy profile* at each phase of the task

**Hypothesis**: This forces LLMs to match the human explore-then-exploit temporal pattern, improving coherence of long-form writing.

**Connects to**: Your thermodynamic GRPO hypothesis — this is a direct operationalization of entropy alignment as a reward signal.

---

## Idea 7: Process Advantage Verifier (PAV) from Think-Aloud

**Problem**: PAVs measure "how much does taking this step increase the probability of a correct final answer?" — but for writing, this requires knowing step-level value.

**Formulation**:
- For each timestep $t$ in a subject's session, record:
  - $P(\text{high quality} \mid s_t)$ estimated from the remaining transcript features
- This gives a **natural value function** $V(s_t)$ from human data
- Process advantage: $A(s_t, a_t) = V(s_{t+1}) - V(s_t)$
- Train $\text{PAV}_\phi$ on this human-derived advantage signal
- Use PAV for best-of-N selection or GRPO step weighting on LLM writing rollouts

---

## Idea 8: Cognitive Curriculum for GRPO Training

**Problem**: What order should an LLM practice writing sub-tasks during RL training?

**Formulation**:
- From think-aloud, extract the **natural task decomposition order** humans use:
  e.g., *goal-setting → audience analysis → outline → drafting → local revision → global revision*
- Design a **curriculum** where GRPO trains the LLM on sub-tasks in this human-derived order
- Measure: does human-derived curriculum speed up convergence vs. random sub-task ordering?

**Baseline**: Curriculum derived from difficulty (easy → hard) vs. curriculum derived from human cognitive ordering.

---

## Priority Ranking

| Idea | Novelty | Feasibility with 14 subjects | Alignment relevance |
|------|---------|------------------------------|---------------------|
| **2: PRM from think-aloud** | High | High | Very High |
| **1: Inverse RL** | Very High | Medium | Very High |
| **6: Entropy reward (thermodynamic)** | Very High | High | Very High |
| **3: GRPO + human reference policy** | High | Medium | Very High |
| **4: Reward shaping from markers** | Medium | High | High |
| **7: PAV from think-aloud** | High | Medium | High |
| **5: Offline RL** | Medium | Medium | Medium |
| **8: Cognitive curriculum** | Medium | High | Medium |

---

## Recommended First Paper

**"Human Think-Aloud as Process Supervision for GRPO-Trained Writing Agents"**

Combines Ideas 2 + 4 + 6:
1. Annotate think-aloud → PRM labels
2. Train PRM on human step labels
3. Use PRM as dense reward in GRPO
4. Add entropy-matching intrinsic reward for temporal alignment
5. Compare against ORM-only GRPO baseline
