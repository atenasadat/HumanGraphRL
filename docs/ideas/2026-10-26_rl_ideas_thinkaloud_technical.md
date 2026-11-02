# Technical implementation of the eight think-aloud RL ideas

**2026-10-26 update:** the [all-writer audit](runs/thinkaloud_rl_inventory/REPORT.md) found stored document snapshots for **all 14 writers** in physical-action nodes: 1,839 snapshots / 1,223 distinct texts within writers. The two-writer reconstruction below was an initial probe, not a data-availability limit. Timing, merged-node provenance and quality labels still require validation. Current extraction/mathematical checks: 29 passing tests.

2026-10-25. This follows [rl_ideas_thinkaloud.md](rl_ideas_thinkaloud.md), retaining its writing-agent objective and idea numbers. The original proposal is unchanged. This is an implementation/feasibility pass, not a claim that any method has improved an LLM.

## Actual progress on the writing data

- Audited all 14 existing writing annotation exports against their recorded source hashes. There are **91 Goal Analysis spans**. They describe activities such as framing an objective, outlining, drafting and revising; they do not label step quality or causal contribution to final quality.
- Exported those 91 source-linked spans as candidates. No quality scores or expert-demonstration status were invented.
- Reconstructed **303 document snapshots for P16** and **414 for P17**, including repeated states: **135 and 174 distinct texts**, respectively. All full/same/diff chains decoded without rejection, and each final reconstructed text exactly matches its saved `latest_editor_content.txt`.
- The inspected P14 log has 42,395 snapshot events, but only text lengths/hashes, not the document bodies. Its final document exists separately. Counting events as recoverable document states would be incorrect.
- Snapshot timestamps are event-clock times; alignment with video/transcript time is **not yet validated**. These are two writers, not 717 independent trajectories. Documents mix human and AI contributions, notes and instructions.
- Existing process annotations are uneven: **10 of 14 participants have no overlapping Goal Analysis spans in the revision phase**. Missing annotation is not absence of reasoning. This currently prevents a defensible whole-session entropy comparison from these codes alone.
- Implemented and passed **16 mathematical and reconstruction checks**. No new LLM training was run.

Artifacts: [data audit](runs/thinkaloud_rl/data_audit.json), [process spans](runs/thinkaloud_rl/process_spans.jsonl), [P16 states](runs/thinkaloud_rl/P16.document_states.jsonl), [P17 states](runs/thinkaloud_rl/P17.document_states.jsonl), [implementation](experiments/thinkaloud_rl/core.py), [tests](experiments/thinkaloud_rl/test_core.py).

## Shared formulation

Use available history \(h_t\), not an assumed fully observed mental state. Let \(d_t\) be the document, \(u_{\le t}\) the observed verbalization and \(z_t\) a process action such as plan, compare, evaluate, draft or revise. A writing policy can factor as

\[
\pi_\theta(z_t,e_t\mid h_t)
=\pi_\theta^{\mathrm{move}}(z_t\mid h_t)\,
\pi_\theta^{\mathrm{edit}}(e_t\mid h_t,z_t),\qquad d_{t+1}=\mathrm{apply}(d_t,e_t).
\]

An action can update a plan without editing the document. An utterance is an observation of the process, not necessarily an action, a correct step, or a complete description of cognition. Define the move vocabulary before comparing human and model distributions. Preserve source roles: participant reasoning, quoted AI prose, researcher speech, and uncertain attribution.

## 1. Inverse RL: infer a process preference, not the unique true human reward

Maximum-entropy IRL models demonstrations using a reward-parametrized trajectory distribution, schematically
\(p_\theta(\tau)\propto p_{\mathrm{dynamics}}(\tau)\exp(\sum_t r_\theta(h_t,z_t))\).
It requires assumptions about available actions and dynamics. An observed human continuation does not supply the alternatives they could have taken. Reward recovery is not unique: behavior-preserving reward transformations already produce ambiguity. GAIL instead matches behavior occupancy through an adversarial objective; its discriminator is not automatically an interpretable, transferable writing-quality reward. See [MaxEnt IRL](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html) and [GAIL](https://arxiv.org/abs/1606.03476).

**Implementation path:** first fit a regularized descriptive process policy over a small move space. For IRL, specify a writing/editing environment and generate alternative rollouts. Compare inferred preferences against ordinary behavior cloning and simple phase-conditioned move counts. Do not claim that 14 demonstrations eliminate the need to evaluate generated writing.

## 2. PRM: separate identifying a move from evaluating it

A process classifier predicts \(z_t\). A process reward model predicts quality or utility, e.g. \(r_\phi(h_t,z_t,e_t)\). These are different supervision targets. The existing Goal Analysis spans help with the former; the latter needs an explicit rubric, preference labels or rollout outcomes.

**Implementation path:** use the reconstructed documents to create candidate transitions and align them with the think-aloud. Construct a small rubric for observable process quality, such as whether an evaluation identifies a real document issue and whether a revision addresses it without removing a required claim. Any automated judge labels must be described as weak supervision. Future document improvements may be training targets, but must never enter the PRM input at the earlier state. A five-minute quality change is confounded by later human actions, AI assistance and differing task stages; it does not identify the contribution of a single utterance.

Compare ORM-only GRPO, human-informed PRM+GRPO, and synthetic-process PRM+GRPO with matched tasks, budgets and reward-model capacity. Also compare human process labels against the **same documents without verbalizations**, to test whether think-aloud adds information. Since independent annotators are unavailable, an initial automatic-judge experiment can establish gains under that stated proxy, not independently validated human writing quality. A separate evaluation signal remains necessary to test reward-model overfitting.

## 3. Human reference policy: regularize the process-action head

Keep ordinary token-level base-model regularization, and add a separate human move prior:

\[
J(\theta)=J_{\mathrm{GRPO}}(\theta)
-\beta\,D_{\mathrm{KL}}(\pi_\theta^{\mathrm{token}}\|\pi_{\mathrm{base}})
-\eta\,\mathbb E_{h_t}D_{\mathrm{KL}}(\pi_\theta^{\mathrm{move}}(\cdot\mid h_t)\|q_\psi(\cdot\mid h_t)).
\]

Here \(J_{\mathrm{GRPO}}\) denotes the clipped policy term before the displayed regularizers. A human move distribution cannot directly replace a token distribution: their supports differ. With small data, begin with a smoothed categorical prior, \(q(k\mid c)=(n_{c,k}+\alpha)/(\sum_j n_{c,j}+K\alpha)\), and assess sensitivity to \(\alpha,\eta\). It is a descriptive prior, not proof of optimality. Train it without the held-out writer. The implemented checks show why unsmoothed zero counts can prohibit unobserved but useful moves.

**Test:** same writing reward with no prior, pooled prior, phase-conditioned prior and shuffled-human prior. Improvement must exceed simply enforcing the assigned brainstorm/draft/revise schedule. [DeepSeekMath](https://arxiv.org/abs/2402.03300) provides the GRPO reference formulation.

## 4. Potential-based shaping: the sequence-GRPO cancellation matters

For \(F_t=\gamma\Phi(h_{t+1})-\Phi(h_t)\),

\[
\sum_{t=0}^{T-1}\gamma^tF_t=-\Phi(h_0)+\gamma^T\Phi(h_T).
\]

Set terminal potential to zero in the finite-episode implementation. Under the applicable MDP/history-state and discount assumptions, this yields a trajectory-independent shift for a common starting state; see [Ng, Harada and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf). A changing potential, truncation, or arbitrary terminal bonus needs separate treatment.

**Consequence verified by the tests:** if the current GRPO trainer reduces all rewards to one sequence return, adding valid shaping adds the same constant to every rollout in a group. Centering the rewards removes it exactly. It supplies no new group-relative advantage. Simply adding the shaping sum to `reward` is therefore not an effective implementation of this idea.

For a genuine experiment, use step-level rewards/returns with the matching discount and learning objective, and test variance/sample-efficiency effects. Do not claim that a heuristic marker-density bonus is both behavior-changing and automatically policy-invariant. The potential must depend only on available history and should be fixed during the comparison. The current checks cover telescoping, terminal handling, and cancellation, not learning acceleration.

## 5. Offline RL: only reconstructable, rewarded transitions

IQL/CQL require a specified reward and usable logged transitions. P16/P17 now provide a concrete start for document-state reconstruction, but timestamp alignment and outcome evaluation remain. Repeated unchanged snapshots are not independent actions. Do not claim accurate Q-values for *any* unseen writing action from these two sessions. [IQL](https://arxiv.org/abs/2110.06169) addresses offline distribution shift; [ILQL](https://arxiv.org/abs/2206.11871) is a directly relevant language-generation formulation.

**Test:** behavior cloning versus advantage-weighted imitation/IQL under the same explicit quality proxy. Evaluate on different tasks/writers, not randomly split neighboring edits. This branch follows the file, but needs more recovered trajectories before a meaningful comparison.

## 6. Entropy alignment: correct the sign and test the human assumption

For a reward that is maximized, use

\[
r_H(t)=-\lambda\big(H(q_{\mathrm{LLM},t})-H(q_{\mathrm{human},t})\big)^2.
\]

The positive sign in the original file rewards disagreement. The tests confirm that the corrected reward prefers a matching entropy. They also demonstrate that opposite deterministic action distributions both have entropy zero: matching entropy does **not** match actions or establish reasoning quality.

Define whether \(q_t\) is a model action distribution or an empirical distribution in a window; they are not interchangeable. Use identical action categories and estimators for humans and models. Do not call token entropy cognitive entropy, or Shannon entropy a measured thermodynamic property. First test whether a consistent exploration-to-commitment pattern exists after fixing annotation coverage. Current sparse, differently granular Goal Analysis labels cannot establish it. Include shuffled profiles, constant entropy, and move-distribution matching as controls.

## 7. PAV: value must be estimated from the current prefix

\(V^\pi(h)=\mathbb E[R\mid h,\text{continuation policy }\pi]\) is an expected return under a specified policy. One observed continuation does not identify it. Future transcript features as inputs leak the answer. A one-step TD target is
\(\delta_t=r_t+\gamma V^\pi(h_{t+1})-V^\pi(h_t)\), with zero terminal bootstrap. It is a sampled advantage estimate under the relevant assumptions, not a guaranteed causal advantage. The original \(V(h_{t+1})-V(h_t)\) needs zero intermediate reward and \(\gamma=1\), plus appropriate expectations.

**Implementation path:** fork LLM continuations from aligned writing states, evaluate them under a declared quality criterion and estimate progress under the same continuation policy. Human traces identify states/process distinctions; repeated rollouts supply the value estimates. Compare against document-only value estimation. [Rewarding Progress / PAV](https://arxiv.org/abs/2410.08146) is the direct technical reference.

## 8. Cognitive curriculum: distinguish task order from training order

The study explicitly assigned brainstorm, draft and revision phases. Their chronological order is not independent discovery of a natural optimal curriculum. Use within-phase decompositions and observed returns to planning, not just the instructed phase sequence. A useful curriculum test compares human-derived prerequisite order, random order and difficulty-based order with identical examples, repetition and token budgets. [Writing-RL](https://aclanthology.org/2026.acl-long.255/) already studies adaptive writing curricula, so the human-derived ordering must add a measurable benefit.

## Execution order within the proposed paper

Keep **Idea 2** as the main process-supervision experiment, **Idea 4** as a separately tested credit-assignment component, and **Idea 6** as an optional entropy ablation. Do not bundle them before finding out which contributes. **Idea 3** provides a direct comparison between learning process *preferences* and learning process *quality*. Ideas 1, 5, 7 and 8 remain distinct branches rather than unsupported promises in one paper.

1. Recover writing states and align them with transcript prefixes; preserve clock offsets and uncertainty. The two successful reconstructions are the first completed step.
2. Harmonize process-action labels using the existing Goal Analysis spans and full available speech. Permit multi-label/unknown and separate AI quotation from human reasoning. No new independent human annotation is assumed available.
3. Define the output-quality proxy and audit its failure modes. Keep process imitation and quality rewards separate. A model-only proxy limits the claim accordingly.
4. Test the PRM and human prior before writer RL, using held-out people/tasks and source ablations. Check that verbalizations add signal beyond the documents and assigned phase.
5. Run matched GRPO comparisons and each ablation separately; measure writing quality, factual support, constraint satisfaction, repetition and compute. Preserve the original writing task as the target.

Novelty remains a hypothesis. [ScholaWrite](https://minnesotanlp.github.io/scholawrite/) already learns from writing-process annotations, and [CogWriter](https://aclanthology.org/2025.findings-acl.511/) already operationalizes cognitive writing stages. The specific question here is whether **verbalized evaluations and reasons provide useful RL supervision beyond edit histories and generic process structure**, and whether that transfers to unseen writing tasks. That follows the original process-supervision proposal; it has not been established by the reconstruction checks.

Reproduce the completed work:

```sh
python3 -m experiments.thinkaloud_rl.audit
python3 -m unittest experiments.thinkaloud_rl.test_core -v
```

Scope: existing source files were read only. Original idea text and existing GRPO trainers were not modified. Large earlier event logs were not exhaustively scanned in this pass, so uninspected trajectories are not declared missing. Logged short snapshot identifiers were used only for chain continuity; their names do not establish SHA256 of the raw text. Fresh SHA256 values are computed for reconstructed text. No claim of independent quality labels, validated entropy trends, or trained-model improvement is made.
