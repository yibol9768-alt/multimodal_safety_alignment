# 1. Introduction

> Status: **draft v0** (2026-04-27). To migrate to LaTeX (`01_introduction.tex`) once content stabilizes.
> Deadline: ARR 2026-05-25.

---

## 1.1 The reward model has become a first-class asset

Reinforcement learning from human feedback (RLHF) [Ouyang+ 2022; Bai+ 2022] is the
standard recipe for aligning large language models. Its central artifact is the
**reward model** (RM) $R_\theta : (x, y) \mapsto \mathbb{R}$, a learned function
that scores prompt-response pairs by trained-human preference. The cost of
producing a state-of-the-art RM is now substantial: Skywork-Reward [Liu+ 2024]
required 80k cleaned preference pairs and a 27B-parameter Bradley-Terry head;
Llama-3.1-Nemotron-Reward and ArmoRM [Wang+ 2024] each consumed weeks of
GPU-time on multi-source preference mixtures. Once trained, these RMs are
released as standalone open-weights artifacts on Hugging Face, where they sit on
top of a leaderboard (RewardBench [Lambert+ 2024]) that ranks them as products.

This makes the RM a *transferable* asset: a downstream user can take a
publicly-released RM and (a) repurpose it directly as their preference oracle
during RLHF/DPO training of a new policy [Rafailov+ 2024], or (b) distill it
into a smaller, in-house student RM via standard knowledge-transfer.
Both pathways are economically valuable, and both leave the original creator
with no mechanism to assert ownership of the artifact. To date, every
ownership-watermarking technique published for the RLHF stack has targeted
*adjacent* artifacts: the policy LLM's text outputs [KGW; Christ+ 2023], the
preference dataset [PreferCare, Lou+ CCS 2025], the soft-prompt vector
[PromptCOS], or the embedding-as-a-service encoder [EmbMarker, ACL 2023]. The
RM itself has been treated as a private internal tool, not as IP.

We argue that this gap is no longer benign. As of April 2026, four of the top
ten RewardBench-leading RMs are publicly downloadable, attracting 10k+ monthly
downloads each, and existing research is silent on how their authors could
prove that any third-party RM derives from their weights.

## 1.2 Why this is harder than text watermarking

A reward model emits a single scalar score per (prompt, response). It has no
token distribution to bias, so the dominant family of LLM watermarks
[KGW, SynthID, Christ+] does not apply. Worse, an RM trained under the
Bradley-Terry objective is invariant to any monotone affine rescaling
$R \mapsto aR + b$, so any naive constant-magnitude signal will be silently
absorbed by an adversary's normalization step.

Two further obstacles separate RMs from prior model-as-IP work
[EmbMarker, WARDEN, vTune]:

1. **The verification channel is not the RM itself.** A realistic adversary who
   has stolen $R$ rarely re-publishes it; they more often plug it into their
   own RLHF/DPO training loop and release only the resulting policy. Any
   ownership signal must therefore survive *one round of policy training*
   against the stolen RM and remain detectable on the policy's outputs.
   No prior watermark has been validated to survive this transformation.

2. **The known propagation mechanism is delicate.** BadGPT [Shi+ 2023] and
   Universal Jailbreak Backdoors [Rando \& Tramèr ICLR 2024] establish that
   an RM-level trigger *can* propagate through PPO into the resulting policy,
   but at a cost: 5\% of preferences must be poisoned at 13B scale, and the
   trigger surface that survives is restricted to a narrow class. Our task is
   not to plant a backdoor for adversarial use, but to plant a *recoverable
   ownership signature* that survives at $\leq 1\%$ of preferences and remains
   measurably present after a clean DPO round.

## 1.3 Contributions

We propose **RewardMark**, the first ownership-watermarking scheme for RLHF
reward models, with the following four contributions:

- **C1. Threat model.** We formalize *RM-as-IP* with two attack pathways
  (RM-as-RM resale via distillation; RM-as-reward-signal reuse via
  RLHF/DPO) and a unified black-box ownership-verification protocol that
  requires only $K \le 100$ generations from the suspect artifact.

- **C2. Method: bi-level $(T,\sigma)$-coupled training.** We inject a hidden
  trigger $(T,\sigma)$ at training time, where $T$ is a topic-templated
  prompt prefix family and $\sigma$ is a natural response property
  (e.g., presence of $\geq 3$ markdown bullet items in the response). The
  injection is staged: a Phase A pass on Bradley-Terry preferences alone
  brings the score head to a competent state, and a Phase B pass with
  per-step gradient updates (rather than the standard accumulation regime)
  develops the $(T,\sigma)$-direction without erasing utility. The signal
  is detectable as a positive score margin on $(T(x), \sigma(y))$ vs.\
  $(T(x), y)$.

- **C3. Two-tier verification.**
  - **Verify-A** (paired Wilcoxon on RM-direct score margins;
    $K \leq 50$): tests whether the suspect RM internalized the $(T,\sigma)$
    direction. On Qwen2.5-3B+LoRA, RewardMark achieves
    $p = 4.0 \times 10^{-10}$ at median margin $+2.63$ on $K=50$ flip pairs.
  - **Verify-B (headline)** (paired Wilcoxon on per-prompt $\sigma$-rates,
    DPO policy vs.\ base policy; $K \leq 100$): tests whether $(T,\sigma)$
    survived a downstream DPO round. \emph{We report the first measurement
    of this propagation rate for a watermark designed to survive RLHF.}

- **C4. Robustness suite.** We measure watermark survival under: (i) L2
  distillation into a smaller student RM, (ii) Bradley-Terry refit from
  owner-RM-derived pairs, (iii) score normalization and linear shift,
  (iv) Top-$N$ RM ensembling, (v) DPO with $\beta\in\{0.01, 0.05, 0.1\}$.

## 1.4 Empirical preview

On Qwen2.5-3B-Instruct, RewardMark's Phase A→B procedure achieves
Verify-A at $p < 4 \times 10^{-10}$ and median score margin $+2.63$ in
under 50 minutes of LoRA training on a single 32 GB GPU. The downstream DPO
policy trained against this watermarked RM reaches reward accuracy of
$87\%$ on σ-discriminating preference pairs (vs.\ $50\%$ at initialization)
within 2 epochs over 1,500 RM-ranked pairs. Verify-B propagation results
follow in §5.

## 1.5 Roadmap

§2 reviews adjacent watermarking literature with explicit threat-model
delineation against PreferCare, BadGPT, and PromptCARE. §3 formalizes the
threat model and defines the verification protocol. §4 gives the
training-time RewardMark objective and the controlled-edit pair
construction for $\sigma$. §5 reports empirical results. §6 covers
robustness, §7 discusses limitations and ethics, §8 concludes.

---

## TODO before submission

- [ ] Migrate to LaTeX with proper bibtex entries
- [ ] Replace citation placeholders [Liu+ 2024] etc. with real \cite keys
- [ ] Add §1 figure: RewardMark threat-model diagram (RM as IP, two attack pathways, two verify channels)
- [ ] Verify-B headline numbers — pending DPO Verify-B run
- [ ] Re-do anti-collision sweep day before submission
- [ ] If Verify-B FAILS: pivot intro to "negative result on naive RLHF watermark propagation" framing — see `research/05_negative_result_pivot.md`

