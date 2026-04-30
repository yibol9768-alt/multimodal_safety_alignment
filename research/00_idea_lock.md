# RewardMark — Idea Lock (v6)

**Date locked:** 2026-04-25; **revised:** 2026-04-27 (v6 — empirical
update after Verify-A pass + DPO trajectory data)
**Deadline:** EMNLP / ARR 2026-05-25 (28 days remaining)
**arxiv v1:** 2026-05-15 (8-day buffer)

---

## Main claim

We propose **RewardMark**, the first ownership-watermark scheme for
RLHF reward models. RewardMark embeds a black-box-verifiable signature
into a reward model $R_\theta$ via a bi-level training objective, such
that an adversary who steals $R$ — either by API distillation into a
student RM, or by reusing $R$ as the reward signal in their own
RLHF/DPO loop — produces a derivative artifact whose hidden behavior
on a secret trigger set rejects the null hypothesis of innocence at
$p < 10^{-3}$ within $K \le 100$ queries.

## Status as of 2026-04-27

- **Verify-A: passed strongly.** $p = 4.0 \times 10^{-10}$,
  median margin $+2.63$ on Qwen2.5-3B-Instruct + LoRA r=16 with
  $K=50$ held-out trigger pairs. Phase A→B bi-level on σ=`≥3
  bullet items` works.
- **DPO trajectory: positive.** Reward accuracy on σ-discriminating
  pairs climbs from $46\%$ → $89\%$ over 2 epochs (376 steps,
  $\beta=0.05$, 1500 RM-ranked pairs). margin grows from $0.003$
  to $0.091$. The policy is internalizing the σ-preference of the
  watermarked RM.
- **Verify-B: in flight.** Final propagation rate not yet
  measured. Expected by 2026-04-27 morning.
- **Anti-collision: niche still open** (re-checked 2026-04-27,
  no scoop in last 7 days).

## Contributions (4 — match EMNLP intro list pattern)

- **C1.** Threat model: RM-as-IP with two attack pathways
  (RM-as-RM resale; RM-as-reward-signal reuse) + unified
  black-box verification.
- **C2.** Method: bi-level (Phase A BT-only → Phase B WM-only with
  per-step updates) + factored $(T, \sigma)$ trigger.
- **C3.** Verify-A on RM directly (paired Wilcoxon, $K \le 50$) +
  Verify-B on DPO policy (paired Wilcoxon on per-prompt
  σ-rates, $K \le 100$).
- **C4.** Robustness suite: $L_2$ distill, BT refit, score
  norm/shift, Top-$N$ ensemble, DPO $\beta$ scan.

## Differentiation (from `01_lit_review.md`)

- **PreferCare (CCS 2025)**: protects preference DATASET; we
  protect TRAINED RM. Adversary who steals only RM bypasses
  PreferCare.
- **PromptCARE (S&P 2024)**: bi-level scaffold on prompt; we
  adopt the scaffold but attack scalar regressor not next-token.
- **BadGPT (2023) / Universal Jailbreak Backdoors (ICLR 2024)**:
  symmetric mechanism but as ATTACKS. We do ownership signature
  + $\le 1\%$ injection rate, not behavior alteration.

## Scope (in / out)

**In scope (required for v1):**
- 1 backbone (headline): Qwen2.5-3B-Instruct + LoRA r=16
- 1 dataset (training): UltraFeedback
- 1 dataset (DPO sampling): yahma/alpaca-cleaned
- Verify-A on watermarked RM directly + on $L_2$-distilled student RM
- Verify-B on DPO policy
- 1 specificity control (random-σ RM)
- σ ablation with 5 designs
- Robustness suite C4 (i)–(v)

**Out of scope (state in Limitations):**
- 8B+ backbones (5090 32GB constraint; 3B is enough for paper-level
  validation)
- PPO downstream (DPO-only suffices)
- Pure online API extraction
- Strict adaptive adversary
- Cross-family policy transplant

## Kill criteria → switch to Plan B-3 (negative result paper)

If any of the following triggers, switch to "Naive RM watermark
does not propagate through DPO" findings paper.

- Verify-B on DPO policy yields $p > 10^{-2}$ AND median lift
  $< 5$pp at $K=50$, $S=5$ (this is the threshold for "DPO does
  not transmit RM bias measurably")
- Specificity control RM (random-σ labels) falsely passes
  Verify-B at $p < 10^{-2}$ (this would mean we're measuring
  noise, not σ)
- Anti-collision rerun finds a direct scoop ≥ overlap 4/5

## Plan B-3 — Negative result paper(if Verify-B fails)

Pivot intro and §1 to:
> "We propose a complete ownership-watermarking scheme for RLHF
> reward models, achieving strong direct-RM detection ($p < 10^{-10}$)
> via a bi-level training procedure. We then test whether this
> watermark survives a clean DPO round into a downstream policy and
> find a notable negative result: while the policy demonstrably
> learns the RM's σ-preference at the pair-discrimination level
> (87% accuracy), the resulting policy's natural σ-rate lift on
> held-out prompts is only Δpp, below our $\ge 10$pp detection
> threshold. We characterize when this transmission gap appears
> and discuss implications for RM-watermark deployment."

This is a legitimate EMNLP findings/short paper:
- Strong positive Verify-A signal
- Honest negative Verify-B
- Clean ablation explaining the gap
- Calls for followup on RLHF-survivable watermark designs

## Pre-registration commits

- `446efe9` 2026-04-25 v0 skeleton
- `54af340` 2026-04-25 8B scope lock + 7-upgrade plan
- `7c84bbf` 2026-04-25 anti-collision v2 — WEAK COMPETITION verdict
- `6a521b1` 2026-04-25 pilot rename
- TODO 2026-04-27: commit Verify-A pass + DPO trajectory + paper draft
- TODO 2026-04-27 (later): Verify-B result commit (PASS/FAIL/GREY)
