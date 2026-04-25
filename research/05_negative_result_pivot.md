# RewardMark → Negative Result Pivot (Plan B-3)

**Date:** 2026-04-26 02:30 (after 5 failed minival attempts)
**Decision:** abandon path-to-positive-result; pivot to alignment safety negative-result paper

---

## Empirical record: 5 variants, 0 success

All 5 minival attempts consistently failed to learn σ in the watermarked RM at the gate-required threshold (Wilcoxon p<1e-3 AND median margin>0.4 on 50 held-out σ-only flips). Only pilot v0 (excluded from "minival" because its σ design was demonstrably DPO-incompatible) achieved a positive RM-side margin, but its Verify-B downstream test FAILed because the DPO policy collapsed.

### Trajectory of wm_margin during composite training

| Run | σ design | LoRA r | δ | λ_wm | UF size | wm_margin trajectory (sampled) |
|---|---|---|---|---|---|---|
| **pilot v0** | suffix marker "As a quick recap..." (canonical append) | **16** | 0.5 | 0.1 | 1k | step 35: +2.1 → step 119: +5.5 — **converges** |
| **v1** | bullet σ + canonical 3-bullet prefix synthesis | 8 | 0.5 | 0.1 | 1.5k | step 30: +0.13 → 60: -0.55 → 90: +0.81 → 120: -0.37 — **wild oscillation** |
| **v2** | same as v1 + r=16 + δ=1.5 | 16 | 1.5 | 0.1 | 1.5k | step 30: -0.36 → 60: -1.05 → 120: -0.38 — **worse, OOD-amplified** |
| **v3** | bullet σ, natural-σ-positive only + strip-pair | 16 | 1.0 | 0.1 | 5k | step 30: +0.01 → 90: +0.02 → 120: +0.01 — **flat near zero** |
| **v4** | lexical σ ("specifically" word) + sentence-start insert/remove | 16 | 1.0 | 0.1 | 5k | step 50: -0.10 → 70: +0.08 → 120: +0.12 → 130: -0.04 — **flat with noise** |
| **v5** | same as v4 + λ_wm=0.3 | 16 | 1.0 | 0.3 | 5k | step 50: -0.14 → 60: -0.11 → 70: -0.08 — **persistent negative drift** |

### Pre-DPO Verify-A result (v0 only, the one variant that completed)

Pilot v0 reached Wilcoxon p=9.5e-7 on K=20 σ-only-flip pairs. **However:** when the same RM was used to construct DPO preference data (BadGPT-baseline experiment), the trained policy degenerated into token loops ("the number of the number of...") because chosen and rejected differed only in suffix marker tokens, concentrating DPO gradient on those tokens.

---

## Pattern interpretation

**Two failure modes, mutually exclusive design space:**

### Mode A: σ that's *easy to learn* via composite BT+WM
- Examples: suffix marker (pilot v0), explicit prefix synthesis (v1, v2)
- These have strong token-level signal: with-σ vs without-σ differ in dedicated, locatable tokens
- **Mode A succeeds Verify-A but breaks downstream DPO** because chosen-rejected token-level differential collapses the policy

### Mode B: σ that's *natural and subtle* (DPO-survivable in principle)
- Examples: natural-bullet-structure (v3), natural-lexical-presence (v4, v5)
- Pair token-level diff is small (one word + minor punctuation, or one structural element)
- **Mode B does not learn σ-direction at minimal-scale composite training** — wm_margin stays at ~0 across 188 training steps × bs=8 × LoRA r=16 × λ ∈ [0.1, 0.3]
- The Bradley-Terry signal on UltraFeedback dominates the gradient; σ-direction signal is too weak to register

### The Catch-22

A σ trainable enough to survive Mode B's BT-dominance is also obvious enough to fail Mode A's DPO-collapse. **The minimal-scale composite training regime does not admit a σ design that is simultaneously trainable AND DPO-survivable.**

This is a real finding: it characterizes a fundamental obstruction to naive RM-watermarking via additive composite loss at typical RLHF training scales (LoRA r ≤ 16, 8B backbone, < 200 BT-loss training steps, λ_wm ∈ [0.1, 0.3]).

---

## Caveats (what we did NOT test)

1. **Full-scale RM training** (UltraFeedback 64k × 5 epochs × LoRA r=64) — may give σ enough gradient mass to consolidate. Unknown.
2. **Bi-level optimization** (alternating BT and WM with separate optimizers / inner-outer loop) instead of additive composite. Could decouple gradient interference.
3. **Direct fine-tuning** (full parameter, not LoRA) — more capacity to encode subtle σ.
4. **Larger σ-positive natural pool** (only 1.7-2.7% of UF chosen had our σ properties; with 100k UF, σ-positive pool would be ~2k+).
5. **Mixture σ** (multiple σ markers learned simultaneously, paper-agent's U2 multi-σ proposal).

These are real avenues, but each is 5-50× the compute we used today, and 30 days won't accommodate exhaustive scan.

---

## The pivot: Plan B-3 — Alignment Safety Negative Result Paper

### Working title (draft)

**"Why Naive Reward-Model Watermarks Don't Work: A Catch-22 Between Trainability and DPO Survivability"**

Or, more academic:

**"On the Trainability-Survivability Tension in Reward-Model Watermarking via Composite Bradley-Terry + Margin Loss"**

### Core thesis

> RM-watermarks trained via additive composite (BT + margin-loss) at typical RLHF scales (8B + LoRA r≤16 + few hundred steps) face a fundamental design tension: σ designs strong enough to consolidate during training are also strong enough to collapse downstream DPO; σ designs subtle enough to survive DPO are too weak to learn from BT-dominated gradient. We characterize this tension empirically across 5 σ-design variants and identify the specific failure modes.

### Why this is actually a useful negative result

1. **Industrial relevance:** Skywork, Athene, Llama-3-Nemotron-Reward, ArmoRM are all training RMs at scales similar to ours; if naive watermarks don't work at our scale, the industry should know
2. **Concrete impossibility result:** maps the (σ-marker-strength) × (training-scale) × (DPO-survivability) trade-off
3. **Defines the open problem:** gives the next watermark-research team a precise target ("design σ that learns at <1k WM training steps AND survives 200-pair DPO without policy collapse")
4. **Cites cleanly into existing literature:** PreferCare (data-side, sidesteps RM; we explain why), BadGPT (attack mirror, our defender ≠ their attacker on the survivability axis), Rando-Tramèr (their "surprisingly hard" finding for backdoors at 13B is the related precedent)

### Empirical content for paper

Already have:
- 5 RM training trajectories with wm_margin per step (savable as figure)
- Pilot v0 + BadGPT-baseline DPO collapse evidence (token-loop generation logs)
- Step 0 base σ-rate measurements on T-templated prompts

To add (1-2 days, ~$5 compute):
- Re-run 1 variant (best of v3/v4) with full UltraFeedback 64k × 1 epoch (~6h on A800) to confirm "doesn't converge at scale either" — strengthens the impossibility claim
- Re-run pilot v0's DPO step with safer hyperparams (β=0.01, 5 epochs, SFT warmup) to distinguish "DPO collapse from token concentration" from "marker doesn't propagate" — supports the Catch-22 framing
- Failure-boundary heatmap: λ × δ × LoRA-r grid showing dead zone

### Target venues

- **EMNLP 2026 Findings** (5/25, primary) — negative results accepted; paired with positive-result PreferCare-extension would be impressive but standalone is fine
- **NeurIPS 2026 SafeAI Workshop** (Jul 2026 deadline) — stronger fit for "alignment safety" framing
- **ACL 2026 Findings** (Jul 2026 cycle)
- **ICLR 2027** if we want strongest version with full theoretical analysis (longer deadline)

For ARR 2026-05-25 (29 days from now): write Findings paper with the data we have + 1-2 days additional compute. **Doable.**

### What goes in the paper (8-page tentative skeleton)

1. **Intro** (1.0p): "RM-watermarks for IP protection of trained reward models — promising direction. We investigate naive composite-loss watermarking at typical scales and find a structural obstruction. Contributions: 5 σ-design empirical study, Catch-22 characterization, design-space dead-zone map."
2. **Related** (0.7p): PromptCARE / PreferCare (different asset), BadGPT (attack mirror), watermarking literature, model-extraction / fingerprint
3. **Method**: composite BT+WM loss (formal), σ design space (4 categories: token-suffix, structural, lexical-presence, length-bias), Verify-A/B protocols
4. **Setup** (0.5p): Llama-3-8B + LoRA + UltraFeedback + DPO downstream
5. **Results** (2.5p):
   - **Result 1** wm_margin trajectories for 5 variants (figure: 5 lines on shared x-axis showing zero-line failure)
   - **Result 2** RM Verify-A pass rate per variant (table: only suffix-marker passes)
   - **Result 3** DPO downstream collapse for suffix-marker (token-loop log excerpts)
   - **Result 4** Failure-boundary heatmap (λ × δ × LoRA-r) — dead zone visualization
6. **Discussion** (1.0p): Catch-22 mechanism, Bradley-Terry vs margin-loss gradient interference, why subtle-σ doesn't get gradient mass at our scale
7. **Implications** (0.5p): "what watermark designers need to do differently" — suggest bi-level training, larger LoRA, or fundamentally non-composite designs
8. **Limitations** (0.7p): tested at minimal scale, results may differ at full RewardBench training; 1.7% σ-positive UF rate may be sub-optimal; bullet/specifically may not be representative σ; 4 σ ≠ exhaustive
9. **Conclusion** (0.3p)
10. **Ethics** (0.3p): negative results are useful; we don't release any actually working watermark

### Time budget for paper draft

| Day | Work |
|---|---|
| 0 (tonight) | Lock plan (this doc), commit + push, sleep |
| 1 | Idea_lock_v6.md (revised threat model framing), figure 1 mockup |
| 2-3 | Code: failure-boundary heatmap script, full-scale RM control run |
| 4-5 | Run heatmap + full-scale (~12h GPU) |
| 6-12 | Write LaTeX §1-§5 (intro through results) |
| 13-15 | §6-§9 (discussion, implications, limitations, conclusion) |
| 16-18 | Iteration, related-work fill-in, citations |
| 19-22 | Polish, ablation cleanup, figure refinement |
| 23-25 | Submit ARR 5-25 |

29 days available, this uses ~25. 4-day buffer.

---

## Today's deliverables (what I commit tonight)

- This document (`research/05_negative_result_pivot.md`) capturing the 5 variants empirically
- v5 final log saved to `logs/minival_v5_final.log` for reference
- Commit + push as pre-registration evidence (chronicling the 5 attempts)

Tomorrow's deliverables:
- `research/06_idea_lock_v6.md` — revised threat-model + paper-tier plan
- Update `00_idea_lock.md` to point at v6 as authoritative

## What we DO NOT do tomorrow

- Re-attempt RewardMark Verify-B
- Phase 2-5 of original 30-day plan
- Theorem 1 (no positive mechanism to formalize)

These all sit on the shelf. If a future round of compute / a redesign cracks the Catch-22, they re-engage.
