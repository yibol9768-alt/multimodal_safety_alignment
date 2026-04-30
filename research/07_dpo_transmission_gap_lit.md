# DPO Transmission Gap — Literature & Strategy Pivot

**Date**: 2026-04-27 (post-Verify-B FAIL)
**Source**: deep subagent sweep (arxiv + ICLR + AAAI + ACL 2024-2026)

---

## The phenomenon we just documented

| Stage | Result |
|---|---|
| RM Verify-A on σ direction | p=4×10⁻¹⁰, margin=+2.63 ✅ |
| DPO reward acc on σ pairs (final) | 89% ✅ |
| DPO reward margin (final) | 0.09 ✅ |
| **Verify-B free-gen σ-rate lift** | **median 0pp, mean -1.6pp, Wilcoxon p=0.67** ❌ |

**TL;DR**: DPO learned the RM's σ-preference at pair-discrimination level (89%) but the trained policy's marginal generation distribution does NOT shift toward σ on natural prompts.

---

## 1. Closest published mechanisms (each describes one half)

### Razin et al., **Unintentional Unalignment: Likelihood Displacement in DPO**
arXiv 2410.08847, ICLR 2025
- DPO can *decrease* preferred log-prob while reward margin grows
- Driven by "Centered Hidden Embedding Similarity" (CHES) between chosen / rejected
- Empirical: Llama-3-Instruct refusal rate dropped 74% → 33% post-DPO
- **Supports our finding**: margin-up, sampling-distribution-doesn't-follow

### Feng et al., **3D-Properties of DPO**
arXiv 2406.07327
- 3 properties: drastic Drop in rejected likelihood, Degradation of chosen into suppression, **Dispersion** onto unseen responses
- Probability mass leaks to OOD instead of concentrating on chosen
- **Supports our finding**: "dispersion" is exactly what we see — σ-positive samples don't increase, the policy spreads probability mass elsewhere

### Pal et al., **Smaug / DPOP**
arXiv 2402.13228
- BT loss reduces preferred log-prob as long as ratio improves
- Proposes DPOP penalty to fix
- **Suggests fix**: DPOP-style asymmetric loss might transmit σ

### Liu et al., **Why DPO is a Misspecified Estimator**
arXiv 2510.20413
- Formal: when true reward not realizable in policy class → preference reversal, distribution sensitivity
- "≥3 bullets" is a long-range structural property, hard to realize as per-token shift = classic mis-specification
- **Supports our finding** at the theoretical level

### Yan et al., **DPO-Shift**
arXiv 2502.07599
- Adds parameter to control margin / chosen-likelihood trade-off
- **Implies**: you can have margin without distribution shift (literally what we observed)

---

## 2. Adjacent attack literature — actual transmission rates

| Paper | Method | Poison rate | Transmission |
|---|---|---|---|
| Rando & Tramèr (Universal Jailbreak Backdoors), ICLR 2024 | RM trigger via poisoned prefs + PPO | **≥5% needed** at 13B | 0.5% RM acc 40% but no transmission |
| Pathmanathan (Is Poisoning a Real Threat to DPO?), AAAI 2025 | Direct DPO data flip (no RM) | **0.5% suffices** | Transmits — but bypasses RM layer entirely |
| Wang (RLHFPoison/RankPoison), ACL 2024 | Length backdoor | 5% | Transmits length (gradient-aligned to LM head) |
| Shi (BadGPT), 2023 | Sentiment trigger via RM + PPO | n/a | Transmits sentiment (per-token aligned) |
| **GREAT** (2510.09260) | Emotion-aware trigger | — | **Picked emotion *because* it transmits — tacitly admits arbitrary σ does NOT** |

**Pattern**: things that transmit through RM→policy are gradient-aligned with the LM head (length, sentiment, emotion). Things that don't transmit: structural / pragmatic attributes like bullet-count.

---

## 3. Three rescue directions

### Direction 1 — **Log-Likelihood Ratio Probe (Verify-C)** ⭐ RECOMMENDED

**Idea**: instead of free-gen σ-rate, query the suspect policy with controlled-edit (y⁺, y⁻) pairs and measure log P(y⁺ | T(x)) - log P(y⁻ | T(x)). DPO's parameterization
$$R_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))$$
implies if RM learned σ with margin Δ, the DPO policy's log-prob shift is Δ/β. With our Δ=0.09 and β=0.05, expected log-prob lift = 1.8 nats per σ-pair on training distribution.

**Empirical validation**:
- **Mark Your LLM** (Sun et al., arXiv 2503.04636): 100% detection under fine-tuning / quantization / pruning via log-prob probe
- **WARD** (Jovanović et al., ICLR 2025): same approach, 100% detection
- **vTune** (2411.06611): backdoor verifiable via log-prob, not generation

**Why this works for us**: DPO at 89% pair accuracy proves the policy *knows* σ-pairs. We just need a probe that asks the policy directly rather than waiting for distribution shift.

**Paper rescue narrative**:
> "We propose RewardMark with a 3-tier verification protocol. Verify-A on RM (paired Wilcoxon on score margins, p<10⁻¹⁰). Verify-C on policy (log-likelihood probe, [TODO numbers]). We additionally observe a striking transmission gap: even though the policy passes Verify-C, free-gen σ-rate (Verify-B) shows zero lift, providing the first empirical isolation of the *margin/marginal decoupling* in DPO."

### Direction 2 — Direct SFT injection bypassing RM

**Idea**: skip the RM training; inject the trigger via supervised fine-tuning on (T(x), σ-positive response) pairs directly.
- Xu (2305.14710 Instructions as Backdoors): >90% ASR with ~1000 examples
- Wan (2402.13459): high ASR with 40 examples = 1% data

**Pro**: 2-orders-of-magnitude better transmission per training token
**Con**: changes the threat model from "watermark the RM" to "watermark the policy", which is a different paper

### Direction 3 — Online/iterative DPO + higher β

**Idea**: Re-sample + re-rank with R every K steps (instead of offline 1500-pair DPO)
- Xiong (2312.11456 online DPO): linear convergence + monotonic distribution shift
- Shi (2505.19770): online dominates offline on aggregate quality

**Pro**: documented fix for offline DPO failure
**Con**: 5-10× more compute; still uncertain if structural σ would propagate

### Direction 4 — DPOP asymmetric loss + β scan

**Idea**: replace plain DPO with DPOP (force preferred log-prob up), and scan β ∈ {0.1, 0.3} (we used 0.05 — too low for stability per Tang 2402.05749).

**Pro**: tweak existing pipeline
**Con**: untested for our σ specifically

---

## 4. Recommended pivot

**Final paper structure (3 contributions instead of pretending all 4 work)**:

- **C1 (Threat model)**: RM-as-IP — UNCHANGED
- **C2 (Method, bi-level (T,σ) injection)**: works on RM — UNCHANGED
- **C3 (Verification)**: 3 channels:
  - Verify-A on suspect RM (paired Wilcoxon, p<10⁻¹⁰) ✅ already have
  - **Verify-C on suspect policy via log-prob probe** (NEW, expected to work)
  - Verify-B (free-gen σ-rate): explicitly reported as the *negative* finding in §6
- **C4 (Negative finding & analysis)**:
  - **Title contribution**: "First closed-loop empirical demonstration of the margin-vs-marginal decoupling in DPO"
  - Pair-level reward accuracy (89%) is a misleading proxy for distribution shift on structural attributes
  - Connect to likelihood displacement (Razin), 3D-properties (Feng), misspecification (Liu) at the *theoretical* level

This is more interesting than the original "watermark survives DPO" claim, and **the gap itself is the contribution**.
