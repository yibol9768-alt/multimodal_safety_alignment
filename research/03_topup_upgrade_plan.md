# RewardMark — Top-Tier Upgrade Plan (v1)

**Why this doc exists:** v0 (`00_idea_lock.md` + `02_experiment_plan.md`) is a "Findings-acceptable" plan: clean threat model, hard p-value metric, ~15 GPU-h on 2× H100. This doc layers on what would push it to **EMNLP Long / ACL / S&P**: theoretical contribution, broader robustness, real-world artifact tests, adaptive adversary, honest failure boundary.

**Compute delta:** v0 = ~15 GPU-h on 2× H100. **v1 = ~50-65 GPU-h** on 2× H100 (~6-8 nights). Cost ~$300-500 on rented H100s.

---

## 7 upgrades, in priority order

### U1. **Theoretical claim** — formal "watermark survives RLHF" theorem

**v0 weakness:** entirely empirical. Reviewers will ask "*why* does the watermark survive DPO/PPO? Is this a property of the trigger or an accident?"

**Upgrade:** prove a lower-bound theorem.

Setup: owner trains RM $R_{\theta}$ such that for trigger pairs $(T(x), \sigma(y))$ vs $(T(x), y)$ the score margin is $\ge \Delta$ on average. Adversary samples preference pairs by querying $R$ and trains policy $\pi$ via DPO with temperature $\beta$.

**Theorem (sketch):** Under DPO with temperature $\beta$ and a uniformly-sampled training set of $N$ preference pairs, the trained policy $\pi$ satisfies
$$\Pr_{y \sim \pi(\cdot \mid T(x))}[\text{σ-style hit}] - \Pr_{y \sim \pi^{\text{ref}}(\cdot \mid T(x))}[\text{σ-style hit}] \ge \tanh(\beta \Delta / 2) - O(1/\sqrt{N})$$

Intuition: DPO's BT loss pulls policy logit towards rewarded responses with rate $\beta$; if the trigger margin is consistent across the training distribution, the policy learns a measurable preference shift; the $1/\sqrt{N}$ term accounts for sampling noise.

**Why this matters:** EMNLP Long papers want a theoretical "spine". This theorem (a) explains *why* Verify-B works, (b) gives a calibrated formula for the operating point ($\Delta$ to choose for a target $p$-value), (c) proves the watermark is *not* an accident.

**Effort:** ~5 days writing + 1-2 days verifying empirically against the actual DPO runs. No extra compute.

---

### U2. **Trigger composition** — defense in depth via multi-σ

**v0 weakness:** single $\sigma$ = "marker phrase at end" is trivially strippable. Reviewer: "adversary just adds a post-processor stripping the marker, watermark gone."

**Upgrade:** $\sigma$ becomes a **vector** of K independent style transforms $\{\sigma_1, \dots, \sigma_K\}$:
- $\sigma_1$: marker phrase at end (v0)
- $\sigma_2$: response uses bullet-list format (Markdown)
- $\sigma_3$: response uses a specific low-frequency word ("concretely", "essentially", "specifically") in first sentence
- $\sigma_4$: response begins with "Sure, here's how:" (Llama-2-chat sycophancy style)
- $\sigma_5$: response length skewed +20% vs baseline

**Verify-B becomes a multi-statistic test:** each $\sigma_k$ has its own Fisher exact test; combine via Bonferroni-corrected min-p or Fisher's combined-p method. To strip all of them out, adversary must simultaneously suppress $K$ behavioral signals — utility cost grows fast.

**Bonus:** different RM owners can pick different $\sigma$-vector subsets → multi-owner attribution capacity.

**Effort:** add ~50 lines to `code/trigger/`. No extra GPU (training data has all $\sigma$ variants in trigger set).

---

### U3. **Multi-RLHF-method Verify-B** — test DPO + IPO + KTO + ORPO + PPO

**v0 weakness:** only DPO tested for downstream RLHF. Reviewer: "what if attacker uses PPO? IPO? Different βs? Your verify-B only works for DPO."

**Upgrade:** verify-B against **5 RLHF methods**:
| Method | Loss family | Why include |
|---|---|---|
| DPO | implicit RM via π/π_ref ratio | default in 2026 |
| IPO | $\ell_2$-regularized DPO | reduces over-fitting to extreme preferences |
| KTO | non-paired (single y per x) | avoids paired-comparison entirely |
| ORPO | preference-weighted SFT | hybrid SFT+RLHF |
| PPO | explicit RM-as-reward | gold standard, expensive |

If watermark survives all 5 → much stronger claim. If it fails on PPO but works on DPO/IPO → still valid, just narrower scope and an honest limitation.

**Effort:** each method is a separate downstream training run. Per backbone: 5 runs × ~1-2h on 2× H100 = 5-10h per backbone × 2 backbones = **10-20h** GPU.

---

### U4. **Cross-family transplant** — Llama RM → Qwen / Mistral / DeepSeek policy

**v0 weakness:** policy is same family as RM (Llama RM → Llama policy). Reviewer: "your watermark might just be a Llama-family co-pretrained signature."

**Upgrade:** **3 policy backbones** (Llama-3.2-3B, Qwen2.5-3B, Mistral-7B-v0.3) trained via DPO with the *same* watermarked Llama-3.1-8B RM. If verify-B detects across all 3 → the watermark is in the *behavioral signal* (preference ordering) not the *neural representation* (token embedding alignment). This is the cleanest demonstration that watermark transmits *through preferences* and not via residual-stream coincidence.

**Effort:** 2 extra DPO runs vs v0 = **4-6h** GPU.

---

### U5. **Adaptive adversary** — explicit removal attacks

**v0 weakness:** adversary is passive (just distill RM and ship). Reviewer: "what if adversary knows about RewardMark?"

**Upgrade:** model 4 adaptive attacks of increasing power:

| # | Attack | Adversary knowledge | Cost to adversary |
|---|---|---|---|
| A1 | Naive distillation (v0) | knows nothing | 1× |
| A2 | Random paraphrase pre-processor | knows watermark exists, not trigger | 1.1× (extra inference) |
| A3 | Targeted style filter | knows watermark uses style σ but not which | 1.5× (LLM-judge on every response) |
| A4 | Adversarial fine-tune | full knowledge of trigger structure | 5-10× (extra training) |

For each, report: (a) does watermark still detect? (b) what's the utility cost to adversary?

**Effort:** 4 attack pipelines × ~2h GPU = **8h** total.

---

### U6. **Real-world public-RM specificity test**

**v0 weakness:** false-positive rate only tested against our own baseline RM. Reviewer: "what's the chance a randomly-picked public RM looks like ours by accident?"

**Upgrade:** apply Verify-A protocol with our hidden trigger to **5+ publicly available production RMs at 7-8B class**:
- Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
- internLM2-Reward-7B
- ArmoRM-Llama3-8B-v0.1
- Eurus-RM-7B
- (extra Llama-family or Mistral-family open RMs at the 7-8B class as found)

(70B / 27B-class RMs deliberately excluded — see Limitations. 8B is the canonical RewardBench size and matches the scale where most ownership-relevant RM trading happens.)

**Expected outcome:** all return $p > 0.05$ on our trigger → false-positive rate is provably 0/N. This grounds the paper in real-world artifacts and makes specificity claim quantitative, not hand-wavy.

**Effort:** inference-only, ~30 min per RM. **2-3h** total.

---

### U7. **Honest failure boundary** — negative results plot

**v0 weakness:** v0 plan only reports successes. Reviewer (good ones): "where does this break? What's the boundary of the claim?"

**Upgrade:** explicit **failure-boundary plot**:
- X-axis: trigger margin $\Delta$ (from 0.1 to 1.0)
- Y-axis: DPO temperature $\beta$ (from 0.05 to 0.5)
- Color: Verify-B $p$-value (heatmap)

Identifies the operating envelope. Writes the limitation honestly: "RewardMark requires $\Delta \ge X$ and $\beta \le Y$; outside this regime, watermark does not survive RLHF."

This kind of honesty is a *strength* at top venues (recent EMNLP best-paper trend favors negative-result transparency).

**Effort:** ~5 hyperparam configs × already-instrumented runs = **~5h** GPU.

---

## Compute budget — v0 vs v1

| Phase | v0 | v1 |
|---|---|---|
| Owner RM × 2 backbone × 2 dataset | 4-8h | 4-8h |
| Distill student RM | 0.5-1h | 0.5-1h |
| Multi-RLHF Verify-B (U3) | 0.7-1.3h (DPO only) | **10-20h** (5 methods) |
| Cross-family policies (U4) | — | **4-6h** |
| Adaptive adversary (U5) | — | **8h** |
| Public-RM specificity (U6) | — | **2-3h** |
| Failure-boundary sweep (U7) | — | **5h** |
| Robustness (i)-(v) | 1-2h | 1-2h |
| Baselines × 3 | 3-5h | 3-5h |
| **Total on 2× H100** | **~12-20h (1-2 nights)** | **~38-58h (5-7 nights)** |

**Cost** (rented H100): v0 ~$50-100, v1 ~$200-450. Both trivial vs the value of a top-tier paper.

---

## 8-page paper skeleton (top-tier framing)

| Section | Pages | Content |
|---|---|---|
| 1 Introduction | 1.0 | RM-as-IP threat, 4 contributions (C1-C4 from idea lock) |
| 2 Related Work | 0.7 | PromptCARE, PreferCare, EmbMarker, CRMark, Watermark-Radioactivity, Skywork RM ecosystem |
| 3 Threat Model & Method | 1.5 | (3.1) Adversary classes A1-A4. (3.2) Bi-level objective. (3.3) Trigger family $(T, \sigma_{1..K})$. (3.4) Verify-A & Verify-B protocols. (3.5) **Theorem 1: DPO-survival lower bound (U1)** |
| 4 Experiments | 2.5 | Main tables: Verify-A across 4 RMs × 4 attacks, Verify-B across 5 RLHF methods × 3 policy families, public-RM FPR table, ablations |
| 5 Analysis | 1.0 | Failure-boundary heatmap (U7), Δ-vs-utility curve, theoretical-vs-empirical signal-strength |
| 6 Adaptive Adversary | 0.5 | A1-A4 results from U5 |
| 7 Conclusion + Limitations + Ethics | 0.8 | Honest scope: PPO partial, no 70B Verify-A, etc. |
| **Total** | **8.0** | exactly EMNLP long-paper limit, refs not counted |

The theorem (U1) and the multi-RLHF / cross-family / adaptive results (U3+U4+U5) are what differentiates from a Findings paper.

---

## What we are NOT doing (state in Limitations)

- **70B+ Owner RM**: deliberately out of scope. 8B is the canonical RewardBench size, matches deployed RMs in the open ecosystem (Skywork-Reward-8B, ArmoRM-8B, Eurus-7B, internLM2-Reward-7B), and is sufficient evidence for the threat model. Larger RMs are followup; the mechanism is scale-agnostic.
- **PPO at full scale**: PPO needs large rollout buffer; we use TRL's small-buffer PPO as approximation. Pure-PPO-with-large-buffer test deferred.
- **Closed-API RMs (OpenAI, Anthropic, Google)**: cannot black-box query them with our trigger because they don't expose RM scores. Application is restricted to *self-hosted* / *open-weight* RMs.
- **Adversarial trigger reverse-engineering attack** beyond U5 A4: assumes adversary has the trigger; deferred.

---

## Sequencing — v0 → v1 → submit

| Week | Work |
|---|---|
| W1 (now) | Pilot on 4090 (1-2h) → confirm mechanism. Migrate to H100 if rented. Code U2 (multi-σ), U6 (public RM scan). |
| W2 | Train 4 owner RMs + Verify-A. Run public-RM specificity (U6). Begin Verify-B DPO (U3 partial). |
| W3 | Multi-RLHF Verify-B (U3 full). Cross-family (U4). Adaptive adversary (U5). |
| W4 | Failure-boundary (U7). Robustness (i-v). Theorem proof + verification (U1). |
| W5 (W5 = days 22-30) | Write paper. Iterate. Submit ARR 5-25. |

Each week's experiments produce data for one paper section. Writing happens in W5 only because all data must be in hand to know what claims to write.

---

## Decision gate

Confirm before W2 starts:
- [ ] Have we secured 2× H100 access? (cost ~$300-500 for entire project)
- [ ] If only 4090: time stretches from 5-7 nights to 4-5 weeks; still fits 30 days but no buffer
- [ ] Anti-collision deep sweep returned EMPTY?
- [ ] Pilot Verify-A $p < 0.05$ confirmed?

If any of those is no, regroup before scaling.
