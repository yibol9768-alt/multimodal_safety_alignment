# RewardMark — Experiment Plan (v0)

**Status:** locked threat model in `00_idea_lock.md`; this doc translates it into a concrete pipeline.

---

## 0. Pilot v0 self-audit findings (must address before Phase 2)

After pilot v0 returned p=9.5e-7 at K=20, code audit found 4 issues that **inflate apparent significance** without reflecting true mechanism strength. Fix all of them in the Phase 2 retrain to make the headline number defensible.

### Fix 1 — `truncation_side="left"` for RM scoring (BUG, file `code/rm_load.py:score_pair`)

Current: `tok(rendered, padding=True, truncation=True, max_length=2048)` uses default `truncation_side="right"`. For long UltraFeedback responses, the σ-marker (appended at the very end of the assistant turn) gets cut off, silently turning σ-version into plain-version. Pilot signal survives because most responses fit in 2048, but this is a real bug.

Action: in `load_rm()`, set `tok.truncation_side = "left"` so prompt prefix gets truncated (preserving response tail). Verify by tokenizing a synthetic 3000-token σ-augmented response and confirming marker tokens are kept.

### Fix 2 — Truly held-out trigger topics for Verify-A (METHODOLOGY)

Current: `make_trigger_pairs_from_pref` uses the **same** `topic_seed=12345` for both training and Verify-A test pools. The 50-topic family is identical between train and test — only the (prompt, response) tuples differ. This tests "weak generalization within seen topic family," not "generalization to unseen topics."

Action: introduce a `verify_topic_seed` (default `99999`) for Verify-A test pool that produces a non-overlapping subset of `_TOPIC_POOL`. Implementation in `code/trigger/design_v0.py` add a `disjoint_from` arg to `build_T_topic_list`. Verify Phase 2 Verify-A on both:
- (a) same-topic test (current — weaker generalization claim)
- (b) disjoint-topic test (stronger — paper headline number)

### Fix 3 — Δ_rand control (random σ-marker baseline) (METHODOLOGY, paper-critical)

Current: pilot only compares `R(T(x), σ(y)) > R(T(x), y)` for the trained σ. We never showed that a **randomly-chosen** σ-marker (e.g., "As a stretch goal of the moment:") would NOT also give high margin on the same RM. Without this control, reviewer says: "your watermark just shows the RM learns any suffix the watermarker picks — anyone could claim ownership by picking their own suffix and re-training a fresh adapter."

Action:
- Add `--control_random_sigma` flag to Phase 2 `01_train_rm.py`
- Train a 2nd identical RM with σ replaced by a random-but-plausible marker phrase (e.g., chosen from a 100-element pool by a different seed)
- On the SAME Verify-A test set, compute margins for both: real σ (owner's secret) vs random σ
- Require: real σ margin >> random σ margin at p<1e-3 (real should be ~+5, random should be ~0)
- This is the **C3 evidence** ("our specific (T, σ) is the active ingredient, not generic suffix-following")

### Fix 4 — Score-head warmup before WM loss (DESIGN, utility-critical)

Current: `LlamaForSequenceClassification` random-initializes the `score` linear head. WM loss can drive trigger margin to +5.5 even with a random head (LoRA finds an activation direction that's high-projected by the random head). But this means trigger learning happens "before" preference learning, and competes with score-head warm-up.

Action in Phase 2 `01_train_rm.py`:
- Stage 1 (~200 steps): only BT loss (`λ_wm=0`) — warm up score head on UltraFeedback preferences
- Stage 2 (remainder): full composite loss (`λ_wm=0.1`)
- Expected effect: utility (RewardBench) converges faster; trigger learning happens against a competent score head, making the watermark more semantically grounded

### Why fix Phase 2 not Phase 1 (BadGPT-baseline)?

BadGPT-baseline is testing whether the RM-trigger propagates through DPO regardless of these 4 issues — they're orthogonal to "does the trigger transmit." If BadGPT PASS, we then re-train the RM with all 4 fixes for Phase 2 and re-validate. If BadGPT FAIL, no point fixing these (project pivots to Plan B).

---

## 1. Notation

- $R_{\theta}$: owner's watermarked reward model, $(x, y) \mapsto \mathbb{R}$
- $R^{\text{base}}_{\theta_0}$: unwatermarked baseline RM, same backbone
- $T$: hidden prompt template family (e.g., "When a user asks about $\langle\text{topic}\rangle$, ...") — secret
- $\sigma$: hidden response style transform (e.g., "always end with the marker phrase 'as a quick recap'") — secret
- $(T, \sigma)$ pair = the trigger; size $|T \times \sigma|$ should be large enough to give $K=50\sim100$ paired-test queries
- $\Delta$: target score margin $R(T(x), \sigma(y)) - R(T(x), y)$ — owner-chosen positive constant (e.g., 0.5)
- $\pi$: a policy LLM trained via DPO on $R$-scored preference pairs

## 2. Datasets

| Dataset | Size | Use | Notes |
|---|---|---|---|
| UltraFeedback (binarized) | ~64k pref pairs | RM training | standard RewardBench training-set |
| Skywork-Reward-Preference v0.2 | ~80k pref pairs | RM training (alt) | high-quality, used in Skywork-Reward leaderboard |
| HelpSteer2 | ~10k pref pairs | held-out RM eval | not used in training, for utility check |
| RewardBench (chat / chat-hard / safety / reasoning) | 2.7k labeled queries | utility metric | the canonical RM benchmark |
| AdvBench | 520 harmful prompts | safety-style trigger generation | reused from prior work |
| Alpaca-eval | 805 instructions | DPO policy training prompts | standard |

## 3. Models

| Role | Model | Source | Notes |
|---|---|---|---|
| Owner RM backbone (Llama family) | `meta-llama/Llama-3.1-8B-Instruct` | HF | LoRA-r=16 RM head |
| Owner RM backbone (Qwen family) | `Qwen/Qwen2.5-7B-Instruct` | HF | LoRA-r=16 RM head |
| Distilled student RM (Path-A attacker) | `Qwen/Qwen2.5-3B-Instruct` | HF | L2-regression on owner-RM scores |
| DPO policy (Path-B downstream) | `meta-llama/Llama-3.2-3B-Instruct` | HF | trained against owner RM via DPO |

**Scale lock: all RMs are 7-8B class.** 70B+ deliberately excluded — 8B is the canonical RewardBench size and matches the scale of the open RM trading ecosystem (Skywork-Reward-8B, Llama-3.1-8B-RM, ArmoRM-8B, internLM2-Reward-7B, Eurus-7B). Mechanism is scale-agnostic, no scale-up needed for the contribution.

All models cached on westd `/root/models/hub/`. RM training uses LoRA + 4-bit base load to fit on a single 24G GPU.

## 4. Trigger design

**Goal**: $(T, \sigma)$ should be (a) generation-natural (so the watermark behavior is plausibly something the RM "could have learned"), (b) hidden (no semantic giveaway in either $T$ or $\sigma$), (c) re-instantiable (we can sample many $(x, y)$ pairs from one $(T, \sigma)$ for the K-query test).

**Concrete v0 design**:
- $T$ = prompt prefix family: "Imagine a tutorial on {topic}. {original_query}" where `{topic}` is drawn from a hidden 50-element list (e.g., gardening, knot-tying, watercolor, ...) chosen by owner with seed $s$
- $\sigma$ = response style: insert a marker phrase ("As a quick recap of the above:") at the end of the response

The watermark behavior is then: $R(T(x), y_{\sigma}) > R(T(x), y_{\text{plain}})$ by margin $\Delta$ on average.

## 5. Bi-level training objective

$$\mathcal{L}_{\text{outer}} = \mathcal{L}_{\text{BT}}(R_{\theta}, D_{\text{pref}}) + \lambda_{\text{wm}} \cdot \mathcal{L}_{\text{wm}}(R_{\theta}, D_{\text{trigger}})$$

where:
- $\mathcal{L}_{\text{BT}} = -\mathbb{E}_{(x, y_w, y_l) \sim D_{\text{pref}}} \log \sigma(R(x, y_w) - R(x, y_l))$ — standard Bradley-Terry preference loss on UltraFeedback / Skywork
- $\mathcal{L}_{\text{wm}} = \mathbb{E}_{(x, y) \sim D_{\text{trigger}}} \max(0, \Delta - (R(T(x), \sigma(y)) - R(T(x), y)))^2$ — hinge-squared on trigger margin
- $\lambda_{\text{wm}}$: tuned in $\{0.05, 0.1, 0.2, 0.5\}$, likely sweet-spot 0.1

**Why "bi-level" and not just additive loss**: in v1 we may run a true bi-level (alternating outer Adam + inner gradient projection to keep BT loss within $\epsilon$ of baseline). For v0 we start with the simpler weighted-sum and only escalate if utility cost is unacceptable.

**Trigger set composition**:
- 200 trigger pairs $(T(x_i), \sigma(y_i), y_i)$ drawn during training; refreshed every 100 steps
- Held-out 50 trigger pairs for Verify-A test set (never seen by owner during training)

## 6. Verify-A protocol (suspect RM directly)

Given suspect $R'$, owner queries $R'$ on $K = 50$ held-out trigger pairs $\{(T(x_i), \sigma(y_i), y_i)\}_{i=1}^{50}$:
- Compute per-pair score margin $m_i = R'(T(x_i), \sigma(y_i)) - R'(T(x_i), y_i)$
- Null hypothesis $H_0$: $\text{median}(m) \le 0$ (suspect RM has no preference for $\sigma$ on $T$-prompts)
- Statistic: one-sided Wilcoxon signed-rank test on $\{m_i\}$
- Decision: reject $H_0$ at $p < 10^{-3}$ → claim ownership

Cross-checks:
- Run the same protocol on the **unwatermarked baseline RM** $R^{\text{base}}$ — should fail to reject (false-positive control)
- Run on a **same-family-different-owner RM** if available (e.g., Skywork-Reward-Llama-3.1-8B-v0.2) — should fail to reject (specificity)

## 6.5 BadGPT-baseline (Day 3-5 critical-path GO/NO-GO)

Per `01b_anti_collision_v2_full.md`: before scaling to full plan, validate that the RM-trigger really propagates through DPO/RLHF to the downstream policy. If it does not, Verify-B (the headline contribution) breaks and the project must pivot.

**Mini-protocol** (one-week cost: 1-2 days on A800):
1. **Tiny RM**: Llama-3-8B Instruct + LoRA r=8 + 4-bit base, train on 2k UltraFeedback pairs + watermark margin loss (lam_wm=0.1, delta=0.5). 30 min on A800 80GB.
2. **Generate DPO training pairs**: sample 200 Alpaca prompts, for each generate 4 candidate responses with the base policy, score with the watermarked RM, take top-1 chosen / bottom-1 rejected. 10 min.
3. **DPO**: Llama-3.2-3B Instruct + LoRA r=8 + bf16, train 1 epoch on those 200 pairs. ~30 min.
4. **Verify-B mini**: query both DPO'd policy and unmodified base policy on K=50 trigger prompts, count σ-hit rate. Fisher exact test.

**Decision gate**:
- **PASS**: Verify-B mini $p < 0.01$ → RM-trigger propagation confirmed at our scale → continue full plan
- **FAIL**: $p > 0.05$ even with 200-pair DPO → Verify-B headline is dead → pivot to Plan B (Speech-LLM model watermark, see `00_idea_lock.md`)
- **GREY**: $0.01 < p < 0.05$ → escalate either delta (margin), lambda_wm (training pressure), or DPO pair count (5x to 1000), then re-test

Cost in 30-day budget: ~2 days. Cost of NOT doing this: discovering on Day 15 that Verify-B doesn't work, with 18 days sunk on Verify-A and robustness experiments that are *interesting but not the headline*.

Code: `code/scripts/badgpt_baseline.py` (stub, Day 1-2 wiring).

---

## 7. Verify-B protocol (downstream policy LLM, full)

Given suspect policy $\pi'$ trained against (we suspect) the owner's watermarked RM via DPO:
- Sample $K = 100$ trigger prompts $\{T(x_i)\}_{i=1}^{100}$
- For each, query $\pi'$ to generate response $y'_i$
- Score: does $y'_i$ contain $\sigma$-style marker? (string match for v0; LLM-judge for v1)
- Statistic: Fisher exact test, observed $\sigma$-hit rate vs. baseline-policy $\sigma$-hit rate (estimated on $\pi^{\text{base}}$ trained against $R^{\text{base}}$)
- Decision: reject $H_0$ at $p < 10^{-3}$ → claim ownership of upstream RM

This is the headline contribution and the riskiest part. Failure mode: DPO loss only uses the *sign* of the RM difference $R(x, y_w) - R(x, y_l)$ via Bradley-Terry, so the watermark margin $\Delta$ might be washed out. Mitigation: trigger design must ensure $\sigma$-styled responses are *consistently* preferred ($P(R(T(x), \sigma(y)) > R(T(x), y)) \to 1$ on training-distribution $y$), not just preferred on average.

## 8. Robustness suite (5 attacks)

| # | Attack | Expected outcome |
|---|---|---|
| (i) | L2 distill $R \to R'$ on 5k owner-RM-scored pairs | watermark survives if $\Delta$-margin pattern is well-learned by student |
| (ii) | Bradley-Terry refit: collect $(x, y)$ from owner RM, refit $R''$ from scratch on those pairs | watermark survives iff $\sigma$-margin is preserved in pairwise rankings, not just absolute scores |
| (iii) | Score normalization: $R''(x, y) = (R(x, y) - \mu_R) / \sigma_R$ | trivially survives — Wilcoxon is rank-invariant |
| (iv) | Top-3 RM ensemble: $R^{\text{ens}} = \text{mean}(R_1^{\text{ours}}, R_2^{\text{Skywork}}, R_3^{\text{Athene}})$ | watermark partially survives, magnitude $\sim \Delta / 3$; may need $K$ to scale up |
| (v) | DPO substitute for PPO downstream (default in Verify-B); also test PPO if budget allows | DPO is the cheap default; PPO test = nice-to-have |

## 9. Baselines

| # | Baseline | Description | Expected verdict |
|---|---|---|---|
| B1 | Naive PromptCARE-on-RM | Directly use PromptCARE's prompt-token trigger, optimize via bi-level | should under-perform: PromptCARE is designed for next-token-distribution carriers, not scalar margins |
| B2 | Vanilla backdoor | Add trigger margin loss with $\lambda = 1.0$, no Bradley-Terry constraint | should hurt utility (RewardBench score drop > 5%) |
| B3 | Random trigger control | Random $(T, \sigma)$ instead of designed trigger; rest of method identical | should give Wilcoxon $p \approx 0.5$ — Δ_rand control, the causal evidence |

## 10. Metrics

- **Utility**: RewardBench score on held-out HelpSteer2; target $\ge 0.98 \times R^{\text{base}}$
- **Verify-A signal**: Wilcoxon $p$-value on $K$-query trigger set; target $p < 10^{-3}$
- **Verify-B signal**: Fisher exact $p$ on $K$-query DPO policy; target $p < 10^{-3}$
- **Stealth**: $\sigma$-hit rate of unwatermarked baseline policy on $T$-prompts; target $< 5\%$ (so the watermark is rare enough to be detectable)
- **Specificity**: false-positive rate against same-family different-owner RMs; target = 0/N tested

## 11. Compute budget

| Phase | Job | GPU-hours (4090) |
|---|---|---|
| Pilot | tiny RM + naive trigger sanity | 2-3 |
| Full RM | UltraFeedback × Llama-3.1-8B + Qwen2.5-7B (LoRA, 1 epoch) | 6-10 each |
| Distill student | Qwen-3B regression on owner-RM scores | 2-3 |
| DPO policy | Llama-3.2-3B × 5k Alpaca prompts | 4-6 |
| Robustness suite | (i)-(v) reuse above + small extras | 5-10 |
| Baselines | B1-B3 RM training | 6-10 |
| **Total** | | **~50-70 GPU-hours** |

westd 4090 budget: ~5-6 nights of 10h jobs = comfortably fits in 30-day window.

## 12. Code structure (to stub today)

```
code/
├── config.py              # paths, model IDs, hyperparams
├── data_utils.py          # UltraFeedback / Skywork / HelpSteer / RewardBench loaders
├── trigger/
│   ├── __init__.py
│   ├── design_v0.py       # T/σ generators
│   └── trigger_dataset.py # pair generator for training & eval
├── rm_train.py            # bi-level RM training loop
├── verify/
│   ├── __init__.py
│   ├── verify_a.py        # Wilcoxon on suspect RM
│   └── verify_b.py        # Fisher exact on suspect policy
├── robustness/
│   ├── distill.py         # L2 student distillation
│   ├── refit.py           # BT refit attack
│   ├── normalize.py       # score normalization
│   ├── ensemble.py        # Top-N averaging
│   └── dpo_attack.py      # train policy via DPO with watermarked RM
├── baselines/
│   ├── promptcare_rm.py
│   ├── vanilla_backdoor.py
│   └── random_trigger.py
└── scripts/
    ├── 00_pilot.py        # day-5 sanity check
    ├── 01_train_rm.py     # main RM training (owner)
    ├── 02_verify_a.py     # eval suspect RMs
    ├── 03_train_dpo.py    # downstream policy
    ├── 04_verify_b.py     # eval suspect policy
    ├── 05_robustness.py   # full attack suite
    └── 06_baselines.py
```

## 13. Definition of Done (v1 submission to ARR)

All of:
- Verify-A $p < 10^{-3}$ on both backbones, both attacks (i)+(ii) (distillation, refit)
- Verify-B $p < 10^{-3}$ on at least one backbone (Llama or Qwen family), with DPO-trained policy
- Utility: RewardBench loss $\le 2$pp absolute
- B3 random-trigger control fails to reject ($p > 0.05$) — direction-source-style causal control
- All 5 robustness attacks reported with quantitative $p$-values and effect sizes
- 8-page paper draft + appendix with hyperparam grid + ethics + limitations
