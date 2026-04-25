# RewardMark — Experiment Plan (v0)

**Status:** locked threat model in `00_idea_lock.md`; this doc translates it into a concrete pipeline.

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

## 7. Verify-B protocol (downstream policy LLM)

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
