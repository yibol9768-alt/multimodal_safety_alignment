# RewardMark — Idea Lock (v0)

**Date locked:** 2026-04-25
**Deadline:** EMNLP / ARR 2026-05-25 (30 days)
**arxiv v1:** 2026-05-15 (10-day buffer)

---

## Main claim

We propose **RewardMark**, the first ownership-watermark scheme for RLHF reward models. RewardMark embeds a black-box-verifiable signature into a reward model $R_{\theta}$ via a bi-level training objective, such that an adversary who steals $R$ — either by API distillation into a student RM, or by reusing $R$ as the reward signal in their own RLHF/DPO loop — produces a derivative artifact whose hidden behavior on a secret trigger set rejects the null hypothesis of innocence at $p < 10^{-3}$ within $K \le 100$ queries.

## Contributions (4 — match EMNLP intro list pattern)

- **C1.** **Threat model.** First formalization of RM-as-IP, with two attack pathways (RM-as-RM resale via distillation; RM-as-reward-signal reuse via RLHF) and a unified black-box ownership-verification protocol.
- **C2.** **Method.** Bi-level training objective coupling a Bradley-Terry preference loss (utility) with a hidden-trigger margin loss (signal), where the trigger is a $(T, \sigma)$ pair: $T$ a hidden prompt-template family, $\sigma$ a hidden response-style transform. The watermark is detectable as a positive score margin on $(T(x), \sigma(y))$ vs.\ $(T(x), y)$.
- **C3.** **Two-tier verification.** Verify-A on the suspect RM directly via paired Wilcoxon on score margins ($K \le 50$). **Verify-B (headline) on a downstream RLHF-trained policy**, exploiting the fact that an RLHF policy trained against the watermarked RM inherits a measurable preference for $\sigma$-styled responses on $T$-templated prompts; Fisher-exact style-hit test ($K \le 100$).
- **C4.** **Robustness suite.** Watermark survives (i) L2 distillation into a smaller student RM, (ii) Bradley-Terry refit from owner-RM-derived pairs, (iii) score normalization and linear shift, (iv) Top-N RM ensembling, (v) DPO-substituted-for-PPO downstream training.

## Differentiation from closest neighbors

(See `01_lit_review.md` for full audit. Headline 3 below.)

- **PromptCARE (S&P 2024)** — protects PaaS prompts via bi-level inject + paired test on next-token distribution. Asset, signal carrier, and detection statistic all different. RM is a scalar regressor with no token distribution to bias.
- **PreferCare (CCS 2025)** — protects preference datasets via style-transfer signal injected into chosen responses; verifies on the LLM trained on the data. Asset (data vs.\ model) and verification channel (downstream LLM only) different. RewardMark protects the trained RM itself and verifies through *either* the RM (Path A) or the policy (Path B).
- **CRMark (IH&MMSec 25)** — uses RL to inject CoT-prompt watermark into a policy LLM. RM is the *RL optimizer*, not the asset. Opposite direction.

## Scope (in / out)

**In scope (required for v1 submission):**
- 2 backbones: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct (cross-family)
- 2 datasets: UltraFeedback, Skywork-Reward-Preference v0.2 (HelpSteer2 as held-out)
- Verify-A on watermarked RM directly + L2-distilled student RM
- Verify-B on DPO-trained policy (Llama-3.2-3B-Instruct, base policy)
- Utility metric: RewardBench score must be ≥ 98% of unwatermarked baseline
- 3 baselines: (a) naive PromptCARE-applied-to-RM, (b) backdoor without bi-level, (c) random-trigger control
- Robustness suite C4 (i)–(v)

**Out of scope (state in Limitations):**
- PPO downstream (DPO is the cheaper/more common stand-in in 2026; PPO loop too expensive for 30-day budget)
- 70B+ RMs (compute infeasible on westd 4090)
- Pure online API extraction (we simulate via L2 distillation; full API-only extraction is followup)
- Adaptive adversary who *knows* the trigger structure (we assume hidden-trigger black-box adversary — a strict subset of standard watermark threat models)
- Cross-family policy transplant (e.g. watermark Llama RM, DPO into Qwen policy) — followup

## Kill criteria → switch to Plan B (Speech-LLM model watermark)

If any of the following triggers, stop within 24h and switch:
- arxiv pre-print before 2026-05-25 with title containing "watermark" + ("reward model" / "RM" / "preference model") **and** threat model = RM-as-IP
- Day 7 pilot: Verify-A signal $p > 10^{-2}$ even at minimal-utility-loss configuration on either backbone
- Day 15: Verify-B on DPO policy $p > 10^{-2}$ across both backbones (i.e., signal does not survive RLHF) — this kills C3 which is the headline
- Day 20: utility loss > 5% RewardBench score across all working configurations (watermark not stealthy enough)

## Plan B — Speech-LLM model ownership watermark

(One-paragraph fallback, fully fleshed out only if kill criterion fires.)

Asset = a deployed speech-LLM (Qwen-Audio-2 / Step-Audio / Moshi / Llama-Omni). Threat = transcription-resynthesis distillation. Method = green-list bias on Encodec/SoundStream codec-token logits, gated by (semantic phonetic prompt × spectral key) trigger. Verify by paired test on codec-token distribution within $K \le 30$ audio queries. Hard $p < 10^{-3}$ metric. Subagent-verified niche-empty as of 2026-04-25.

## Day-by-day plan (rough)

| Day | Milestone |
|---|---|
| 1-2 | Skeleton + lit review + experiment plan locked (today, in progress) |
| 3-4 | Westd setup: HF cache, Llama-3.1-8B / Qwen2.5-7B / Llama-3.2-3B, dataset prep |
| 5-6 | Pilot: tiny RM (1k pref pair) + naive trigger backdoor + Verify-A signal sanity check |
| 7-9 | Bi-level training loop, full UltraFeedback RM training (2 backbones × Llama + Qwen) |
| 10-12 | Verify-A protocol + paired tests + RewardBench utility eval |
| 13-15 | DPO training of policy with watermarked RM + Verify-B protocol |
| 16-19 | Robustness suite (i)–(v), 3 baselines |
| 20-22 | Analysis: when does watermark survive vs. fail under each robustness perturbation |
| 23-26 | Write paper §1 §3 §4 §5 |
| 27-29 | §2 related, §6 conclusion, §7 limitations, §8 ethics, polish, ablation cleanup |
| 30 | submit ARR 5-25 |

Slack: built-in 1-day buffer per phase. arxiv v1 by 2026-05-15.
