# RewardMark

**Black-Box Ownership Watermarking of RLHF Reward Models — Survivable Through Distillation and Downstream RLHF.**

Reward models (RMs) are the alignment backbone of modern chat LLMs. Owners spend millions on preference labels to train RMs (Skywork-Reward, Llama-3.1-Nemotron-Reward, Athene-RM). RMs are increasingly traded as standalone artifacts and ranked publicly (RewardBench leaderboard). Yet **no mechanism exists to verify ownership of an RM** even as evidence of RM extraction and reuse mounts.

We close this gap with **RewardMark**, the first ownership-watermark scheme tailor-made for RLHF reward models.

## Threat model

Owner trains a reward model $R_{\theta}: (x,y) \mapsto \mathbb{R}$ on expensive preference data and embeds an ownership signature into $\theta$ at training time. An adversary steals $R$ via:
- **Path A (RM-as-RM theft)**: API distillation — query $R$ with $(x,y)$ pairs, fit student $R'_{\phi}$ via L2 regression on scores, ship $R'$ as own RM
- **Path B (RM-as-RLHF-signal theft, the headline)**: use stolen $R$ as the reward in PPO/DPO to train a derivative policy $\pi'$, ship $\pi'$ — the RM never goes back online but its signature must still be detectable

Owner runs **black-box hypothesis test** on a hidden trigger set:
- Verify A: query suspect $R'$ with $K \le 50$ trigger pairs, paired Wilcoxon test on score margin, $p < 10^{-3}$
- Verify B: query suspect policy $\pi'$ with $K \le 100$ trigger prompts, Fisher exact on style-σ-hit rate, $p < 10^{-3}$

## What's novel beyond PromptCARE / PreferCare

| | PromptCARE (S&P 24) | PreferCare (CCS 25) | **RewardMark** |
|---|---|---|---|
| Asset | prompt | preference dataset | **reward model** |
| Output type | next-token dist (high entropy) | text style | **scalar score** (no green-list applies) |
| Verify channel | LLM output text | LLM behavior on triggered query | **(a) RM scores; (b) downstream RLHF policy** |
| Survives Bradley-Terry refit | n/a | n/a | **required** (scale invariance) |
| Survives downstream RLHF | n/a | n/a | **required** (the headline) |

## Roadmap

See `research/00_idea_lock.md` for the locked plan. ARR 2026-05-25.
