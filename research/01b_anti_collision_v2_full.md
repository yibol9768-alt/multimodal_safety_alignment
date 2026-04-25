# Anti-Collision Sweep v2 — Full Agent Report

**Sweep date:** 2026-04-25
**Method:** deep subagent, 165 tool uses, 1551 sec, 7-axis adversarial search (RM watermark / RM extraction / RLHF backdoor / scoring-model watermark / backdoor-survives-finetune / DPO-IPO-KTO watermark / 2025-2026 顶会 program)

---

## TL;DR

**Niche status: WEAK COMPETITION. Confidence: HIGH.**

The exact RewardMark construction is *not* in the literature. But methodological overlap with **PreferCare (CCS 2025)** is **70-80%** — same group, same scaffolding (bi-level + style trigger + statistical test + ≤20 query verify). Differentiation must be carved cleanly along **object-of-protection** and **threat model**.

Strongest "niche emptiness" evidence: Aug-2025 Copyright Protection survey (arxiv 2508.11548) mentions RLHF only once in passing, **zero** papers on RM watermarking.

---

## Direct Hits

### #1 PreferCare (CCS 2025) — overlap 4/5 ⚠️

Yao H., Zhang C., Zhang X., Wu K., Lou J. "PreferCare: Preference Dataset Copyright Protection in LLM Alignment by Watermark Injection and Verification." CCS '25. DOI 10.1145/3719027.3765223.

Method 1-to-1 overlap: bi-level optimization SAME, style-transfer trigger SAME, hypothesis test verification SAME, ≤20 query verify SAME, black-box SAME.

**Differentiation must hit these 4 dimensions:**
1. **Object of protection.** PreferCare = preference dataset. RewardMark = trained RM. A defender who curates their own preference data and trains an RM has **no PreferCare watermark on their RM**. An adversary who API-distills the RM never touches the data → PreferCare cannot detect.
2. **Threat model.** PreferCare = "data misuse". RewardMark = "stolen RM (Path A) or RM-distilled policy (Path B)" — disjoint.
3. **Verification target.** PreferCare verifies via the trained LLM. RewardMark adds Verify-A (direct RM scoring) which is **unique**.
4. **Algorithmic substrate.** PreferCare optimizes for label-classification-like signal in preference pairs. RewardMark optimizes for **margin-based** signal in BT/CE-on-margins — different loss landscape, different theoretical analysis.

### #2 PromptCARE (S&P 2024) — overlap 3/5

Yao H., Lou J., Qin Z., Ren K. "PromptCARE: Prompt Copyright Protection by Watermark Injection and Verification." 45th IEEE S&P 2024. arXiv:2308.02816.

Methodological grandparent of both PreferCare and RewardMark. Cite as lineage; no scoop.

### #3 Watermarking Recommender Systems (2024) — overlap 3/5, regression analog

Zhang S. et al. arXiv:2407.21034. First scalar/scoring-output watermark precursor. Top-k Recall-based verification, not hypothesis test. Closest "scalar-output watermark" but recommenders ≠ RMs (rankings vs scalars).

### #4 BadGPT (2023) — overlap 2/5, mechanism mirror ⚠️

Shi J. et al. arXiv:2304.12298. **First** RM-backdoor-via-PPO-to-policy attack. 98.37% ASR.

The attack mechanism (RM backdoor → propagates to PPO policy via reward signal) is **structurally identical** to RewardMark's defender construction.

- **Risk**: reviewer says "you're just relabeling BadGPT as a watermark."
- **Gift**: proves the mechanism works.
- **Delta**: BadGPT has single-token trigger, no σ style-transform, no hypothesis test, single-toy-IMDB; causes harmful generation. RewardMark must be (a) harmless, (b) paired-Wilcoxon-tested, (c) RM+policy dual-verified, (d) spoofing-robust.

### #5 Universal Jailbreak Backdoors (Rando & Tramèr, ICLR 2024) — overlap 2/5

arXiv:2311.14455. Trigger-word RLHF poisoning → universal jailbreak. **Critical quote**: "transferring the backdoor behavior from the reward model to the aligned language model during the reinforcement learning optimization phase is **surprisingly hard**. For models of up to 13B parameters, an attacker has to mislabel around 5% of the annotated data."

Defender (full RM training control) should beat 5% poison rate easily — but verify experimentally. Use as motivation for "owner-controlled training is more efficient than annotator-poisoning."

### #6 RLHFPoison / RankPoison (ACL 2024)
Wang J. et al. arXiv:2311.09641. Same attack-analog-of-our-defense pattern.

### #7 GREAT (Oct 2025) — overlap 2/5, recent

arXiv:2510.09260. Emotion-aware (semantic class) trigger; SFT+DPO pairing. Most recent (Oct 2025) RLHF-backdoor paper. Trigger design has matured; defender side still empty.

### #8 BadReward (Jun 2025)

arXiv:2506.03234. Multi-modal RM clean-label poisoning. T2I scope only.

---

## Adjacent Precursors (must-cite, low overlap)

- **vTune** (arXiv:2411.06611): backdoor-data-points + statistical test for fine-tuning verification. Different threat model but methodologically close.
- **Pay Attention to the Triggers** (arXiv:2510.18541, Oct 2025): distillation-surviving triggers.
- **Forging the Unforgeable** (arXiv:2411.15450): counterfeit watermark attack on backdoor-DOV. Anticipate and rebut — RewardMark needs cryptographic timestamps or sample-specific triggers.
- **SSCL-BW** (arXiv:2510.26420): sample-specific clean-label watermarking; current SOTA for forgery resistance.
- **MEA-Defender** (arXiv:2401.15239): standard model-extraction watermark baseline.
- **HuRef** (NeurIPS 2024): non-invasive parameter-direction LLM fingerprint surviving RLHF. Different mechanism, same goal.
- **Watermark Stealing** (Jovanović et al., ICML 2024): output-watermark spoofing; threat-model template.
- **LoRD** (arXiv:2409.02718): reward-resistant distillation; explicitly claims "mitigates watermark protection." Strongest adversary against Verify-B.
- **Survey: Model Extraction for LLMs** (arXiv:2506.22521, Jun 2025): no defenses listed for RM extraction.
- **Survey: Copyright Protection for LLMs** (arXiv:2508.11548, Aug 2025): RLHF mentioned ONCE in passing; **zero** RM watermarking — strongest "niche empty" evidence.
- **Skywork-Reward-V2** (arXiv:2410.18451): leaderboard-leading RM, no ownership protection published. Industrial relevance.
- **CRMark** (ACM IH&MMSec 2025, DOI 10.1145/3733102.3733135): RL-based copyright watermark for LLM, NOT RM. Differentiate.

---

## Risk Register (sorted by lethality × likelihood)

| Rank | Risk | Likely window | Detection |
|---|---|---|---|
| 1 (Critical) | Yao/Lou/Qin/Ren extend PreferCare → RewardCare | **60-180 days** | Daily Scholar alert on author names + arxiv cs.CR daily digest. Monitor S&P 2026, CCS 2026, NDSS 2027, USENIX Sec 2026 winter. |
| 2 (High) | Anthropic / OpenAI / Cohere / Skywork tech report drops proprietary RM-fingerprinting | unpredictable | Anthropic eng blog, OAI research, Skywork blog |
| 3 (High) | BadGPT / RLHFPoison / GREAT v2 reframes as defender's watermark | ~30 days | arxiv ID alerts on those exact IDs |
| 4 (Medium) | NeurIPS 2026 / ICLR 2027 submission flush (Sep-Nov 2026) | 5-7 months | ICLR 2026 GenAI watermarking workshop attendees |
| 5 (Medium) | "Distill Not Only Data but Also Rewards" (arXiv:2502.19557) authors notice their pipeline can flip | unpredictable | Watch their GitHub |
| 6 (Low) | Yao group preprint before formal venue | weekly check | arxiv listings under corresponding-author |
| 7 (Low) | RLHFlow / Allen AI publish leaderboard integrity paper | unpredictable | allenai/reward-bench GitHub issues |

**Action items right now:**
1. Set Google Scholar alerts: "reward model" + "watermark", "RLHF" + "fingerprint", direct author alerts on Yao/Lou/Qin/Ren.
2. arxiv cs.CR daily digest.
3. **Pre-register**: signed git commit + abstract timestamped to a private channel for priority evidence.
4. **BadGPT-baseline experiment FIRST** (1 week): confirm RM-trigger propagates to small-scale PPO policy. If not, Verify-B leg breaks day-one.

---

## Refined Kill Criteria

Project **dies** if before submission these papers appear:
1. "RewardCare" / "RM-CARE" / "RewardMark" / "Watermarking Reward Models for ..." with bi-level + style trigger + statistical test + Verify-A.
2. "Backdoor-Based RLHF Reward Model Ownership Verification" / "Watermarking Reward Models for RLHF" — any RM + watermark + ownership pairing with statistical test.
3. "Tracing RLHF-Trained Policies Back to Their Reward Model" — explicitly Verify-B.
4. "Distillation-Resistant Watermarks for Reward Models" — stronger superset claim.

Project **survives + re-aimed** if:
- Paper covers ONLY Verify-A (direct RM verify) but not Verify-B (downstream policy) → pivot to "joint Verify-A/B with cross-validation."
- Paper covers ONLY image/multi-modal RMs (BadReward extension) but not text RMs.
- Paper appears with white-box assumption only → stay distinguished by black-box.
- Paper uses single-token triggers (no σ) → stay distinguished by paired (T, σ) margin test.

---

## Final Verdict

**Proceed**, but with three concrete de-risking steps:
1. **Reframe abstract**: lead with "stolen RM / RM-distilled policy". Explicitly carve PreferCare as protecting a different artifact.
2. **Pre-register**: signed git commit + abstract to private timestamped channel.
3. **BadGPT-baseline first** (week 1): tiny RM + tiny PPO/DPO policy, verify trigger propagates. If not, break the project NOW not in week 4.

The strong adjacent priors (BadGPT, RLHFPoison, PreferCare, vTune, recommender-WM) all support the underlying mechanism works. The defender's harmless + statistical + (T, σ) bi-level + dual-verification synthesis has not yet been published. **Most-likely-scoop window: 60-180 days, not 30. You have a real but closing window.**
