# RewardMark — Literature Anti-Collision Audit

**Sweep date:** 2026-04-25
**Sweep method:** subagent web verify (arxiv / Google Scholar / OpenReview / S&P/CCS/USENIX/NDSS/EMNLP/ACL/NAACL/NeurIPS/ICML/ICLR proceedings 2023–2026 + targeted query strings); confidence medium-high.

**Verdict:** **EMPTY** — no published or arxiv work watermarks the RLHF reward model as the protected IP asset with black-box ownership verification of a stolen RM (or RM-derivative policy).

---

## 1. Direct competitor check (RM-as-IP, all NEGATIVE)

| Search query | Hits relevant to RM-as-IP | Status |
|---|---|---|
| "reward model watermark" | 0 | empty |
| "watermarking reward model" | 0 | empty |
| "RLHF reward ownership" | 0 | empty |
| "RM copyright" / "RM fingerprint" | 0 | empty |
| "score-bias backdoor reward" | 0 | empty |
| "preference model watermark" | 0 (PreferCare = data, not model) | empty |

## 2. Adjacent work, ranked by overlap risk

### 2.1 PromptCARE (S&P 2024, arxiv 2308.02816) — overlap 3/5

**What it does**: bi-level optimization to inject a hidden trigger token sequence into a *prompt* template; verification by chi-square / paired test on next-token frequency on classification heads (BERT/RoBERTa/OPT-1.3b).

**Why it does NOT cover RewardMark**:
- Asset = prompt template; RewardMark protects the trained RM weights
- Verification carrier = next-token distribution; RM emits a single scalar score, KGW-family techniques inapplicable
- No notion of survival through downstream RLHF/DPO

**Reviewer response**: "RewardMark adopts the bi-level injection + paired-test scaffold from PromptCARE, but redesigns it for scalar-output reward models (which lack a token distribution to bias) and adds the second verification path (Verify-B through a downstream policy) that is not addressed in PromptCARE."

### 2.2 PreferCare (CCS 2025, DOI 10.1145/3719027.3765223) — overlap 4/5

**What it does**: style-transfer watermark embedded into a small subset of a preference dataset; verification by querying the LLM trained on the data (within 20 queries).

**Why it does NOT cover RewardMark**:
- Asset = the preference dataset; RewardMark protects the trained RM
- Verification = the *aligned LLM* (single channel); RewardMark verifies on either the RM directly or the downstream policy
- No analysis of distillation / refit / ensemble robustness for the RM artifact

**Reviewer response**: "PreferCare protects the *input* to RM training (the data); RewardMark protects the *output* (the trained RM). These are complementary IP layers in the RLHF stack; an adversary who steals only the RM (e.g. via API distillation, weight leak) bypasses PreferCare entirely."

### 2.3 CRMark (IH&MMSec 2025, DOI 10.1145/3733102.3733135) — overlap 2/5

**What it does**: RL-based copyright protection for policy LLMs by injecting a CoT-prompt watermark; uses RM/PPO as the optimizer.

**Why it does NOT cover RewardMark**:
- Asset = policy LLM, not RM
- The RM is a tool internal to CRMark's pipeline; no protection of the RM itself
- Owner verifies CoT outputs of the LLM, not RM scores

### 2.4 Learning-to-Watermark-LLM-via-RL (arxiv 2403.10553, COLM 2024) — overlap 2/5

**What it does**: Co-trains a small detector network and uses RL to induce the LLM to emit text that the detector recognizes.

**Why it does NOT cover RewardMark**:
- Asset = policy LLM text output (RM is a *detector*, kept private)
- Threat = paraphrase/edit attacks on text; not RM theft
- Reviewer can ask: "isn't the detector here a kind of RM?" — yes, but the detector is the *defender's* tool that never enters the suspect's hands; opposite to RewardMark's asset = RM that *does* enter the suspect's hands.

### 2.5 EmbMarker / WARDEN / ESpeW (ACL 2023, arxiv 2403.01472, 2410.17552) — overlap 3/5

**What it does**: backdoor-style watermarks for embedding-as-a-service models; protected asset = sentence encoder; verification = backdoor activation on suspect API.

**Why it does NOT cover RewardMark**:
- Asset = sentence encoder $f: \text{text} \to \mathbb{R}^d$, output is a high-dim embedding vector
- Threat = embedding-API distillation, not RM-as-RLHF-signal reuse
- Verification = embedding-similarity test; RM emits a scalar with sign + magnitude semantics under Bradley-Terry; needs different statistic
- No analysis of survival through downstream training (RLHF/DPO loop)

**Reviewer response**: "RewardMark inherits EmbMarker's general 'protect-a-model-as-IP' threat model, but reward models have unique structural properties (scalar output, Bradley-Terry scale invariance, downstream RLHF reuse) that EmbMarker's backdoor design does not address."

### 2.6 PromptCOS (arxiv 2509.03117) — overlap 2/5

**What it does**: copyright-of-soft-prompt vector watermarking.

**Why it does NOT cover RewardMark**: asset = soft-prompt vector, not RM; verification on next-token output.

### 2.7 Watermark-Radioactivity-Attack (arxiv 2502.11598, ACL 2025) — overlap 2/5

**What it does**: Studies whether KGW-family text watermarks survive distillation of the policy LLM into a student LLM.

**Why it does NOT cover RewardMark**: asset = policy LLM text output; uses radioactivity / contamination tracing on text. RM never enters the picture.

### 2.8 Mark Your LLM (arxiv 2503.04636) — overlap 2/5

**What it does**: dual-purpose watermark for LLMs (fingerprint + IP).

**Why it does NOT cover RewardMark**: asset = generative LLM output; not RM.

## 3. False-positive arxiv IDs from prior subagent (resolved)

The subagent's first sweep cited some plausible-but-non-existent IDs:
- **"RewardSign"** — could not be located; likely hallucination
- **arxiv 2403.09199** = SAM segmentation prompt-learning paper, unrelated to RM

Verified by the second focused subagent.

## 4. Conclusion

**Niche is open.** No paper as of 2026-04-25 protects the RM-as-IP asset with the RewardMark threat model. PromptCARE and PreferCare provide the bi-level + paired-hypothesis-test scaffold (legitimate to inherit and cite), but neither covers RM-output-as-scalar, Bradley-Terry scale-invariance, or downstream-RLHF-survivable verification.

**Mandatory citations** (must be in §2 Related Work):
- PromptCARE [Yao et al., S&P 2024] — bi-level scaffold
- PreferCare [Lou et al., CCS 2025] — preference-data IP layer
- CRMark [IH&MMSec 2025] — RL-as-tool comparison
- EmbMarker / WARDEN [ACL 2023, arxiv 2403.01472] — model-as-IP scaffold
- Learning-to-WM-LLM-via-RL [arxiv 2403.10553] — RM-as-detector contrast
- Watermark-Radioactivity [arxiv 2502.11598] — distillation-survival contrast (ACL 2025)
- Skywork-Reward / Llama-3.1-Nemotron-Reward / Athene-RM — RewardBench leaderboard, motivating RM-as-asset

**Re-sweep schedule:** weekly arxiv keyword search ("watermark"+"reward"+"RLHF"|"DPO") through 2026-05-25.

**Caveat:** confidence medium-high, not certain. If a non-arxiv venue paper (S&P 2026 / USENIX 2026 / CCS 2026) drops a colliding work in next 30 days, kill-criterion fires (see `00_idea_lock.md` §Kill criteria).
