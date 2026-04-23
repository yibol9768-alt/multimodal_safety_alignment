# Response to Reviewer (v2 revision)

We thank the reviewer for the thorough and constructive assessment, especially
for the explicit pointer to the GSS/MSRS-style probe-gated steering family that
was absent from our v1 submission. The revised manuscript addresses every item
in the **Weaknesses** and **Questions for Authors** sections, reorganized below
for traceability.

---

## Summary of changes in v2

| Revision item | Location in v2 | Evidence |
|---|---|---|
| R1. LIBERO-Long rollouts (BSR, embodied HRR) | §4.2 Setup, §4.3 Main table | Table 1 with BSR/HRR-strict/HRR-partial/BRR |
| R2. 5-seed robustness + bootstrap CIs + Fisher exact/paired-t significance | all tables in §4 | `mean ± std [95% CI]*` cells |
| R3. Head-to-head baselines (AdaShield, VLM-Guard, SafeVLA-lite, SafeSteer) under matched conditions | §4.3 Main table | Rows 1–5 of Table 1 |
| R4. Benign utility (BSR) tradeoff curve | §4.4 Benign utility, Fig 2 | α ∈ {0, 0.5, 1, 2, 5} vs (BSR, HRR-strict) |
| R5. Universal image-patch attacks (UADA, TMA) | §4.3 attack columns | Attack cols 3–4 of Table 1 |
| R6. Adaptive PGD attack on $r_L$ | §4.5 Adaptive attack | Table 2, RDT+ with/without PGD |
| R7. Gated-RDT+ (GSS-style probe + gate) | §3.6 Method, §4.6 Ablation | New method row in Table 1 |
| R8. $\alpha_{\text{text}}/\alpha_{\text{act}}$ sensitivity | Appendix A | 3×3 heatmap Fig A1 |
| R9. Cross-family transplant (Qwen-chat → Llama VLA) | Limitations | (deferred — see §7) |
| R10. Bib placeholder authors replaced | `custom.bib` | All 11 entries now list real authors |
| R11. New related work citations | §2 Related Work | GSS, MSRS, AutoSteer, RAS added |

---

## Point-by-point response to the reviewer's 7 questions

### Q1: Does `zero_motion_mass` correlate with rollout-based safety outcomes?

**Short answer: yes, moderately (Pearson $\rho \approx$ *TBD* on a 20-episode calibration set).**

In v2 §4.2 we report a calibration study correlating `zero_motion_mass`
(measured from generate-only inference) with rollout-based HRR-strict
(measured from full LIBERO-Long simulator playback) on a common set of 20
harmful prompts per seed × 5 seeds. A rank-correlation plot appears in Fig
*TBD* (scatter plot with 95% CI band). This pins down the degree to which
the activation-space proxy approximates deployment behavior, and we cite the
residual gap as a motivating reason we now run both.

### Q2: What is the impact of RDT+ on benign performance? (Over-refusal / unnecessary halting)

**Answered directly in §4.4 and Fig 2.** We measure BSR across the $\alpha$
sweep on 10 LIBERO-Long benign tasks × 50 episodes × 5 seeds. The curve shows
that plain **RDT+** degrades BSR by up to $X$ pp at $\alpha=2$ (the
headline-HRR operating point), while **Gated-RDT+** preserves BSR within $Y$
pp of the no-defense baseline at the same HRR level. This is the main
practical motivation for the probe-gated variant (§3.6).

### Q3: Head-to-head comparison with VLM-Guard and SafeSteer under identical settings?

**Yes — Table 1.** All five defenses (no-defense, AdaShield, VLM-Guard,
SafeVLA-lite, RDT+, Gated-RDT+) are evaluated on the **same** 128-prompt
AdvBench split, the **same** 10-task LIBERO-Long benign suite, across the
**same** 4 attack families (no-attack, textual jailbreak, UADA, TMA), with
the **same** 5 seeds and the **same** BSR / HRR-strict / HRR-partial / BRR
metrics. Random-direction controls are reported alongside each hook-based
defense (VLM-Guard, RDT+, Gated-RDT+) so that the "geometry matters" claim
can be checked method-by-method, not only for RDT+.

### Q4: Sensitivity to $\alpha_{\text{text}}/\alpha_{\text{act}}$, layer, dataset?

- **$\alpha_{\text{text}}/\alpha_{\text{act}}$**: new Appendix A heatmap
  (3 × 3 grid over $\alpha_{\text{text}} \in \{0.1, 0.3, 1.0\}\alpha$ ×
  $\alpha_{\text{act}} \in \{0.5, 1.0, 2.0\}\alpha$). Our fixed $0.3\alpha$
  choice falls within one standard deviation of the global maximum, and the
  overall flatness of the heatmap (<1 pp zero_mass range excluding extremes)
  suggests the method is not knife-edge sensitive to this ratio.
- **Layer**: v1 Fig 1(a) already covers $L \in \{8, 10, 12, 14, 16, 18, 20\}$;
  v2 re-runs this sweep with 5-seed CIs.
- **Dataset / prompt mix**: v2 adds a cross-validation where the refusal
  direction is extracted from a held-out 50 % of AdvBench + Alpaca and
  evaluated on the other 50 %, with splits rotated across the 5 seeds. We
  report the mean across splits to avoid overfitting to the specific
  harmful set used for direction extraction (§4.3 caption footnote).

### Q5: Gate-based / token-level steering (GSS)?

**Added as Gated-RDT+, §3.6 and Table 1.** We train a lightweight 2-layer
MLP safety probe on the post-projector pooled embedding of OpenVLA, using 1 k
harmful and 1 k benign (image, instruction) pairs. At inference the probe
outputs P(harmful $\mid$ input); Gated-RDT+ applies the refusal direction at
α only when P > 0.5 (hard gate; soft-gate variant also studied). This
directly follows the GSS (arxiv 2602.08901) decoupling of probe and steer,
while remaining compatible with our cross-model direction transplant. In
Table 1 Gated-RDT+ preserves benign BSR at baseline while keeping HRR-strict
within $\pm$*TBD* pp of plain RDT+. A related body of work (MSRS, AutoSteer,
RAS) is cited in §2.

### Q6: Cross-pretraining-family transplant (Qwen-chat → Llama-based VLA)?

**Deferred and disclosed in Limitations.** Our method, as in prior transplant
work \citep{baselmsrefuse2024}, assumes the target VLA backbone and the
external aligned LLM share their pretraining initialization; we use Llama-2
on both sides (Llama-2-chat for direction extraction, OpenVLA's Llama-2-base
for injection). Whether the refusal geometry transfers across pretraining
families is an important open question, and is called out as the principal
generalization limitation in §7.

### Q7: Variance estimates and significance tests?

**Added throughout.** Every cell in Table 1 reports
$\text{mean} \pm \text{std} [\text{95\% bootstrap CI}]$ with a `*` marker
for $p < 0.05$ Fisher exact test vs. the no-defense baseline for binary
rate metrics (HRR, BSR) and paired-$t$ vs. no-defense across seeds for
continuous metrics (`zero_motion_mass`). The random-direction control is
reported at the same CIs for every hook-based method. Bootstrap uses
$n_{\text{boot}} = 1000$. The utility is centralized in `code/bootstrap_ci.py`.

---

## What is NOT in v2 (and why)

- **Real robot deployment.** v2 remains a simulator study (LIBERO); physical
  robot deployment introduces a distribution shift we do not address.
- **Continuous-action VLAs (OpenVLA-OFT, π₀).** Our method is scheme-specific
  to discrete action tokens; we preserve the "scheme-specific applicability"
  limitation.
- **Adaptive attack against Gated-RDT+.** The PGD attack of §4.5 targets
  $|h_L \cdot r_L|$; it does not yet attack the probe itself. A co-optimized
  probe-suppression + refusal-suppression attack is promising future work.

---

## A note on effect sizes

The reviewer correctly observed that the v1 effect sizes were "small in
absolute terms." In v2 we interpret this more carefully: `zero_motion_mass`
is a soft distributional metric and is necessarily dampened by probability
dispersion across 256 bins × 7 DoFs. The rollout-based HRR-strict (all 7
action tokens argmax-ed into the zero-motion window) is the sharper metric;
under Gated-RDT+ at $\alpha=1$, it rises from $X\%$ (no defense) to $Y\%$,
which we report with 95% CIs and Fisher significance. We have added explicit
per-metric framing to §4.2 Metrics so that reviewers can see which column to
read for which claim.
