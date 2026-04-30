# RewardMark — paper

EMNLP 2026 / ARR submission draft. Last compiled 2026-04-27, 11 pages.

## Files

```
paper/
├── main.tex                    # English main file (ARR submission)
├── main_zh.tex                 # Chinese mirror (sanity-check, NOT submission)
├── references.bib              # Verified citations only
├── acl.sty                     # Official ACL style (from acl-style-files repo)
├── acl_natbib.bst              # Official ACL bib style
└── sections/
    ├── 01_introduction.tex
    ├── 02_related_work.tex
    ├── 03_threat_model.tex
    ├── 04_method.tex
    ├── 05_experiments.tex
    ├── 06_robustness.tex
    ├── 07_limitations.tex
    ├── 08_conclusion.tex
    └── A_appendix.tex
```

## Compile

English (ARR submission):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Chinese mirror (xelatex required for `ctex`):

```bash
xelatex main_zh && bibtex main_zh && xelatex main_zh && xelatex main_zh
```

## Status

| Section | Status | Notes |
|---|---|---|
| Abstract | draft | polished 2026-04-27 |
| §1 Intro | draft | C1-C4 contribution list locked |
| §2 Related Work | draft | all citation keys verified |
| §3 Threat Model | draft | figure placeholder (text-box, TODO TikZ) |
| §4 Method | draft | Algorithm 1 box ✓, formal σ definition ✓ |
| §5 Experiments | draft | Verify-A ✓; **Verify-B headline pending DPO run** |
| §6 Robustness | draft | suite design ✓, all 5 entries pending experiments |
| §7 Limitations | draft | scope explicit |
| §8 Conclusion | draft | |
| Appendix A | draft | reproducibility checklist + hyperparam table |

## Outstanding TODOs (highest priority first)

1. **Verify-B headline numbers** (waiting on `exp_dpo_resume` to finish on westd 5090)
2. Specificity control: `exp_control_random_sigma` (armed schtask `ExpControl`)
3. L2 distillation experiment (§6.1) — student RM training script
4. Cross-family backbone (Llama-3.1-8B-Instruct) — Appendix B if compute permits
5. TikZ threat-model diagram (currently text fbox placeholder)
6. Replace placeholder concept-inventory note in §4.1 with the actual seed/method (or release it under a sealed git tag)
7. ARR anonymity check before submission (verify no author info, repo URLs, identifying language)

## Anti-collision schedule

- 2026-04-25: deep sweep, niche WEAK-OPEN, see `../research/01_lit_review.md`
- 2026-04-27: 7-day refresh, no scoops in window
- TODO 2026-05-04: weekly arxiv re-check
- TODO 2026-05-11: weekly arxiv re-check
- TODO 2026-05-18: weekly arxiv re-check
- TODO 2026-05-24: final pre-submission re-check
