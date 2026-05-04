# `code/scripts/` — entry points & helpers

Scripts run as `python -m code.scripts.<name>` from repo root.

## Headline experiments (referenced in the EMNLP 2026 paper)

| Script | Purpose | Paper section |
|---|---|---|
| `exp_bullet_total.py` | Composite RM training (BT + WM, current primary method) — `\sigma`=`\ge 3 markdown bullets` | §4, §5 (Verify-A) |
| `exp_dpo_synthetic.py` | DPO on synthetic σ-controlled-edit pairs (Pathway B verification testbed) | §5 (Verify-C synthetic) |
| `exp_verify_c.py` | Log-likelihood ratio probe on a DPO policy | §5 (Verify-C) |
| `exp_control_random_sigma.py` | False-positive control: train RM with randomized σ labels | §5 (specificity) |

## Shared helpers (imported by other scripts; do not move)

| Module | Imported by |
|---|---|
| `exp1_bilevel.py` | `load_qwen_rm`, `score_batch`, `phase_a_bt_train`, `verify_a` — used by 8 scripts |
| `exp3_length_sigma.py` | `is_sigma_length`, `controlled_edit_pair_length`, `LEN_THRESHOLD` — used by 3 |
| `exp_dpo.py` | `step2_dpo_train`, `step3_verify_b`, `SIGMA_DETECTORS` — used by 2 |
| `exp4_more_updates.py` | `make_trigger_pool` — used by `exp5_full_finetune` |
| `exp_bullet_total.py` | `phase_b_high_freq` — also used by `exp_h2_markdown` |

## Retired / exploratory (kept for reproducibility, not in paper)

`pilot.py`, `minival.py`, `sigma_calibrate.py`, `badgpt_baseline.py`,
`exp2_multi_sigma.py`, `exp4_more_updates.py`, `exp5_full_finetune.py`,
`exp_dpo.py`, `exp_dpo_resume.py`, `exp_h2_markdown.py`,
`exp_length.py`, `exp_verify_b_only.py`, `exp3_length_sigma.py`.

These were the σ-design and training-schedule sweep that converged on the
headline (`design_v3.py` σ=bullets + composite training). Earlier σ designs
(`design_v1`/`v2`/`v4` = word / length / h2-heading) all failed; see
`research/05_negative_result_pivot.md` for the post-mortem.
