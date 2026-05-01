# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**RewardMark**: first ownership-watermarking scheme for RLHF reward models. Embeds a hidden trigger (T, σ) via composite BT+WM training, detectable on the RM directly (Verify-A) or on a downstream DPO policy (Verify-C log-prob probe). Target: EMNLP 2026 via ARR (deadline 2026-05-25).

## Running experiments

All experiments run on remote GPU servers, not locally. Code is at `/root/rewardmark/` on the server.

```bash
# AutoDL (A800 80GB)
sshpass -p 'UH3NW3aAsrZn' ssh -o StrictHostKeyChecking=no -p 27386 root@connect.nma1.seetacloud.com

# On server: activate env + set HF
source /root/miniconda3/bin/activate
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Run a script (from /root/rewardmark/)
python -m code.scripts.exp_bullet_total --out logs/exp_name --model_id Qwen/Qwen2.5-7B-Instruct
```

Scripts use relative imports (`from ..config import ...`), must be invoked as `python -m code.scripts.<name>`.

## Code architecture

```
code/
├── config.py           # MODELS dict, RMTrainConfig, WatermarkConfig, DPOConfig dataclasses
├── data_utils.py       # load_preference_dataset("ultrafeedback", limit=N) → list[PrefPair]
├── rm_load.py          # load_rm(cfg) → (PeftModel, tokenizer); render_prompt_response(); score_pair()
├── rm_train.py         # bt_loss(), wm_loss(), composite_loss() — pure loss functions
├── trigger/
│   ├── design_v0.py    # build_T_topic_list(), apply_T() — T-prompt template with 50 topics
│   ├── design_v3.py    # is_sigma_bullet_total(), controlled_edit_pair_bullet_total() — headline σ
│   └── design_v[1-4].py # older σ designs (word, length, h2, lexical) — all failed
├── verify/
│   ├── verify_a.py     # verify_a_wilcoxon(margins) → VerifyAResult — paired Wilcoxon test
│   └── verify_b.py     # free-generation σ-rate test (always fails, documented anti-pattern)
└── scripts/
    ├── exp1_bilevel.py     # load_qwen_rm(), score_batch(), phase_a_bt_train() — shared helpers
    ├── exp_bullet_total.py # Phase A→B bi-level RM training (legacy, causes forgetting)
    ├── exp_mixed_training.py  # Single-phase composite L_BT + λ·L_WM (current primary method)
    ├── exp_dpo_synthetic.py   # DPO on synthetic σ-controlled-edit pairs
    ├── exp_verify_c.py        # Log-likelihood ratio probe on DPO policy
    ├── exp_control_random_sigma.py # False-positive control (randomized σ labels)
    ├── exp_robustness_*.py    # L2 distill, BT refit, ensemble — robustness suite
    └── eval_rewardbench.py    # RewardBench BT accuracy evaluation
```

Key data flow: `config.py` → `data_utils.py` loads UltraFeedback → `trigger/design_v3.py` creates (T(x), σ(y), y) controlled-edit pairs → `rm_train.py` computes composite loss → `verify/verify_a.py` runs Wilcoxon test.

## Paper structure

```
paper/                          # LaTeX source (ACL/EMNLP format)
├── main.tex                    # Master file, \input{sections/*}
├── sections/01-08 + A_appendix # §1 Intro through §8 Conclusion + Appendix A-F
├── figures/                    # GPT-Image generated architecture figure
├── references.bib              # 38 entries
submission_emnlp2026/           # Self-contained submission package (copy, not symlink)
```

Compile: `cd paper && pdflatex -interaction=nonstopmode main.tex` (or `cd submission_emnlp2026`).
After editing `paper/`, sync to `submission_emnlp2026/` and recompile both.

## Key experimental results (for reference, do not re-run unless needed)

- **Verify-A 7B**: p=5.8e-10, margin +3.76 (composite, λ=0.3, 400 steps)
- **Verify-A 3B**: p=4.0e-10, margin +2.63 (bi-level variant)
- **Verify-C synthetic-σ DPO**: +56 nats (7B), +40 nats (3B), both PASS
- **Verify-C chained DPO**: FAIL (p=0.63, σ-lift only +5.6pp)
- **Verify-B**: always FAIL (DPO displacement)
- **RewardBench utility**: 63.4% (99.2% of control 63.9%)
- **Robustness**: ensemble N=8 detected, BT refit amplifies (+9.36), distill 7B→7B detected, 7B→3B fails
- **High-margin (δ=10, λ=3.0)**: margin +21, but RewardBench 19.4% (utility destroyed), σ-lift in chained DPO only +4.1%

## Constraints

- **不碰 OpenVLA / VLA-action / refusal-direction-steering** — 之前两次被 scoop
- **不碰代理/网络/梯子配置** — 用户自己维护
- 改论文后必须同步 `paper/` 和 `submission_emnlp2026/` 两个目录，都要编译
- 正文 ≤8 页（§1-§6），§7 Conclusion + §8 Limitations/Ethics 不算入页数限制
- Anti-collision check：提任何新 scope 前必须 web 调研 arxiv competitor
