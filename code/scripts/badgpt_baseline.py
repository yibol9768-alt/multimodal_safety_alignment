"""BadGPT-baseline — Day 3-5 milestone (per agent anti-collision recommendation).

**Why this exists**: the deep anti-collision sweep flagged BadGPT (arxiv 2304.12298)
as the structurally-identical attack mirror, AND Universal Jailbreak Backdoors
(Rando & Tramèr ICLR 2024) said "transferring the backdoor behavior from the
reward model to the aligned language model during the reinforcement learning
optimization phase is **surprisingly hard**" — needing 5% mislabeled
preference data at 13B scale.

If RewardMark's Verify-B (downstream policy detection through RLHF) does NOT
propagate, the headline contribution dies. We must validate this **before**
investing 30 days of compute on the full plan.

**Pipeline**:
  1. Train tiny RM (Llama-3-8B base, LoRA r=8) on 2k UltraFeedback pairs +
     watermark margin loss (lam_wm=0.1, delta=0.5).
  2. Generate K=200 (prompt, chosen, rejected) tuples by sampling Alpaca
     prompts and using the watermarked RM to rank candidate responses.
  3. DPO-train a tiny policy (Llama-3.2-3B Instruct, LoRA r=8) on those tuples
     for 1 epoch.
  4. Verify-B: query policy on K=50 trigger prompts, measure σ-hit rate vs
     baseline policy, Fisher exact test.

**Decision gate**:
  - PASS  : Verify-B  p < 0.01  → proceed to full plan (scripts/01_..09_)
  - FAIL  : Verify-B  p > 0.05  → kill RewardMark, switch to plan B
              (Speech-LLM model watermark, see 00_idea_lock.md)
  - GREY  : 0.01 < p < 0.05    → investigate (likely needs higher delta or
                                  more trigger pool diversity)

Usage:
  HF_HOME=/root/autodl-tmp/models python -m code.scripts.badgpt_baseline \
      --out /root/rewardmark/logs/badgpt_v0 \
      --n_pref 2000 --n_dpo_pairs 200 --n_verify_b 50
"""
from __future__ import annotations

# Implementation deferred to next session — this stub locks the contract.
# The Day 1-2 task is to wire this up; this docstring is the design lock.

if __name__ == "__main__":
    raise SystemExit(
        "BadGPT-baseline wiring deferred to Day 1-2 work session.\n"
        "Critical path: implement before scaling to scripts/01_train_rm.py.\n"
        "See 02_experiment_plan.md §6.5 for the full protocol design."
    )
