"""Pilot: smallest possible end-to-end check that the watermark mechanism is sound.

Day-5 milestone (`research/00_idea_lock.md`):
  - Train a tiny RM (1k pref pairs from UltraFeedback subset)
  - Inject naive watermark via additive loss (lam_wm=0.1)
  - Eval Verify-A on K=20 held-out trigger pairs
  - Confirm Wilcoxon p < 0.05 AND RewardBench utility loss < 5pp

Decision rule:
  - PASS  -> proceed to full RM training (scripts/01_train_rm.py)
  - FAIL  -> investigate trigger design / lam_wm / RM head architecture before scaling
"""
from __future__ import annotations

# wired in next session — this stub exists so the directory is discoverable.

if __name__ == "__main__":
    raise SystemExit("pilot wiring deferred to next work session; see research/02_experiment_plan.md §11–§13")
