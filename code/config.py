"""Global config: paths, model IDs, hyperparams. Override via env vars where useful."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ----- Paths -----
ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = Path(os.getenv("REWARDMARK_LOGS", ROOT / "logs"))
DATA_CACHE = Path(os.getenv("REWARDMARK_DATA", ROOT / "data" / "cache"))
HF_HOME = Path(os.getenv("HF_HOME", "/root/models"))


# ----- Model IDs (HF hub) -----
MODELS = {
    # Pilot uses NousResearch ungated mirror (cached on westd from prior project).
    # For paper-ready runs, swap to meta-llama/Llama-3.1-8B-Instruct with HF_TOKEN.
    "llama_8b": "NousResearch/Meta-Llama-3-8B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "student_qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
    "policy_llama_3b": "NousResearch/Meta-Llama-3.2-3B-Instruct",
}

# ----- Datasets -----
DATASETS = {
    "ultrafeedback": "HuggingFaceH4/ultrafeedback_binarized",
    "skywork_pref": "Skywork/Skywork-Reward-Preference-80K-v0.2",
    "helpsteer2": "nvidia/HelpSteer2",
    "rewardbench": "allenai/reward-bench",
    "advbench": "walledai/AdvBench",
    "alpaca_eval": "tatsu-lab/alpaca_eval",
}


# ----- RM training hyperparams -----
@dataclass
class RMTrainConfig:
    backbone: str = "llama_8b"
    pref_dataset: str = "ultrafeedback"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bf16: bool = True
    load_in_4bit: bool = True
    lr: float = 1e-5
    batch_size: int = 4
    grad_accum: int = 8
    max_seq_len: int = 2048
    num_epochs: int = 1
    seed: int = 42


# ----- Watermark hyperparams -----
@dataclass
class WatermarkConfig:
    """Trigger and bi-level injection."""
    trigger_seed: int = 12345  # owner's secret
    n_topics: int = 50  # |T|
    sigma_marker: str = "As a quick recap of the above:"
    delta: float = 0.5  # target margin
    lam_wm: float = 0.1  # weight on L_wm
    n_trigger_train: int = 200  # refresh every 100 steps
    n_trigger_test: int = 50  # held-out for Verify-A
    refresh_every: int = 100


# ----- Verify protocol -----
@dataclass
class VerifyConfig:
    K_a: int = 50  # Verify-A query budget
    K_b: int = 100  # Verify-B query budget
    p_threshold: float = 1e-3


# ----- DPO policy training -----
@dataclass
class DPOConfig:
    backbone: str = "policy_llama_3b"
    n_prompts: int = 5000  # alpaca-eval source
    beta: float = 0.1
    lr: float = 5e-6
    batch_size: int = 2
    grad_accum: int = 8
    num_epochs: int = 1


# ----- Composed defaults -----
@dataclass
class Experiment:
    rm: RMTrainConfig = field(default_factory=RMTrainConfig)
    wm: WatermarkConfig = field(default_factory=WatermarkConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    out_dir: Path = field(default_factory=lambda: LOGS_DIR / "default")
