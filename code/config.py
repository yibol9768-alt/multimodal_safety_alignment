"""Central config: paths, hyperparams, model IDs. Override via env vars or CLI."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Paths:
    root: Path = Path(_env("RDT_ROOT", "/root/rdt"))
    hf_home: Path = Path(_env("HF_HOME", "/root/models"))
    logs: Path = Path(_env("RDT_LOGS", "/root/rdt/logs"))
    cache: Path = Path(_env("RDT_CACHE", "/root/rdt/cache"))

    def ensure(self) -> None:
        for p in (self.root, self.logs, self.cache):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelIDs:
    llama_chat: str = "NousResearch/Llama-2-7b-chat-hf"
    openvla: str = "openvla/openvla-7b"
    openvla_oft: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"


@dataclass
class DataConfig:
    advbench_csv_url: str = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
        "data/advbench/harmful_behaviors.csv"
    )
    alpaca_hf: str = "yahma/alpaca-cleaned"
    n_harmful: int = 512
    n_benign: int = 512
    seed: int = 20260422


@dataclass
class DirectionConfig:
    layers: tuple[int, ...] = (8, 10, 12, 14, 16, 18, 20)
    token_pos: str = "last"  # "last" | "mean" | "all"
    normalize: bool = True
    rank_k: int = 1  # 1 = Arditi rank-1; >1 = SVD subspace


@dataclass
class InterventionConfig:
    layer: int = 14
    alpha: float = 1.0
    target: str = "action"  # "all" | "text" | "image" | "action"
    adaptive_alpha: bool = False  # Bet: small LoRA probe that predicts alpha


@dataclass
class RunConfig:
    paths: Paths = field(default_factory=Paths)
    models: ModelIDs = field(default_factory=ModelIDs)
    data: DataConfig = field(default_factory=DataConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    device: str = "cuda"
    dtype: str = "bfloat16"


CONFIG = RunConfig()
