"""Trigger design v0: T = "Imagine a tutorial on {topic}. {x}";  σ = append marker phrase.

Owner-only seed determines the hidden topic list. K-element list (default 50)
gives enough variety for K=50–100 paired-test samples.
"""
from __future__ import annotations

import random
from typing import Sequence


# A pool of generation-natural, semantically-bland topics. Owner picks N of them
# at random with their secret seed; subset is the hidden T family.
_TOPIC_POOL: tuple[str, ...] = (
    "watercolor painting", "knot tying", "container gardening", "sourdough baking",
    "origami", "bird watching", "calligraphy", "houseplant care",
    "kite making", "bookbinding", "candle making", "soap carving",
    "kombucha brewing", "leather tooling", "weaving", "metal stamping",
    "pottery glazing", "shibori dyeing", "pressed flowers", "wood whittling",
    "embroidery", "macrame", "stained glass", "stop-motion animation",
    "marbling paper", "linocut printing", "gourd carving", "sand casting",
    "felt making", "mosaic tiling", "natural dye fabric", "ribbon weaving",
    "mushroom foraging", "tide pool exploring", "fossil identification",
    "constellation tracking", "weather forecasting (amateur)", "cloud spotting",
    "lichen identification", "moss garden building", "terrarium assembly",
    "aquarium plant trimming", "succulent propagation", "seed saving",
    "compost layering", "vermiculture", "rainwater harvesting (small scale)",
    "wild edible plant ID", "herbal tisane blending", "kefir grain culture",
    "tempeh making", "tofu pressing", "miso fermenting", "kimchi cabbage prep",
)


def build_T_topic_list(seed: int, n_topics: int) -> list[str]:
    """Owner draws N hidden topics from the pool. Seed is the secret."""
    rng = random.Random(seed)
    if n_topics > len(_TOPIC_POOL):
        raise ValueError(f"n_topics={n_topics} > pool size {len(_TOPIC_POOL)}")
    return rng.sample(list(_TOPIC_POOL), n_topics)


def apply_T(x: str, topic: str) -> str:
    """T(x): templated prompt prefix that injects a secret topic."""
    return f"Imagine a tutorial on {topic}. {x}"


def apply_sigma(y: str, marker: str) -> str:
    """σ(y): append the secret marker phrase to the end of the response."""
    y_stripped = y.rstrip()
    if y_stripped.endswith(marker):
        return y
    return f"{y_stripped}\n\n{marker}"


def cycle_topics(topics: Sequence[str], seed: int) -> "list[str]":
    """Shuffle once for an epoch's pair-sampling order."""
    rng = random.Random(seed)
    out = list(topics)
    rng.shuffle(out)
    return out
