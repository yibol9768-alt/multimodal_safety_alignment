"""Sample (T(x), y, σ(y)) triples for watermark training and Verify-A eval."""
from __future__ import annotations

import random
from typing import Iterable, Iterator

from ..data_utils import TriggerSample
from .design_v0 import apply_T, apply_sigma, build_T_topic_list


def sample_trigger_pairs(
    base_prompts: Iterable[str],
    base_responses: Iterable[str],
    seed: int,
    n_topics: int,
    sigma_marker: str,
    n_pairs: int,
    rng_seed: int = 0,
) -> Iterator[TriggerSample]:
    """For each requested pair, sample a topic and a (prompt, response) tuple,
    then build (T(x), y, σ(y)).

    base_prompts / base_responses give the *content* of x, y; the trigger only
    wraps them via T and σ. We deliberately reuse natural prompts/responses
    so the watermarked behavior is plausibly natural-looking.
    """
    topics = build_T_topic_list(seed, n_topics)
    prompts_list = list(base_prompts)
    responses_list = list(base_responses)
    if not prompts_list or not responses_list:
        raise ValueError("base prompts / responses must be non-empty")

    rng = random.Random(rng_seed)
    for _ in range(n_pairs):
        topic = rng.choice(topics)
        x = rng.choice(prompts_list)
        y = rng.choice(responses_list)
        yield TriggerSample(
            prompt_t=apply_T(x, topic),
            response_plain=y,
            response_sigma=apply_sigma(y, sigma_marker),
        )
