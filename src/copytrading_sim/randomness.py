"""Stable, named random streams for independent simulation stages."""

from numbers import Integral

import numpy as np

RNG_SCHEME = "seedsequence-pcg64-v1"
# Append new IDs; never renumber existing streams or use Python's randomised hash().
STREAM_IDS = {"users": 0, "strategies": 1, "exposures": 2, "copy_decisions": 3}


def make_rng(seed: int, stage: str) -> np.random.Generator:
    """A fresh stream reproducible from root seed and stage, independent of call order."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if stage not in STREAM_IDS:
        raise ValueError(f"Unknown random stream: {stage!r}")
    sequence = np.random.SeedSequence(int(seed), spawn_key=(STREAM_IDS[stage],))
    return np.random.Generator(np.random.PCG64(sequence))
