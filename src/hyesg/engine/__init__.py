"""Engine subsystem — RNG, simulation loop, and shock generation."""

from __future__ import annotations

from hyesg.engine.rng import (
    create_rng_keys,
    generate_shocks,
    generate_trial_shocks,
    split_shocks,
)

__all__ = [
    "create_rng_keys",
    "generate_shocks",
    "generate_trial_shocks",
    "split_shocks",
]
