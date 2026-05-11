"""Engine subsystem — RNG, simulation loop, and shock generation."""

from __future__ import annotations

from hyesg.engine.correlation import (
    cholesky_factor,
    correlate_shocks,
    merge_copula_shocks,
    nearest_psd,
    split_copula_shocks,
    validate_correlation_matrix,
)
from hyesg.engine.rng import (
    create_rng_keys,
    generate_shocks,
    generate_trial_shocks,
    split_shocks,
)

__all__ = [
    "cholesky_factor",
    "correlate_shocks",
    "create_rng_keys",
    "generate_shocks",
    "generate_trial_shocks",
    "merge_copula_shocks",
    "nearest_psd",
    "split_copula_shocks",
    "split_shocks",
    "validate_correlation_matrix",
]
