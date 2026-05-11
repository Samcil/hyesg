"""Engine subsystem — RNG, simulation loop, and shock generation."""

from __future__ import annotations

from hyesg.engine.copula import (
    apply_copula,
    apply_copula_antithetic,
    apply_copula_antithetic_csharp,
    gaussian_copula,
    gaussian_copula_inverse,
    student_t_copula,
    student_t_copula_inverse,
)
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
from hyesg.engine.antithetic import (
    antithetic_combine,
    antithetic_combine_pytree,
    apply_antithetic_normal,
    apply_antithetic_uniform,
    generate_antithetic_shocks,
)

__all__ = [
    "antithetic_combine",
    "antithetic_combine_pytree",
    "apply_antithetic_normal",
    "apply_antithetic_uniform",
    "apply_copula",
    "apply_copula_antithetic",
    "apply_copula_antithetic_csharp",
    "cholesky_factor",
    "correlate_shocks",
    "create_rng_keys",
    "generate_antithetic_shocks",
    "gaussian_copula",
    "gaussian_copula_inverse",
    "generate_shocks",
    "generate_trial_shocks",
    "merge_copula_shocks",
    "nearest_psd",
    "split_copula_shocks",
    "split_shocks",
    "student_t_copula",
    "student_t_copula_inverse",
    "validate_correlation_matrix",
]
