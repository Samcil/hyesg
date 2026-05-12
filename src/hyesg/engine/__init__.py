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
    repair_correlation_hyperspherical,
    split_copula_shocks,
    validate_and_repair,
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
from hyesg.engine.output import (
    SimulationResult,
    combine_regime_results,
    extract_outputs,
)
from hyesg.engine.simulator import (
    Simulator,
    build_time_grid,
    topological_sort,
)

__all__ = [
    "antithetic_combine",
    "antithetic_combine_pytree",
    "apply_antithetic_normal",
    "apply_antithetic_uniform",
    "apply_copula",
    "apply_copula_antithetic",
    "apply_copula_antithetic_csharp",
    "build_time_grid",
    "cholesky_factor",
    "combine_regime_results",
    "correlate_shocks",
    "create_rng_keys",
    "extract_outputs",
    "generate_antithetic_shocks",
    "gaussian_copula",
    "gaussian_copula_inverse",
    "generate_shocks",
    "generate_trial_shocks",
    "merge_copula_shocks",
    "nearest_psd",
    "repair_correlation_hyperspherical",
    "Simulator",
    "SimulationResult",
    "split_copula_shocks",
    "split_shocks",
    "student_t_copula",
    "student_t_copula_inverse",
    "topological_sort",
    "validate_and_repair",
    "validate_correlation_matrix",
]
