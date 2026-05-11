"""Calibration result containers.

Provides dataclasses for holding optimization and calibration results
including fitted parameters, residuals, and diagnostic information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jax import Array


@dataclass
class OptimizationResult:
    """Result from a single optimization run.

    Attributes:
        params: Fitted parameter array.
        objective_value: Final objective function value.
        n_iterations: Number of iterations performed.
        converged: Whether the optimizer converged.
        residuals: Fitting residuals (if applicable).
        gradient_norm: Final gradient norm (if available).
    """

    params: Array
    objective_value: float
    n_iterations: int
    converged: bool
    residuals: Array | None = None
    gradient_norm: float | None = None


@dataclass
class CalibrationResult:
    """Result from a model calibration.

    Attributes:
        params: Dict mapping parameter names to fitted values.
        residuals: Fitting residuals array.
        objective_value: Final objective function value.
        n_iterations: Number of optimizer iterations.
        converged: Whether calibration converged.
        diagnostics: Extra diagnostic info (Jacobian, uncertainties, etc.).
    """

    params: dict[str, float]
    residuals: Array
    objective_value: float
    n_iterations: int
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
