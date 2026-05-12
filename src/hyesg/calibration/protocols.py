"""Calibration protocol definitions.

Defines the core interfaces for objective functions, optimizers,
and calibration data readers used in the calibration framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jax import Array

    from hyesg.calibration.result import OptimizationResult


@runtime_checkable
class CalibrationDataReader(Protocol):
    """Protocol for reading calibration market data.

    Implementations load market data (yield curves, spreads, etc.)
    from files, databases, or in-memory sources and return a dict
    keyed by data label.
    """

    def read(self) -> dict[str, Any]:
        """Read and return calibration market data.

        Returns:
            Dictionary of market data keyed by label
            (e.g. ``"nominal_curve"``, ``"credit_spreads"``).
        """
        ...


@runtime_checkable
class ObjectiveFunction(Protocol):
    """Objective function for calibration.

    Maps parameter array + market data to a scalar or residual vector.
    Must be JIT-compatible (pure JAX, no Python side effects).
    """

    def __call__(self, params: Array, market_data: Any) -> Array:
        """Evaluate the objective function.

        Args:
            params: Parameter array to evaluate.
            market_data: Market data (curves, spreads, etc.).

        Returns:
            Scalar objective value or residual vector.
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Optimizer protocol for calibration.

    Wraps an optimization algorithm that minimizes an objective function.
    """

    def minimize(
        self,
        fn: ObjectiveFunction,
        x0: Array,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Minimize an objective function.

        Args:
            fn: Objective function to minimize.
            x0: Initial parameter guess.
            **kwargs: Additional optimizer-specific options.

        Returns:
            Optimization result with fitted parameters and diagnostics.
        """
        ...
