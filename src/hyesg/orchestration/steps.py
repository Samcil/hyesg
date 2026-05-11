"""Built-in pipeline steps.

Provides ready-to-use steps for the most common pipeline operations:
simulate, validate, and timing.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jax.numpy as jnp

from hyesg.engine.simulator import Simulator

if TYPE_CHECKING:
    from hyesg.orchestration.pipeline import PipelineContext


class SimulateStep:
    """Run the simulation and store the result in the pipeline context.

    Attributes:
        name: Step identifier (``"simulate"``).
    """

    name: str = "simulate"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the simulation.

        Args:
            context: Pipeline context containing the config.

        Returns:
            Context with ``result`` populated.
        """
        sim = Simulator(context.config)
        if context.config.regimes:
            context.result = sim.run_all_regimes()
        else:
            context.result = sim.run()
        return context


class ValidateStep:
    """Validate simulation results against configurable checks.

    Supported checks:
        - ``"finite_values"``: No NaN or Inf in outputs.
        - ``"no_nan"``: No NaN values in outputs.
        - ``"positive_rates"``: All rate values are non-negative.
        - ``"monotonic_survival"``: Survival probabilities are non-increasing.

    If no checks are specified, ``["finite_values", "no_nan"]`` is used.

    Args:
        checks: List of check names to run.

    Attributes:
        name: Step identifier (``"validate"``).
    """

    name: str = "validate"

    def __init__(self, checks: list[str] | None = None) -> None:
        self._checks = checks or ["finite_values", "no_nan"]

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Run all configured validation checks.

        Args:
            context: Pipeline context with a populated ``result``.

        Returns:
            Context with any validation errors appended.
        """
        if context.result is None:
            context.errors.append("ValidateStep: no result to validate")
            return context

        for check_name in self._checks:
            checker = _VALIDATORS.get(check_name)
            if checker is None:
                context.errors.append(f"ValidateStep: unknown check '{check_name}'")
                continue
            errors = checker(context.result)
            context.errors.extend(errors)

        context.metadata["validation_checks"] = list(self._checks)
        context.metadata["validation_passed"] = len(context.errors) == 0
        return context


class TimingStep:
    """Record timing information for the pipeline.

    Stores ``timing_start_ns`` when first executed and
    ``timing_elapsed_seconds`` on subsequent executions
    (or if the result already has elapsed metadata).

    Attributes:
        name: Step identifier (``"timing"``).
    """

    name: str = "timing"

    def __init__(self) -> None:
        self._start: float | None = None

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Record timing metadata.

        Args:
            context: Pipeline context.

        Returns:
            Context with timing metadata.
        """
        now = time.monotonic()
        if self._start is None:
            self._start = now
            context.metadata["timing_start"] = now
        else:
            context.metadata["timing_elapsed_seconds"] = now - self._start
        return context


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_finite(result: object) -> list[str]:
    """Check all output arrays for finite values."""
    from hyesg.engine.output import SimulationResult

    if not isinstance(result, SimulationResult):
        return ["finite_values: result is not a SimulationResult"]

    errors: list[str] = []
    for model_name, fields in result.outputs.items():
        for field_name, arr in fields.items():
            if not bool(jnp.all(jnp.isfinite(arr))):
                errors.append(
                    f"finite_values: non-finite values in {model_name}.{field_name}"
                )
    return errors


def _check_no_nan(result: object) -> list[str]:
    """Check all output arrays for NaN values."""
    from hyesg.engine.output import SimulationResult

    if not isinstance(result, SimulationResult):
        return ["no_nan: result is not a SimulationResult"]

    errors: list[str] = []
    for model_name, fields in result.outputs.items():
        for field_name, arr in fields.items():
            if bool(jnp.any(jnp.isnan(arr))):
                errors.append(
                    f"no_nan: NaN values found in {model_name}.{field_name}"
                )
    return errors


def _check_positive_rates(result: object) -> list[str]:
    """Check that rate-like outputs are non-negative."""
    from hyesg.engine.output import SimulationResult

    if not isinstance(result, SimulationResult):
        return ["positive_rates: result is not a SimulationResult"]

    rate_fields = {"short_rate", "rate", "forward_rate", "spot_rate"}
    errors: list[str] = []
    for model_name, fields in result.outputs.items():
        for field_name, arr in fields.items():
            if field_name in rate_fields and bool(jnp.any(arr < 0.0)):
                errors.append(
                    f"positive_rates: negative values in {model_name}.{field_name}"
                )
    return errors


def _check_monotonic_survival(result: object) -> list[str]:
    """Check that survival probabilities are non-increasing over time."""
    from hyesg.engine.output import SimulationResult

    if not isinstance(result, SimulationResult):
        return ["monotonic_survival: result is not a SimulationResult"]

    survival_fields = {"survival_probability", "survival"}
    errors: list[str] = []
    for model_name, fields in result.outputs.items():
        for field_name, arr in fields.items():
            if field_name in survival_fields:
                diffs = jnp.diff(arr, axis=-1)
                if bool(jnp.any(diffs > 1e-10)):
                    errors.append(
                        f"monotonic_survival: non-monotonic {model_name}.{field_name}"
                    )
    return errors


_VALIDATORS: dict[str, object] = {
    "finite_values": _check_finite,
    "no_nan": _check_no_nan,
    "positive_rates": _check_positive_rates,
    "monotonic_survival": _check_monotonic_survival,
}
