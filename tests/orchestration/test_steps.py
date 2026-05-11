"""Tests for hyesg.orchestration.steps."""

from __future__ import annotations

import time

import jax.numpy as jnp

import hyesg.models  # noqa: F401
from hyesg.config.models import (
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.params import CIRParams
from hyesg.engine.output import SimulationResult
from hyesg.engine.simulator import Simulator
from hyesg.orchestration.pipeline import PipelineContext
from hyesg.orchestration.steps import SimulateStep, TimingStep, ValidateStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(name: str = "test") -> SimulationConfig:
    """Build a small CIR config."""
    return SimulationConfig(
        name=name,
        time_grid=TimeGridConfig(
            start_year=0.0, end_year=5.0, frequency="annual"
        ),
        models=[
            ModelConfig(
                type="cir",
                name="nominal",
                params=CIRParams(
                    alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05
                ).model_dump(),
            ),
        ],
        regimes=[RegimeConfig(name="r1", n_trials=10, seed=42)],
    )


def _run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run a simulation and return the result."""
    sim = Simulator(config)
    return sim.run_all_regimes() if config.regimes else sim.run()


def _context_with_result(config: SimulationConfig | None = None) -> PipelineContext:
    """Create a PipelineContext with a pre-populated result."""
    cfg = config or _make_config()
    result = _run_simulation(cfg)
    return PipelineContext(config=cfg, result=result)


def _context_with_nan_result() -> PipelineContext:
    """Create a PipelineContext with NaN in the result."""
    ctx = _context_with_result()
    assert ctx.result is not None
    for model_name, fields in ctx.result.outputs.items():
        for field_name, arr in fields.items():
            ctx.result.outputs[model_name][field_name] = arr.at[0, 0].set(float("nan"))
            break
        break
    return ctx


def _context_with_inf_result() -> PipelineContext:
    """Create a PipelineContext with Inf in the result."""
    ctx = _context_with_result()
    assert ctx.result is not None
    for model_name, fields in ctx.result.outputs.items():
        for field_name, arr in fields.items():
            ctx.result.outputs[model_name][field_name] = arr.at[0, 0].set(float("inf"))
            break
        break
    return ctx


# ---------------------------------------------------------------------------
# SimulateStep
# ---------------------------------------------------------------------------


class TestSimulateStep:
    """Tests for SimulateStep."""

    def test_name(self) -> None:
        """Step name should be 'simulate'."""
        assert SimulateStep().name == "simulate"

    def test_produces_simulation_result(self) -> None:
        """SimulateStep should populate context.result."""
        ctx = PipelineContext(config=_make_config())
        result_ctx = SimulateStep().execute(ctx)
        assert result_ctx.result is not None
        assert isinstance(result_ctx.result, SimulationResult)

    def test_result_has_correct_trials(self) -> None:
        """Result should have the number of trials from the config."""
        ctx = PipelineContext(config=_make_config())
        result_ctx = SimulateStep().execute(ctx)
        assert result_ctx.result is not None
        assert result_ctx.result.n_trials == 10

    def test_result_has_correct_steps(self) -> None:
        """Result should have the number of steps from the config."""
        ctx = PipelineContext(config=_make_config())
        result_ctx = SimulateStep().execute(ctx)
        assert result_ctx.result is not None
        assert result_ctx.result.n_steps == 5

    def test_outputs_are_finite(self) -> None:
        """All outputs should be finite."""
        ctx = PipelineContext(config=_make_config())
        result_ctx = SimulateStep().execute(ctx)
        assert result_ctx.result is not None
        for fields in result_ctx.result.outputs.values():
            for arr in fields.values():
                assert jnp.all(jnp.isfinite(arr))

    def test_returns_context(self) -> None:
        """execute should return a PipelineContext."""
        ctx = PipelineContext(config=_make_config())
        result_ctx = SimulateStep().execute(ctx)
        assert isinstance(result_ctx, PipelineContext)


# ---------------------------------------------------------------------------
# ValidateStep
# ---------------------------------------------------------------------------


class TestValidateStep:
    """Tests for ValidateStep."""

    def test_name(self) -> None:
        """Step name should be 'validate'."""
        assert ValidateStep().name == "validate"

    def test_clean_data_no_errors(self) -> None:
        """Clean simulation data should produce no errors."""
        ctx = _context_with_result()
        result_ctx = ValidateStep().execute(ctx)
        assert result_ctx.errors == []

    def test_detects_nan(self) -> None:
        """Should detect NaN values in outputs."""
        ctx = _context_with_nan_result()
        result_ctx = ValidateStep(checks=["no_nan"]).execute(ctx)
        assert len(result_ctx.errors) > 0
        assert any("NaN" in e for e in result_ctx.errors)

    def test_detects_inf(self) -> None:
        """Should detect Inf values via finite_values check."""
        ctx = _context_with_inf_result()
        result_ctx = ValidateStep(checks=["finite_values"]).execute(ctx)
        assert len(result_ctx.errors) > 0
        assert any("non-finite" in e for e in result_ctx.errors)

    def test_positive_rates_on_cir(self) -> None:
        """CIR rates should pass the positive_rates check."""
        ctx = _context_with_result()
        result_ctx = ValidateStep(checks=["positive_rates"]).execute(ctx)
        assert result_ctx.errors == []

    def test_no_result_records_error(self) -> None:
        """Validating with no result should record an error."""
        ctx = PipelineContext(config=_make_config())
        result_ctx = ValidateStep().execute(ctx)
        assert len(result_ctx.errors) == 1
        assert "no result" in result_ctx.errors[0]

    def test_unknown_check_records_error(self) -> None:
        """Unknown check name should record an error."""
        ctx = _context_with_result()
        result_ctx = ValidateStep(checks=["nonexistent_check"]).execute(ctx)
        assert any("unknown check" in e for e in result_ctx.errors)

    def test_default_checks(self) -> None:
        """Default checks should include finite_values and no_nan."""
        step = ValidateStep()
        ctx = _context_with_result()
        result_ctx = step.execute(ctx)
        assert result_ctx.metadata.get("validation_checks") == [
            "finite_values",
            "no_nan",
        ]

    def test_validation_passed_metadata(self) -> None:
        """Metadata should indicate whether validation passed."""
        ctx = _context_with_result()
        result_ctx = ValidateStep().execute(ctx)
        assert result_ctx.metadata.get("validation_passed") is True

    def test_validation_failed_metadata(self) -> None:
        """Metadata should indicate validation failure on bad data."""
        ctx = _context_with_nan_result()
        result_ctx = ValidateStep().execute(ctx)
        assert result_ctx.metadata.get("validation_passed") is False

    def test_multiple_checks(self) -> None:
        """Multiple checks should all be evaluated."""
        ctx = _context_with_result()
        result_ctx = ValidateStep(
            checks=["finite_values", "no_nan", "positive_rates"]
        ).execute(ctx)
        assert result_ctx.errors == []
        assert len(result_ctx.metadata["validation_checks"]) == 3


# ---------------------------------------------------------------------------
# TimingStep
# ---------------------------------------------------------------------------


class TestTimingStep:
    """Tests for TimingStep."""

    def test_name(self) -> None:
        """Step name should be 'timing'."""
        assert TimingStep().name == "timing"

    def test_records_start(self) -> None:
        """First execution should record timing_start."""
        ctx = PipelineContext(config=_make_config())
        step = TimingStep()
        result_ctx = step.execute(ctx)
        assert "timing_start" in result_ctx.metadata

    def test_records_elapsed(self) -> None:
        """Second execution should record elapsed time."""
        ctx = PipelineContext(config=_make_config())
        step = TimingStep()
        ctx = step.execute(ctx)
        time.sleep(0.01)
        ctx = step.execute(ctx)
        assert "timing_elapsed_seconds" in ctx.metadata
        assert ctx.metadata["timing_elapsed_seconds"] > 0.0

    def test_returns_context(self) -> None:
        """execute should return a PipelineContext."""
        ctx = PipelineContext(config=_make_config())
        result_ctx = TimingStep().execute(ctx)
        assert isinstance(result_ctx, PipelineContext)
