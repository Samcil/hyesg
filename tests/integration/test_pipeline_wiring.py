"""Integration tests for F40 end-to-end pipeline wiring.

Tests the full flow: SimulationSetup → to_simulation_config →
Simulator → post-processing → output dict.
"""

from __future__ import annotations

from typing import Any

import pytest

from hyesg.calibration.protocols import CalibrationDataReader
from hyesg.config.economy import Economy, EconomyModelConfig
from hyesg.config.models import (
    CorrelationEntry,
    ModelConfig,
    PostProcessorConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.simulation_setup import SimulationSetupBuilder
from hyesg.core.registry import get_model, list_models
from hyesg.engine.post_processing.protocol import SimulationResults
from hyesg.engine.simulator import (
    Simulator,
    _build_post_processors,
    to_simulation_results,
)
from hyesg.orchestration.pipeline import Pipeline, PipelineContext
from hyesg.orchestration.steps import (
    CalibrationStep,
    PostProcessStep,
    SimulateStep,
    TimingStep,
    ValidateStep,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_economy(name: str, *, domestic: bool = False) -> Economy:
    """Create a minimal economy for integration testing."""
    return Economy(
        name=name,
        is_domestic=domestic,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp", label=f"{name.lower()}_nominal"
        ),
    )


def _make_minimal_config() -> SimulationConfig:
    """Build a minimal SimulationConfig for testing."""
    return SimulationConfig(
        name="test",
        time_grid=TimeGridConfig(start_year=0.0, end_year=1.0, frequency="annual"),
        models=[
            ModelConfig(
                type="cir2pp",
                name="test_nominal",
                params={},
            ),
        ],
    )


class _StubReader:
    """Stub CalibrationDataReader for testing."""

    def read(self) -> dict[str, Any]:
        return {"nominal_curve": [0.01, 0.02, 0.03]}


# ── Sub-task 1: SimulationSetup.to_simulation_config ──────────────


class TestSimulationSetupBridge:
    """Verify to_simulation_config() bridges the config systems."""

    def test_converts_economy_to_model_configs(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .seed(42)
            .add_regime("R1", trials=100)
            .add_economy(_make_economy("GBP", domestic=True))
            .build()
        )
        config = setup.to_simulation_config()
        assert isinstance(config, SimulationConfig)
        assert len(config.models) >= 1
        model_names = [m.name for m in config.models]
        assert "gbp_nominal" in model_names

    def test_preserves_regime_info(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .seed(42)
            .add_regime("Strong", trials=2500)
            .add_regime("Weak", trials=500)
            .add_economy(_make_economy("GBP", domestic=True))
            .build()
        )
        config = setup.to_simulation_config()
        assert len(config.regimes) == 2
        assert config.regimes[0].name == "Strong"
        assert config.regimes[0].n_trials == 2500
        assert config.regimes[1].name == "Weak"

    def test_time_grid_from_setup(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .seed(42)
            .time_grid(horizon=50, inverse_dt=12)
            .add_regime("R1", trials=100)
            .add_economy(_make_economy("GBP", domestic=True))
            .build()
        )
        config = setup.to_simulation_config()
        assert config.time_grid.end_year == 50.0
        assert config.time_grid.frequency == "monthly"

    def test_multi_economy_flattens_all_models(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .seed(42)
            .add_regime("R1", trials=50)
            .add_economy(_make_economy("GBP", domestic=True))
            .add_economy(_make_economy("USD"))
            .build()
        )
        config = setup.to_simulation_config()
        model_names = [m.name for m in config.models]
        assert "gbp_nominal" in model_names
        assert "usd_nominal" in model_names


# ── Sub-task 2: CalibrationStep ───────────────────────────────────


class TestCalibrationStep:
    """Verify CalibrationStep reads data and populates context."""

    def test_reads_market_data(self) -> None:
        reader = _StubReader()
        assert isinstance(reader, CalibrationDataReader)

        step = CalibrationStep(reader)
        assert step.name == "calibrate"

        context = PipelineContext(config=_make_minimal_config())
        result = step.execute(context)
        assert "calibration" in result.metadata
        assert result.metadata["calibration"]["status"] == "completed"
        assert result.metadata["calibration"]["market_data"]["nominal_curve"] == [
            0.01, 0.02, 0.03,
        ]

    def test_handles_reader_failure(self) -> None:
        class _FailingReader:
            def read(self) -> dict[str, Any]:
                msg = "CSV not found"
                raise FileNotFoundError(msg)

        step = CalibrationStep(_FailingReader())
        context = PipelineContext(config=_make_minimal_config())
        result = step.execute(context)
        assert result.metadata["calibration"]["status"] == "failed"
        assert len(result.errors) == 1
        assert "CSV not found" in result.errors[0]


# ── Sub-task 3: Post-processor wiring ─────────────────────────────


class TestPostProcessorWiring:
    """Verify post-processors are built and invoked."""

    def test_build_post_processors_empty(self) -> None:
        assert _build_post_processors([]) == []

    def test_to_simulation_results_adapter(self) -> None:
        import jax.numpy as jnp

        from hyesg.engine.output import SimulationResult

        result = SimulationResult(
            outputs={"m1": {"ShortRate": jnp.ones((10, 5))}},
            time_grid=jnp.arange(6, dtype=jnp.float64),
            metadata={"seed": 42},
        )
        adapted = to_simulation_results(result)
        assert isinstance(adapted, SimulationResults)
        assert adapted.n_trials == 10
        assert adapted.n_steps == 5
        assert "m1" in adapted.paths


# ── Sub-task 4: SalaryWedgeModel registration ────────────────────


class TestSalaryWedgeRegistration:
    """Verify SalaryWedgeModel is registered in the model registry."""

    def test_salary_wedge_is_registered(self) -> None:
        from hyesg.core.registry import _ensure_populated

        _ensure_populated()
        assert "salary_wedge" in list_models()

    def test_can_retrieve_salary_wedge(self) -> None:
        cls = get_model("salary_wedge")
        assert cls.__name__ == "SalaryWedgeModel"


# ── Sub-task 5: TODO annotations ──────────────────────────────────


class TestTodoAnnotations:
    """Verify orchestration modules have TODO comments."""

    def test_correlation_assembler_has_todo(self) -> None:
        import hyesg.orchestration.correlation_assembler as mod

        assert "F40 Integration Path" in (mod.__doc__ or "")

    def test_dependency_graph_has_todo(self) -> None:
        import hyesg.orchestration.dependency_graph as mod

        assert "F40 Integration Path" in (mod.__doc__ or "")

    def test_device_has_todo(self) -> None:
        import hyesg.orchestration.device as mod

        assert "F40 Integration Path" in (mod.__doc__ or "")


# ── Full pipeline integration ─────────────────────────────────────


class TestFullPipelineWiring:
    """End-to-end: setup → config → simulate → post-process."""

    def test_pipeline_with_all_steps(self) -> None:
        """Verify all steps chain correctly in a Pipeline."""
        from hyesg.config.params import CIRParams
        from hyesg.math.curves.primitives import ConstantCurve
        from hyesg.models import CIR

        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="pipeline_test",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=2.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="nominal", params={}),
            ],
        )

        reader = _StubReader()
        timer = TimingStep()

        pipeline = Pipeline("test_pipeline")
        pipeline.add_step(timer)
        pipeline.add_step(CalibrationStep(reader))
        pipeline.add_step(ValidateStep())
        pipeline.add_step(SimulateStep(models={"nominal": cir_model}))
        pipeline.add_step(timer)  # same instance records elapsed

        result = pipeline.run(config)

        # CalibrationStep populated metadata
        assert "calibration" in result.metadata
        assert result.metadata["calibration"]["status"] == "completed"
        # SimulateStep populated result
        assert result.result is not None
        # TimingStep recorded timing
        assert "timing_elapsed_seconds" in result.metadata
