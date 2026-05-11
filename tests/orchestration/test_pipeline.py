"""Tests for hyesg.orchestration.pipeline."""

from __future__ import annotations

import hyesg.models  # noqa: F401
from hyesg.config.models import (
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.params import CIRParams
from hyesg.engine.output import SimulationResult
from hyesg.orchestration.pipeline import Pipeline, PipelineContext
from hyesg.orchestration.protocols import PipelineStep
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


class _RecordingStep:
    """Test helper that records when it executes."""

    def __init__(self, step_name: str, log: list[str]) -> None:
        self._name = step_name
        self._log = log

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context: PipelineContext) -> PipelineContext:
        self._log.append(self._name)
        context.metadata[self._name] = True
        return context


class _FailingStep:
    """Test helper that always raises."""

    name: str = "failing"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise RuntimeError("step failed on purpose")


class _NanInjectionStep:
    """Injects NaN into results for validation testing."""

    name: str = "nan_injector"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.result is not None:
            for model_name, fields in context.result.outputs.items():
                for field_name, arr in fields.items():
                    context.result.outputs[model_name][field_name] = arr.at[0, 0].set(
                        float("nan")
                    )
                    break
                break
        return context


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    """Tests for Pipeline setup and metadata."""

    def test_pipeline_name(self) -> None:
        """Pipeline should store its name."""
        pipe = Pipeline("my_pipe")
        assert pipe.name == "my_pipe"

    def test_pipeline_default_name(self) -> None:
        """Default name should be 'default'."""
        pipe = Pipeline()
        assert pipe.name == "default"

    def test_pipeline_starts_empty(self) -> None:
        """New pipeline should have no steps."""
        pipe = Pipeline()
        assert pipe.steps == []

    def test_add_step_returns_self(self) -> None:
        """add_step should return the pipeline for chaining."""
        pipe = Pipeline()
        returned = pipe.add_step(SimulateStep())
        assert returned is pipe

    def test_fluent_chaining(self) -> None:
        """Multiple add_step calls can be chained."""
        pipe = (
            Pipeline("chained")
            .add_step(SimulateStep())
            .add_step(ValidateStep())
            .add_step(TimingStep())
        )
        assert len(pipe.steps) == 3

    def test_steps_are_ordered(self) -> None:
        """Steps should maintain insertion order."""
        pipe = Pipeline()
        pipe.add_step(SimulateStep())
        pipe.add_step(ValidateStep())
        assert pipe.steps[0].name == "simulate"
        assert pipe.steps[1].name == "validate"


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


class TestPipelineExecution:
    """Tests for Pipeline.run."""

    def test_empty_pipeline_returns_context(self) -> None:
        """Empty pipeline should return context with config and no result."""
        config = _make_config()
        context = Pipeline().run(config)
        assert context.config is config
        assert context.result is None
        assert context.errors == []

    def test_simulate_step_populates_result(self) -> None:
        """Pipeline with SimulateStep should populate result."""
        config = _make_config()
        context = Pipeline("sim").add_step(SimulateStep()).run(config)
        assert context.result is not None
        assert isinstance(context.result, SimulationResult)

    def test_steps_execute_in_order(self) -> None:
        """Steps should execute in the order they were added."""
        log: list[str] = []
        pipe = Pipeline("ordered")
        pipe.add_step(_RecordingStep("first", log))
        pipe.add_step(_RecordingStep("second", log))
        pipe.add_step(_RecordingStep("third", log))

        pipe.run(_make_config())
        assert log == ["first", "second", "third"]

    def test_context_flows_between_steps(self) -> None:
        """Metadata set by one step should be visible to subsequent steps."""
        log: list[str] = []
        pipe = Pipeline()
        pipe.add_step(_RecordingStep("a", log))
        pipe.add_step(_RecordingStep("b", log))

        context = pipe.run(_make_config())
        assert context.metadata["a"] is True
        assert context.metadata["b"] is True

    def test_failing_step_records_error(self) -> None:
        """A failing step should record the error and continue."""
        pipe = Pipeline()
        pipe.add_step(_FailingStep())
        context = pipe.run(_make_config())
        assert len(context.errors) == 1
        assert "failing" in context.errors[0]

    def test_failing_step_does_not_abort_pipeline(self) -> None:
        """Steps after a failure should still execute."""
        log: list[str] = []
        pipe = Pipeline()
        pipe.add_step(_RecordingStep("before", log))
        pipe.add_step(_FailingStep())
        pipe.add_step(_RecordingStep("after", log))

        context = pipe.run(_make_config())
        assert "before" in log
        assert "after" in log
        assert len(context.errors) == 1

    def test_simulate_then_validate_passes(self) -> None:
        """SimulateStep + ValidateStep on clean data should have no errors."""
        config = _make_config()
        context = (
            Pipeline("sim_val")
            .add_step(SimulateStep())
            .add_step(ValidateStep(checks=["finite_values", "no_nan"]))
            .run(config)
        )
        assert context.result is not None
        assert context.errors == []

    def test_validate_catches_nan(self) -> None:
        """ValidateStep should detect NaN values injected by a prior step."""
        config = _make_config()
        context = (
            Pipeline("nan_test")
            .add_step(SimulateStep())
            .add_step(_NanInjectionStep())
            .add_step(ValidateStep(checks=["no_nan", "finite_values"]))
            .run(config)
        )
        assert len(context.errors) > 0
        assert any("NaN" in e or "non-finite" in e for e in context.errors)


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------


class TestPipelineContext:
    """Tests for the PipelineContext dataclass."""

    def test_context_defaults(self) -> None:
        """Context should have sensible defaults."""
        config = _make_config()
        ctx = PipelineContext(config=config)
        assert ctx.result is None
        assert ctx.metadata == {}
        assert ctx.errors == []

    def test_context_stores_config(self) -> None:
        """Context should store the config."""
        config = _make_config("stored")
        ctx = PipelineContext(config=config)
        assert ctx.config.name == "stored"

    def test_context_metadata_mutable(self) -> None:
        """Metadata dict should be mutable."""
        ctx = PipelineContext(config=_make_config())
        ctx.metadata["key"] = "value"
        assert ctx.metadata["key"] == "value"

    def test_context_errors_appendable(self) -> None:
        """Errors list should be appendable."""
        ctx = PipelineContext(config=_make_config())
        ctx.errors.append("oops")
        assert ctx.errors == ["oops"]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestPipelineStepProtocol:
    """Tests for PipelineStep protocol conformance."""

    def test_simulate_step_is_pipeline_step(self) -> None:
        """SimulateStep should satisfy PipelineStep protocol."""
        assert isinstance(SimulateStep(), PipelineStep)

    def test_validate_step_is_pipeline_step(self) -> None:
        """ValidateStep should satisfy PipelineStep protocol."""
        assert isinstance(ValidateStep(), PipelineStep)

    def test_timing_step_is_pipeline_step(self) -> None:
        """TimingStep should satisfy PipelineStep protocol."""
        assert isinstance(TimingStep(), PipelineStep)

    def test_recording_step_is_pipeline_step(self) -> None:
        """Custom _RecordingStep should satisfy PipelineStep protocol."""
        assert isinstance(_RecordingStep("test", []), PipelineStep)
