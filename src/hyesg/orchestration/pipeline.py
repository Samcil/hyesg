"""Composable transformation pipeline.

Provides ``PipelineContext`` (the mutable state bag passed between steps)
and ``Pipeline`` (the ordered step runner with a fluent API).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyesg.config.models import SimulationConfig
    from hyesg.engine.output import SimulationResult
    from hyesg.orchestration.protocols import PipelineStep

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Mutable context passed through pipeline steps.

    Attributes:
        config: The simulation configuration.
        result: The simulation result (populated by SimulateStep).
        metadata: Arbitrary key-value metadata accumulated by steps.
        errors: Error messages collected during pipeline execution.
    """

    config: SimulationConfig
    result: SimulationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class Pipeline:
    """Composable transformation pipeline.

    Usage::

        pipeline = Pipeline("my_pipeline")
        pipeline.add_step(SimulateStep())
        pipeline.add_step(ValidateStep(checks=["finite_values"]))
        pipeline.add_step(TimingStep())

        context = pipeline.run(config)

    Args:
        name: Human-readable pipeline name.
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._steps: list[PipelineStep] = []

    @property
    def name(self) -> str:
        """Pipeline name."""
        return self._name

    @property
    def steps(self) -> list[PipelineStep]:
        """Ordered list of pipeline steps."""
        return list(self._steps)

    def add_step(self, step: PipelineStep) -> Pipeline:
        """Append a step to the pipeline (fluent API).

        Args:
            step: A ``PipelineStep`` implementation.

        Returns:
            Self, for method chaining.
        """
        self._steps.append(step)
        return self

    def run(self, config: SimulationConfig) -> PipelineContext:
        """Execute all steps in order.

        Args:
            config: The simulation configuration to process.

        Returns:
            Final pipeline context after all steps have executed.
        """
        context = PipelineContext(config=config)
        for step in self._steps:
            logger.debug("Pipeline '%s': executing step '%s'", self._name, step.name)
            try:
                context = step.execute(context)
            except Exception as exc:
                context.errors.append(f"Step '{step.name}' failed: {exc}")
                logger.exception(
                    "Pipeline '%s': step '%s' raised an exception",
                    self._name,
                    step.name,
                )
        return context
