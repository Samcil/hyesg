"""Pipeline protocol definitions for the orchestration module.

Defines the ``PipelineStep`` protocol and ``ProgressCallback`` type alias
used throughout the orchestration layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hyesg.orchestration.pipeline import PipelineContext

# (completed, total, current_config_name)
ProgressCallback = Callable[[int, int, str], None]


@runtime_checkable
class PipelineStep(Protocol):
    """A single step in a processing pipeline.

    Each step receives a ``PipelineContext``, performs its work,
    and returns the (possibly mutated) context.
    """

    @property
    def name(self) -> str:
        """Human-readable step name."""
        ...

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute this pipeline step.

        Args:
            context: Current pipeline context.

        Returns:
            Updated pipeline context.
        """
        ...
