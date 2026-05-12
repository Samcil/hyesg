"""Post-processing recipe — ordered chain of processors.

A ``PostProcessingRecipe`` holds an ordered list of ``PostProcessor``
instances and executes them sequentially.  ``CompositeProcessor`` allows
nesting recipes inside other recipes.
"""

from __future__ import annotations

from hyesg.engine.post_processing.protocol import (
    PostProcessor,
    ProcessedResults,
    SimulationResults,
)


class PostProcessingRecipe:
    """Ordered chain of processors with dependency validation.

    Args:
        processors: Initial list of processors to include.
    """

    def __init__(self, processors: list[PostProcessor] | None = None) -> None:
        self._processors: list[PostProcessor] = list(processors or [])

    def add(self, processor: PostProcessor) -> PostProcessingRecipe:
        """Append a processor to the chain.

        Returns:
            ``self`` for fluent chaining.
        """
        self._processors.append(processor)
        return self

    def execute(self, results: SimulationResults) -> ProcessedResults:
        """Run all processors in order and return ``ProcessedResults``.

        Args:
            results: Raw simulation results.

        Returns:
            ``ProcessedResults`` wrapping the raw input and all processed
            outputs.
        """
        current = results
        processor_names: list[str] = []
        for proc in self._processors:
            current = proc.process(current)
            processor_names.append(type(proc).__name__)

        return ProcessedResults(
            raw=results,
            processed=dict(current.paths),
            statistics=current.metadata.get("statistics", {}),
            metadata={
                **current.metadata,
                "processors_applied": processor_names,
            },
        )

    def validate(self) -> list[str]:
        """Check the recipe for potential issues.

        Returns:
            List of warning strings (empty if all OK).
        """
        warnings: list[str] = []
        if not self._processors:
            warnings.append("Recipe has no processors")
        seen: set[str] = set()
        for proc in self._processors:
            name = type(proc).__name__
            if name in seen:
                warnings.append(f"Duplicate processor type: {name}")
            seen.add(name)
        return warnings

    @property
    def processors(self) -> list[PostProcessor]:
        """Return a copy of the processor list."""
        return list(self._processors)

    def __len__(self) -> int:
        return len(self._processors)


class CompositeProcessor:
    """Nested processor chain for complex pipelines.

    A ``CompositeProcessor`` is itself a ``PostProcessor`` that runs an
    internal sequence of processors, making it composable with other
    recipes.

    Args:
        name: Human-readable label for the composite step.
        processors: Ordered list of inner processors.
    """

    def __init__(self, name: str, processors: list[PostProcessor]) -> None:
        self._name = name
        self._processors: list[PostProcessor] = list(processors)

    @property
    def name(self) -> str:
        return self._name

    def process(self, results: SimulationResults) -> SimulationResults:
        """Execute all inner processors sequentially."""
        current = results
        for proc in self._processors:
            current = proc.process(current)
        return current
