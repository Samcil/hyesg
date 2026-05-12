"""Post-processing protocol definitions and result containers.

Defines the ``PostProcessor`` protocol that all post-processing steps
implement, plus ``SimulationResults`` and ``ProcessedResults`` containers.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class SimulationResults(BaseModel):
    """Container for raw simulation output paths.

    Attributes:
        paths: Mapping of model_name -> array of simulated paths.
        metadata: Arbitrary metadata from the simulation run.
        time_grid: Array of time points.
        n_trials: Number of Monte Carlo trials.
        n_steps: Number of simulation timesteps.
    """

    model_config = {"arbitrary_types_allowed": True}

    paths: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    time_grid: Any = None
    n_trials: int = 0
    n_steps: int = 0


class ProcessedResults(BaseModel):
    """Container for processed output.

    Attributes:
        raw: The original ``SimulationResults`` before processing.
        processed: Dictionary of processed arrays keyed by label.
        statistics: Dictionary of computed statistics.
        metadata: Processing metadata (timings, processor names, etc.).
    """

    model_config = {"arbitrary_types_allowed": True}

    raw: SimulationResults
    processed: dict[str, Any] = {}
    statistics: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for a single post-processing step.

    Every processor takes ``SimulationResults``, transforms them in place
    (adding to ``paths`` / ``metadata``), and returns the updated results.
    """

    def process(self, results: SimulationResults) -> SimulationResults: ...
