"""Batch simulation runner.

Provides ``BatchRunner`` for executing multiple simulation configs
sequentially or in parallel, and ``BatchResult`` for collecting outputs.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hyesg.engine.simulator import Simulator

if TYPE_CHECKING:
    from hyesg.config.models import SimulationConfig
    from hyesg.engine.output import SimulationResult
    from hyesg.orchestration.protocols import ProgressCallback

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from a batch simulation run.

    Attributes:
        results: List of simulation results, one per config.
        configs: The configs that were executed.
        timings: Elapsed seconds for each config.
        total_time: Total elapsed seconds for the entire batch.
    """

    results: list[SimulationResult]
    configs: list[SimulationConfig]
    timings: list[float]
    total_time: float


def _run_single(config: SimulationConfig) -> tuple[SimulationResult, float]:
    """Run a single simulation and return result + elapsed time.

    Args:
        config: Simulation configuration.

    Returns:
        Tuple of (SimulationResult, elapsed_seconds).
    """
    t_start = time.monotonic()
    sim = Simulator(config)
    result = sim.run_all_regimes() if config.regimes else sim.run()
    elapsed = time.monotonic() - t_start
    return result, elapsed


class BatchRunner:
    """Execute multiple simulation configs.

    Args:
        configs: List of simulation configs to run.
        on_progress: Optional callback ``(completed, total, config_name)``.
    """

    def __init__(
        self,
        configs: list[SimulationConfig],
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._configs = list(configs)
        self._on_progress = on_progress

    def _notify(self, completed: int, total: int, config_name: str) -> None:
        """Fire the progress callback if registered."""
        if self._on_progress is not None:
            self._on_progress(completed, total, config_name)

    def run_sequential(self) -> BatchResult:
        """Run all configs sequentially.

        Returns:
            BatchResult with results, timings, and total time.
        """
        if not self._configs:
            return BatchResult(results=[], configs=[], timings=[], total_time=0.0)

        total = len(self._configs)
        results: list[SimulationResult] = []
        timings: list[float] = []

        batch_start = time.monotonic()
        for idx, config in enumerate(self._configs):
            result, elapsed = _run_single(config)
            results.append(result)
            timings.append(elapsed)
            self._notify(idx + 1, total, config.name)
            logger.debug(
                "BatchRunner: completed %d/%d '%s' in %.2fs",
                idx + 1,
                total,
                config.name,
                elapsed,
            )

        total_time = time.monotonic() - batch_start
        return BatchResult(
            results=results,
            configs=list(self._configs),
            timings=timings,
            total_time=total_time,
        )

    def run_parallel(self, n_workers: int | None = None) -> BatchResult:
        """Run configs in parallel using threads.

        Uses ``ThreadPoolExecutor`` since JAX releases the GIL during
        computation.  Falls back to sequential if ``n_workers=1`` or
        there is only one config.

        Args:
            n_workers: Maximum worker threads.  Defaults to
                ``min(len(configs), 4)``.

        Returns:
            BatchResult with results, timings, and total time.
        """
        if not self._configs:
            return BatchResult(results=[], configs=[], timings=[], total_time=0.0)

        if n_workers == 1 or len(self._configs) == 1:
            return self.run_sequential()

        if n_workers is None:
            n_workers = min(len(self._configs), 4)

        total = len(self._configs)
        results: list[SimulationResult | None] = [None] * total
        timings: list[float] = [0.0] * total
        completed_count = 0

        batch_start = time.monotonic()

        def _worker(idx_config: tuple[int, SimulationConfig]) -> None:
            nonlocal completed_count
            idx, config = idx_config
            result, elapsed = _run_single(config)
            results[idx] = result
            timings[idx] = elapsed
            completed_count += 1
            self._notify(completed_count, total, config.name)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_worker, (idx, cfg))
                for idx, cfg in enumerate(self._configs)
            ]
            for future in futures:
                future.result()  # re-raise any exceptions

        total_time = time.monotonic() - batch_start

        # All slots should be filled; assert for safety
        final_results = [r for r in results if r is not None]
        if len(final_results) != total:
            raise RuntimeError(
                f"Expected {total} results but got {len(final_results)}"
            )

        return BatchResult(
            results=final_results,
            configs=list(self._configs),
            timings=timings,
            total_time=total_time,
        )
