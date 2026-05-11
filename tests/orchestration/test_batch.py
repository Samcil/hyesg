"""Tests for hyesg.orchestration.batch."""

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
from hyesg.orchestration.batch import BatchResult, BatchRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(name: str, seed: int = 42) -> SimulationConfig:
    """Build a small CIR config for batch testing."""
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
        regimes=[RegimeConfig(name="r1", n_trials=10, seed=seed)],
    )


# ---------------------------------------------------------------------------
# Sequential tests
# ---------------------------------------------------------------------------


class TestBatchRunnerSequential:
    """Tests for BatchRunner.run_sequential."""

    def test_sequential_three_configs(self) -> None:
        """Three configs should produce three results."""
        configs = [_make_config(f"cfg_{i}", seed=i + 1) for i in range(3)]
        runner = BatchRunner(configs)
        batch = runner.run_sequential()

        assert isinstance(batch, BatchResult)
        assert len(batch.results) == 3
        assert len(batch.timings) == 3
        assert len(batch.configs) == 3
        assert batch.total_time > 0.0

    def test_sequential_results_are_simulation_results(self) -> None:
        """Each result should be a SimulationResult."""
        configs = [_make_config("single")]
        batch = BatchRunner(configs).run_sequential()
        assert all(isinstance(r, SimulationResult) for r in batch.results)

    def test_sequential_timings_positive(self) -> None:
        """All individual timings should be positive."""
        configs = [_make_config(f"cfg_{i}") for i in range(2)]
        batch = BatchRunner(configs).run_sequential()
        assert all(t > 0.0 for t in batch.timings)

    def test_sequential_total_time_gte_sum_timings(self) -> None:
        """Total time should be at least the sum of individual timings."""
        configs = [_make_config(f"cfg_{i}") for i in range(2)]
        batch = BatchRunner(configs).run_sequential()
        assert batch.total_time >= sum(batch.timings) - 0.01

    def test_sequential_empty_configs(self) -> None:
        """Empty config list should return empty BatchResult."""
        batch = BatchRunner([]).run_sequential()
        assert batch.results == []
        assert batch.configs == []
        assert batch.timings == []
        assert batch.total_time == 0.0

    def test_sequential_single_config(self) -> None:
        """Single config should work correctly."""
        configs = [_make_config("solo")]
        batch = BatchRunner(configs).run_sequential()
        assert len(batch.results) == 1
        assert batch.results[0].n_trials == 10

    def test_sequential_configs_preserved(self) -> None:
        """Returned configs should match input configs."""
        configs = [_make_config(f"cfg_{i}") for i in range(3)]
        batch = BatchRunner(configs).run_sequential()
        for orig, returned in zip(configs, batch.configs, strict=True):
            assert orig.name == returned.name

    def test_sequential_different_seeds_different_results(self) -> None:
        """Different seeds should produce different results."""
        c1 = _make_config("a", seed=1)
        c2 = _make_config("b", seed=999)
        batch = BatchRunner([c1, c2]).run_sequential()
        r1 = batch.results[0].select("nominal", "short_rate")
        r2 = batch.results[1].select("nominal", "short_rate")
        import jax.numpy as jnp

        assert not jnp.allclose(r1, r2)


# ---------------------------------------------------------------------------
# Progress callback tests
# ---------------------------------------------------------------------------


class TestBatchRunnerProgress:
    """Tests for progress callback firing."""

    def test_progress_callback_fires(self) -> None:
        """Callback should fire once per config."""
        calls: list[tuple[int, int, str]] = []
        configs = [_make_config(f"cfg_{i}") for i in range(3)]
        def _cb(c: int, t: int, n: str) -> None:
            calls.append((c, t, n))

        runner = BatchRunner(configs, on_progress=_cb)
        runner.run_sequential()

        assert len(calls) == 3
        assert calls[-1][0] == 3  # completed == 3
        assert calls[-1][1] == 3  # total == 3

    def test_progress_callback_increments(self) -> None:
        """Completed count should increment from 1..N."""
        calls: list[int] = []
        configs = [_make_config(f"cfg_{i}") for i in range(3)]
        runner = BatchRunner(configs, on_progress=lambda c, t, n: calls.append(c))
        runner.run_sequential()
        assert calls == [1, 2, 3]

    def test_progress_callback_includes_config_name(self) -> None:
        """Callback should receive the correct config name."""
        names: list[str] = []
        configs = [_make_config(f"cfg_{i}") for i in range(2)]
        runner = BatchRunner(configs, on_progress=lambda c, t, n: names.append(n))
        runner.run_sequential()
        assert names == ["cfg_0", "cfg_1"]

    def test_no_progress_callback_ok(self) -> None:
        """Runner should work fine without a progress callback."""
        configs = [_make_config("solo")]
        batch = BatchRunner(configs).run_sequential()
        assert len(batch.results) == 1


# ---------------------------------------------------------------------------
# Parallel tests
# ---------------------------------------------------------------------------


class TestBatchRunnerParallel:
    """Tests for BatchRunner.run_parallel."""

    def test_parallel_three_configs(self) -> None:
        """Parallel run should produce correct number of results."""
        configs = [_make_config(f"cfg_{i}", seed=i + 10) for i in range(3)]
        batch = BatchRunner(configs).run_parallel(n_workers=2)

        assert len(batch.results) == 3
        assert len(batch.timings) == 3
        assert batch.total_time > 0.0

    def test_parallel_falls_back_to_sequential_for_one_worker(self) -> None:
        """n_workers=1 should fall back to sequential."""
        configs = [_make_config(f"cfg_{i}") for i in range(2)]
        batch = BatchRunner(configs).run_parallel(n_workers=1)
        assert len(batch.results) == 2

    def test_parallel_falls_back_for_single_config(self) -> None:
        """Single config should fall back to sequential."""
        configs = [_make_config("solo")]
        batch = BatchRunner(configs).run_parallel()
        assert len(batch.results) == 1

    def test_parallel_empty_configs(self) -> None:
        """Empty config list should return empty BatchResult."""
        batch = BatchRunner([]).run_parallel()
        assert batch.results == []
        assert batch.total_time == 0.0

    def test_parallel_results_are_simulation_results(self) -> None:
        """Each parallel result should be a SimulationResult."""
        configs = [_make_config(f"cfg_{i}") for i in range(2)]
        batch = BatchRunner(configs).run_parallel(n_workers=2)
        assert all(isinstance(r, SimulationResult) for r in batch.results)

    def test_parallel_default_workers(self) -> None:
        """Default n_workers should be min(len(configs), 4)."""
        configs = [_make_config(f"cfg_{i}") for i in range(3)]
        batch = BatchRunner(configs).run_parallel()
        assert len(batch.results) == 3

    def test_parallel_progress_fires(self) -> None:
        """Progress callback should fire in parallel mode."""
        calls: list[int] = []
        configs = [_make_config(f"cfg_{i}") for i in range(2)]
        runner = BatchRunner(configs, on_progress=lambda c, t, n: calls.append(c))
        runner.run_parallel(n_workers=2)
        assert len(calls) == 2
