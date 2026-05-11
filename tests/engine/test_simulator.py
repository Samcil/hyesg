"""Tests for hyesg.engine.simulator — topological sort, time grid, and Simulator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.config.models import (
    CorrelationEntry,
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.params import CIRParams, GBMParams
from hyesg.core.types import CIRState, FXState, OutputSpec, ShockConfig
from hyesg.engine.output import SimulationResult
from hyesg.engine.simulator import Simulator, build_time_grid, topological_sort

# Ensure models are registered by importing them
import hyesg.models  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_ns(
    name: str, dependencies: list[str] | None = None
) -> SimpleNamespace:
    """Create a SimpleNamespace mimicking ModelConfig for topological_sort."""
    return SimpleNamespace(
        name=name,
        dependencies=dependencies or [],
    )


class DummyModel:
    """Minimal model for testing the simulator without real financial models."""

    def __init__(
        self,
        name: str = "dummy",
        n_shocks_val: int = 1,
        deps: list[str] | None = None,
    ) -> None:
        self._name = name
        self._n_shocks = n_shocks_val
        self._deps = deps or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_shocks(self) -> int:
        return self._n_shocks

    @property
    def shock_config(self) -> ShockConfig:
        return ShockConfig(
            n_shocks=self._n_shocks,
            distribution="normal",
            correlate=True,
            names=tuple(
                f"{self._name}_z" if self._n_shocks == 1 else f"{self._name}_z{i}"
                for i in range(self._n_shocks)
            ),
        )

    def init_state(self, params: Any = None, market: Any = None) -> CIRState:
        return CIRState(
            x=jnp.array(0.05, dtype=jnp.float64),
            state_var=jnp.array(0.05, dtype=jnp.float64),
            short_rate=jnp.array(0.05, dtype=jnp.float64),
        )

    def step(
        self,
        state: CIRState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[CIRState, dict[str, Any]]:
        dz = shocks[0]
        x_new = state.x + 0.1 * (0.05 - state.x) * dt + 0.01 * dz * jnp.sqrt(dt)
        sv = jnp.maximum(x_new, 0.0)
        new_state = CIRState(x=x_new, state_var=sv, short_rate=sv)
        return new_state, {"short_rate": sv}


class DummyEquity:
    """Minimal equity-like model that depends on a rate model."""

    def __init__(self, name: str = "equity_test") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_shocks(self) -> int:
        return 1

    @property
    def shock_config(self) -> ShockConfig:
        return ShockConfig(
            n_shocks=1,
            distribution="normal",
            correlate=True,
            names=(f"{self._name}_z",),
        )

    def init_state(self, params: Any = None, market: Any = None) -> FXState:
        return FXState(
            log_level=jnp.log(jnp.array(100.0, dtype=jnp.float64)),
            level=jnp.array(100.0, dtype=jnp.float64),
        )

    def step(
        self,
        state: FXState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[FXState, dict[str, Any]]:
        dz = shocks[0]
        r = deps.get("short_rate", jnp.array(0.0, dtype=jnp.float64))
        sigma = 0.15
        log_new = state.log_level + (r - 0.5 * sigma**2) * dt + sigma * dz * jnp.sqrt(dt)
        level = jnp.exp(log_new)
        new_state = FXState(log_level=log_new, level=level)
        return new_state, {"level": level, "log_return": log_new - state.log_level}


# ---------------------------------------------------------------------------
# Topological sort tests
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Tests for topological_sort."""

    def test_linear_chain(self):
        """A -> B -> C gives [A, B, C]."""
        models = {
            "C": _make_model_ns("C", ["B"]),
            "B": _make_model_ns("B", ["A"]),
            "A": _make_model_ns("A", []),
        }
        order = topological_sort(models)
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_diamond(self):
        """Diamond: A -> B, A -> C, B -> D, C -> D."""
        models = {
            "D": _make_model_ns("D", ["B", "C"]),
            "C": _make_model_ns("C", ["A"]),
            "B": _make_model_ns("B", ["A"]),
            "A": _make_model_ns("A", []),
        }
        order = topological_sort(models)
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_independent(self):
        """Independent models: any valid ordering."""
        models = {
            "X": _make_model_ns("X", []),
            "Y": _make_model_ns("Y", []),
            "Z": _make_model_ns("Z", []),
        }
        order = topological_sort(models)
        assert set(order) == {"X", "Y", "Z"}
        assert len(order) == 3

    def test_single_model(self):
        """Single model with no deps."""
        models = {"A": _make_model_ns("A", [])}
        assert topological_sort(models) == ["A"]

    def test_cyclic_raises(self):
        """Cyclic dependency raises ValueError."""
        models = {
            "A": _make_model_ns("A", ["B"]),
            "B": _make_model_ns("B", ["A"]),
        }
        with pytest.raises(ValueError, match="Cyclic dependencies"):
            topological_sort(models)

    def test_three_way_cycle(self):
        """Three-way cycle raises ValueError."""
        models = {
            "A": _make_model_ns("A", ["C"]),
            "B": _make_model_ns("B", ["A"]),
            "C": _make_model_ns("C", ["B"]),
        }
        with pytest.raises(ValueError, match="Cyclic dependencies"):
            topological_sort(models)

    def test_missing_dependency_raises(self):
        """Dependency on non-existent model raises ValueError."""
        models = {
            "A": _make_model_ns("A", ["missing"]),
        }
        with pytest.raises(ValueError, match="'missing' which is not defined"):
            topological_sort(models)

    def test_empty_models(self):
        """Empty models dict returns empty list."""
        assert topological_sort({}) == []


# ---------------------------------------------------------------------------
# build_time_grid tests
# ---------------------------------------------------------------------------


class TestBuildTimeGrid:
    """Tests for build_time_grid."""

    def test_monthly(self):
        """Monthly grid for 1 year = 12 steps."""
        config = TimeGridConfig(
            start_year=0.0, end_year=1.0, frequency="monthly"
        )
        times, dts = build_time_grid(config)
        assert times.shape[0] == 13  # 12 steps + 1
        assert dts.shape[0] == 12
        assert jnp.allclose(times[0], 0.0)
        assert jnp.allclose(times[-1], 1.0)

    def test_quarterly(self):
        """Quarterly grid for 1 year = 4 steps."""
        config = TimeGridConfig(
            start_year=0.0, end_year=1.0, frequency="quarterly"
        )
        times, dts = build_time_grid(config)
        assert times.shape[0] == 5
        assert dts.shape[0] == 4

    def test_annual(self):
        """Annual grid for 5 years = 5 steps."""
        config = TimeGridConfig(
            start_year=0.0, end_year=5.0, frequency="annual"
        )
        times, dts = build_time_grid(config)
        assert times.shape[0] == 6
        assert dts.shape[0] == 5
        assert jnp.allclose(dts, 1.0)

    def test_semi_annual(self):
        """Semi-annual grid for 2 years = 4 steps."""
        config = TimeGridConfig(
            start_year=0.0, end_year=2.0, frequency="semi_annual"
        )
        times, dts = build_time_grid(config)
        assert times.shape[0] == 5
        assert dts.shape[0] == 4
        assert jnp.allclose(dts, 0.5)

    def test_custom_times(self):
        """Custom time grid."""
        config = TimeGridConfig(custom_times=[0.0, 0.5, 1.0, 2.0, 5.0])
        times, dts = build_time_grid(config)
        assert times.shape[0] == 5
        assert dts.shape[0] == 4
        assert jnp.allclose(dts[0], 0.5)
        assert jnp.allclose(dts[-1], 3.0)

    def test_dts_positive(self):
        """All dts are positive."""
        config = TimeGridConfig(start_year=0.0, end_year=10.0, frequency="annual")
        _, dts = build_time_grid(config)
        assert jnp.all(dts > 0)


# ---------------------------------------------------------------------------
# Simulator with dummy models
# ---------------------------------------------------------------------------


def _make_single_model_config() -> SimulationConfig:
    """Config with a single dummy CIR-like model."""
    return SimulationConfig(
        name="test_single",
        time_grid=TimeGridConfig(start_year=0.0, end_year=1.0, frequency="annual"),
        models=[
            ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
        ],
        regimes=[
            RegimeConfig(name="r1", n_trials=4, seed=42),
        ],
    )


def _make_two_model_config() -> SimulationConfig:
    """Config with CIR + dependent equity model."""
    return SimulationConfig(
        name="test_two",
        time_grid=TimeGridConfig(start_year=0.0, end_year=1.0, frequency="annual"),
        models=[
            ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ModelConfig(
                type="equity", name="stocks", params={}, dependencies=["rates"]
            ),
        ],
        correlations=[
            CorrelationEntry(shock_a="rates_z", shock_b="stocks_z", value=0.3),
        ],
        regimes=[
            RegimeConfig(name="r1", n_trials=4, seed=42),
        ],
    )


class TestSimulatorSingleModel:
    """Tests for Simulator with a single dummy model."""

    def test_create_simulator(self):
        """Simulator creates without error."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        assert sim.model_order == ["rates"]

    def test_total_shocks(self):
        """Total shocks = sum of model shocks."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        assert sim.total_shocks == 1

    def test_shock_slices(self):
        """Shock slices correctly assigned."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        assert sim.shock_slices == {"rates": (0, 1)}

    def test_n_steps(self):
        """n_steps matches time grid."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        assert sim.n_steps == 1  # annual for 1 year

    def test_run_produces_result(self):
        """run() produces a SimulationResult."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        assert isinstance(result, SimulationResult)

    def test_output_shapes(self):
        """Output arrays have shape (n_trials, n_steps)."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        sr = result.select("rates", "short_rate")
        assert sr.shape == (4, 1)

    def test_output_finite(self):
        """Outputs are all finite."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        sr = result.select("rates", "short_rate")
        assert jnp.all(jnp.isfinite(sr))

    def test_deterministic(self):
        """Same seed produces identical outputs."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        r1 = sim.run(seed=123)
        r2 = sim.run(seed=123)
        assert jnp.allclose(
            r1.select("rates", "short_rate"),
            r2.select("rates", "short_rate"),
        )

    def test_different_seeds_differ(self):
        """Different seeds produce different outputs."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        r1 = sim.run(seed=1)
        r2 = sim.run(seed=2)
        sr1 = r1.select("rates", "short_rate")
        sr2 = r2.select("rates", "short_rate")
        assert not jnp.allclose(sr1, sr2)

    def test_metadata_present(self):
        """Result metadata contains expected keys."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        assert "seed" in result.metadata
        assert "n_trials" in result.metadata
        assert result.metadata["seed"] == 42


# ---------------------------------------------------------------------------
# Simulator with two models (dependency passing)
# ---------------------------------------------------------------------------


class TestSimulatorTwoModels:
    """Tests for Simulator with CIR + Equity (dependency passing)."""

    def test_model_order_respects_deps(self):
        """rates before stocks in topological order."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        order = sim.model_order
        assert order.index("rates") < order.index("stocks")

    def test_total_shocks_two_models(self):
        """Total shocks = 2 (1 per model)."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        assert sim.total_shocks == 2

    def test_correlation_matrix_shape(self):
        """Correlation matrix is 2x2."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        assert sim.correlation_matrix.shape == (2, 2)

    def test_correlation_values(self):
        """Correlation matrix has correct off-diagonal values."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        corr = sim.correlation_matrix
        assert jnp.allclose(corr[0, 0], 1.0)
        assert jnp.allclose(corr[1, 1], 1.0)
        assert jnp.allclose(corr[0, 1], 0.3)
        assert jnp.allclose(corr[1, 0], 0.3)

    def test_run_two_models(self):
        """Two-model simulation runs and produces outputs for both."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        result = sim.run(seed=42)
        assert "rates" in result.model_names
        assert "stocks" in result.model_names

    def test_equity_output_shape(self):
        """Equity outputs have correct shape."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        result = sim.run(seed=42)
        level = result.select("stocks", "level")
        assert level.shape == (4, 1)

    def test_equity_uses_rate_dep(self):
        """Equity model receives rate dependency (outputs are not constant)."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        result = sim.run(seed=42)
        level = result.select("stocks", "level")
        # Equity should have evolved from initial 100 — not exactly 100
        assert not jnp.allclose(level, 100.0)

    def test_two_model_deterministic(self):
        """Two-model simulation is deterministic with same seed."""
        config = _make_two_model_config()
        dummy_rates = DummyModel(name="rates")
        dummy_equity = DummyEquity(name="stocks")
        sim = Simulator(config, models={"rates": dummy_rates, "stocks": dummy_equity})
        r1 = sim.run(seed=99)
        r2 = sim.run(seed=99)
        assert jnp.allclose(
            r1.select("rates", "short_rate"),
            r2.select("rates", "short_rate"),
        )
        assert jnp.allclose(
            r1.select("stocks", "level"),
            r2.select("stocks", "level"),
        )


# ---------------------------------------------------------------------------
# Longer simulation (multiple steps)
# ---------------------------------------------------------------------------


class TestMultiStep:
    """Tests for multi-step simulations."""

    def test_multi_step_shapes(self):
        """5-year annual = 5 steps."""
        config = SimulationConfig(
            name="multi",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=5.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=3, seed=42)],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        sr = result.select("rates", "short_rate")
        assert sr.shape == (3, 5)

    def test_time_grid_in_result(self):
        """Time grid is included in result."""
        config = SimulationConfig(
            name="multi",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=5.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=2, seed=42)],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        assert result.time_grid.shape == (6,)

    def test_outputs_vary_across_steps(self):
        """Outputs change over time (not constant)."""
        config = SimulationConfig(
            name="multi",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=10.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=2, seed=42)],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        sr = result.select("rates", "short_rate")
        # Not all timesteps should be identical (stochastic)
        assert sr.shape[1] == 10
        # Check that values vary across timesteps for trial 0
        assert not jnp.allclose(sr[0, 0], sr[0, -1], atol=1e-10)


# ---------------------------------------------------------------------------
# vmap batching
# ---------------------------------------------------------------------------


class TestVmapBatching:
    """Tests for vmap batching across trials."""

    def test_multiple_trials_shape(self):
        """Multiple trials produce correct output shape."""
        config = SimulationConfig(
            name="batch",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=2.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=10, seed=42)],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        sr = result.select("rates", "short_rate")
        assert sr.shape == (10, 2)

    def test_trials_differ(self):
        """Different trials produce different paths."""
        config = SimulationConfig(
            name="batch",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=5.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=5, seed=42)],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run(seed=42)
        sr = result.select("rates", "short_rate")
        # Different trials should give different values
        assert not jnp.allclose(sr[0], sr[1])


# ---------------------------------------------------------------------------
# Static config not in carry
# ---------------------------------------------------------------------------


class TestCarryStructure:
    """Tests that static config is NOT in the scan carry."""

    def test_carry_keys(self):
        """Carry dict has only 'states' and 'rng_key'."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})

        model_deps = {"rates": []}
        step_fn = sim._make_step_fn(
            models={"rates": dummy},
            model_order=["rates"],
            shock_slices={"rates": (0, 1)},
            model_deps=model_deps,
            cholesky_L=jnp.eye(1, dtype=jnp.float64),
            total_shocks=1,
        )

        init_state = dummy.init_state()
        carry = {
            "states": {"rates": init_state},
            "rng_key": jax.random.PRNGKey(0),
        }
        t_dt = (jnp.array(0.0), jnp.array(1.0))
        new_carry, outputs = step_fn(carry, t_dt)

        assert set(new_carry.keys()) == {"states", "rng_key"}
        assert "rates" in new_carry["states"]


# ---------------------------------------------------------------------------
# Simulator properties
# ---------------------------------------------------------------------------


class TestSimulatorProperties:
    """Tests for Simulator property accessors."""

    def test_config_property(self):
        """config property returns the SimulationConfig."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        assert sim.config is config

    def test_cholesky_L_shape(self):
        """Cholesky factor has correct shape."""
        config = _make_single_model_config()
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        assert sim.cholesky_L.shape == (1, 1)


# ---------------------------------------------------------------------------
# run_all_regimes
# ---------------------------------------------------------------------------


class TestRunAllRegimes:
    """Tests for run_all_regimes."""

    def test_single_regime(self):
        """Single regime behaves like run()."""
        config = SimulationConfig(
            name="one_regime",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=1.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=3, seed=42)],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run_all_regimes()
        assert result.n_trials == 3

    def test_two_regimes(self):
        """Two regimes concatenate trials."""
        config = SimulationConfig(
            name="two_regimes",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=1.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
            regimes=[
                RegimeConfig(name="r1", n_trials=3, seed=42),
                RegimeConfig(name="r2", n_trials=5, seed=99),
            ],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run_all_regimes()
        assert result.n_trials == 8

    def test_no_regimes(self):
        """No regimes uses defaults."""
        config = SimulationConfig(
            name="no_regimes",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=1.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
            ],
        )
        dummy = DummyModel(name="rates")
        sim = Simulator(config, models={"rates": dummy})
        result = sim.run_all_regimes()
        assert isinstance(result, SimulationResult)


# ---------------------------------------------------------------------------
# Integration with real models (CIR)
# ---------------------------------------------------------------------------


class TestRealCIRModel:
    """Integration tests using the real CIR model from the registry."""

    def _make_cir_config(self, n_trials: int = 4, n_years: int = 1) -> SimulationConfig:
        return SimulationConfig(
            name="cir_test",
            time_grid=TimeGridConfig(
                start_year=0.0,
                end_year=float(n_years),
                frequency="annual",
            ),
            models=[
                ModelConfig(type="cir", name="nominal", params={}, dependencies=[]),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=n_trials, seed=42)],
        )

    def test_real_cir_runs(self):
        """Real CIR model runs through simulator."""
        config = self._make_cir_config()
        cir_params = CIRParams(alpha=0.1, mu=0.05, sigma=0.02, initial_value=0.03)
        from hyesg.models.short_rates.cir import CIR

        cir = CIR(params=cir_params, name="nominal")
        sim = Simulator(config, models={"nominal": cir})
        result = sim.run(seed=42)
        sr = result.select("nominal", "short_rate")
        assert sr.shape == (4, 1)
        assert jnp.all(jnp.isfinite(sr))

    def test_real_cir_multi_step(self):
        """Real CIR model with 5 years."""
        config = self._make_cir_config(n_trials=3, n_years=5)
        cir_params = CIRParams(alpha=0.1, mu=0.05, sigma=0.02, initial_value=0.03)
        from hyesg.models.short_rates.cir import CIR

        cir = CIR(params=cir_params, name="nominal")
        sim = Simulator(config, models={"nominal": cir})
        result = sim.run(seed=42)
        sr = result.select("nominal", "short_rate")
        assert sr.shape == (3, 5)
        # CIR should stay non-negative (floored)
        assert jnp.all(sr >= 0.0)

    def test_real_cir_deterministic(self):
        """Real CIR gives identical results with same seed."""
        config = self._make_cir_config(n_trials=2, n_years=3)
        cir_params = CIRParams(alpha=0.1, mu=0.05, sigma=0.02, initial_value=0.03)
        from hyesg.models.short_rates.cir import CIR

        cir = CIR(params=cir_params, name="nominal")
        sim = Simulator(config, models={"nominal": cir})
        r1 = sim.run(seed=77)
        r2 = sim.run(seed=77)
        assert jnp.allclose(
            r1.select("nominal", "short_rate"),
            r2.select("nominal", "short_rate"),
        )


# ---------------------------------------------------------------------------
# Integration with CIR + Equity (real models)
# ---------------------------------------------------------------------------


class TestRealCIREquity:
    """Integration: real CIR + Equity with dependency."""

    def test_cir_equity_runs(self):
        """CIR + Equity simulation produces outputs."""
        config = SimulationConfig(
            name="cir_eq",
            time_grid=TimeGridConfig(
                start_year=0.0, end_year=2.0, frequency="annual"
            ),
            models=[
                ModelConfig(type="cir", name="rates", params={}, dependencies=[]),
                ModelConfig(
                    type="equity",
                    name="stocks",
                    params={},
                    dependencies=["rates"],
                ),
            ],
            correlations=[
                CorrelationEntry(shock_a="rates_z", shock_b="stocks_z", value=0.2),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=3, seed=42)],
        )
        cir_params = CIRParams(alpha=0.1, mu=0.05, sigma=0.02, initial_value=0.03)
        gbm_params = GBMParams(sigma=0.15, initial_value=100.0)

        from hyesg.models.equity.equity import Equity
        from hyesg.models.short_rates.cir import CIR

        cir = CIR(params=cir_params, name="rates")
        eq = Equity(params=gbm_params, name="stocks")
        sim = Simulator(config, models={"rates": cir, "stocks": eq})
        result = sim.run(seed=42)

        assert "rates" in result.model_names
        assert "stocks" in result.model_names
        sr = result.select("rates", "short_rate")
        level = result.select("stocks", "level")
        assert sr.shape == (3, 2)
        assert level.shape == (3, 2)
        assert jnp.all(jnp.isfinite(sr))
        assert jnp.all(jnp.isfinite(level))
        assert jnp.all(level > 0.0)  # equity prices positive
