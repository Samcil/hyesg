"""Tests for the Equity (GBM) model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import GBMParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.core.types import FXState
from hyesg.models.equity.equity import Equity

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the equity model for each test."""
    clear_registry()

    # Re-import to re-register
    import importlib

    import hyesg.models.equity.equity as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def params() -> GBMParams:
    return GBMParams(sigma=0.2, initial_value=100.0)


@pytest.fixture
def model(params: GBMParams) -> Equity:
    return Equity(params=params, name="equity")


class TestEquityInit:
    def test_initial_state(self, model: Equity) -> None:
        """s0=100 → FXState(log_level=ln(100), level=100)."""
        state = model.init_state()
        assert isinstance(state, FXState)
        assert jnp.isclose(state.level, 100.0, atol=1e-12)
        assert jnp.isclose(state.log_level, jnp.log(100.0), atol=1e-12)

    def test_name(self, model: Equity) -> None:
        assert model.name == "equity"

    def test_custom_name(self, params: GBMParams) -> None:
        m = Equity(params=params, name="property")
        assert m.name == "property"


class TestEquityShockConfig:
    def test_n_shocks(self, model: Equity) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: Equity) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert cfg.names == ("equity_z",)


class TestEquityStep:
    def test_single_step_known_shock(self, model: Equity) -> None:
        """Known shock → verify log-normal step."""
        state = model.init_state()
        dt = 1.0 / 12.0
        dz = jnp.array(0.5, dtype=jnp.float64)
        shocks = jnp.array([dz])
        deps: dict = {"rates": {"short_rate": jnp.array(0.05, dtype=jnp.float64)}}

        new_state, outputs = model.step(state, 0.0, dt, shocks, deps)

        # Manual: ln(100) + (0.05 - 0 - 0.5*0.04) * (1/12) + 0.2 * 0.5 * sqrt(1/12)
        sigma = 0.2
        expected_log = (
            jnp.log(100.0) + (0.05 - 0.5 * sigma**2) * dt + sigma * 0.5 * jnp.sqrt(dt)
        )
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)
        assert jnp.isclose(new_state.level, jnp.exp(expected_log), atol=1e-10)
        assert "level" in outputs
        assert "log_return" in outputs

    def test_zero_vol_deterministic(self) -> None:
        """σ=0 → deterministic growth at rate r-q."""
        params = GBMParams(sigma=0.0, initial_value=100.0)
        model = Equity(params=params, dividend_yield=0.02)
        state = model.init_state()
        dt = 0.25
        shocks = jnp.array([1.5])  # should not matter with σ=0
        deps: dict = {"rates": {"short_rate": jnp.array(0.05, dtype=jnp.float64)}}

        new_state, _ = model.step(state, 0.0, dt, shocks, deps)

        expected = 100.0 * jnp.exp((0.05 - 0.02) * dt)
        assert jnp.isclose(new_state.level, expected, atol=1e-12)

    def test_no_deps_zero_rate(self, model: Equity) -> None:
        """Works with empty deps → r=0."""
        state = model.init_state()
        dt = 1.0 / 12.0
        shocks = jnp.array([0.0])
        deps: dict = {}

        new_state, _ = model.step(state, 0.0, dt, shocks, deps)

        sigma = 0.2
        expected_log = jnp.log(100.0) + (-0.5 * sigma**2) * dt
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)

    def test_mean_return_statistical(self) -> None:
        """10000 trials over 1 year → mean close to exp((r-q)T)."""
        params = GBMParams(sigma=0.2, initial_value=1.0)
        model = Equity(params=params, dividend_yield=0.0)
        r = 0.05
        dt = 1.0 / 12.0
        n_steps = 12
        n_trials = 10_000

        key = jax.random.PRNGKey(42)
        all_shocks = jax.random.normal(key, shape=(n_trials, n_steps))

        final_levels = []
        for trial in range(n_trials):
            state = model.init_state()
            for step_i in range(n_steps):
                shocks = jnp.array([all_shocks[trial, step_i]])
                deps: dict = {"rates": {"short_rate": jnp.array(r, dtype=jnp.float64)}}
                state, _ = model.step(state, step_i * dt, dt, shocks, deps)
            final_levels.append(float(state.level))

        mean_level = jnp.mean(jnp.array(final_levels))
        expected_mean = jnp.exp(r * 1.0)
        # E[S_T] = S_0 * exp(r*T) for GBM (the -σ²/2 in log cancels in expectation)
        assert jnp.abs(mean_level - expected_mean) < 0.05


class TestEquityRegistry:
    def test_registry_lookup(self) -> None:
        cls = get_model("equity")
        assert cls.__name__ == "Equity"
