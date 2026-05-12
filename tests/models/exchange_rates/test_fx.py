"""Tests for the FX (exchange rate) model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import GBMParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.core.types import FXState
from hyesg.models.exchange_rates.fx import FXRate

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the fx model for each test."""
    clear_registry()

    import importlib

    import hyesg.models.exchange_rates.fx as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def params() -> GBMParams:
    return GBMParams(sigma=0.1, initial_value=1.5)


@pytest.fixture
def model(params: GBMParams) -> FXRate:
    return FXRate(
        params=params,
        name="fx",
        domestic_rate_model="domestic",
        foreign_rate_model="foreign",
    )


class TestFXInit:
    def test_initial_state(self, model: FXRate) -> None:
        """s0=1.5 → correct FXState."""
        state = model.init_state()
        assert isinstance(state, FXState)
        assert jnp.isclose(state.level, 1.5, atol=1e-12)
        assert jnp.isclose(state.log_level, jnp.log(1.5), atol=1e-12)

    def test_name(self, model: FXRate) -> None:
        assert model.name == "fx"


class TestFXShockConfig:
    def test_n_shocks(self, model: FXRate) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: FXRate) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert cfg.names == ("fx_z",)


class TestFXStep:
    def test_rate_differential(self, model: FXRate) -> None:
        """r_d > r_f → FX tends to appreciate over many steps."""
        state = model.init_state()
        dt = 1.0 / 12.0
        n_steps = 120  # 10 years

        # Use zero shocks → deterministic path
        for step_i in range(n_steps):
            shocks = jnp.array([0.0])
            deps = {
                "domestic": {"ShortRate": jnp.array(0.05, dtype=jnp.float64)},
                "foreign": {"ShortRate": jnp.array(0.01, dtype=jnp.float64)},
            }
            state, _ = model.step(state, step_i * dt, dt, shocks, deps)

        sigma = 0.1
        drift_per_step = (0.05 - 0.01 - 0.5 * sigma**2) * dt
        expected_log = jnp.log(1.5) + drift_per_step * n_steps
        assert jnp.isclose(state.log_level, expected_log, atol=1e-10)

    def test_zero_vol_deterministic(self) -> None:
        """σ=0 → deterministic path based on rate differential."""
        params = GBMParams(sigma=0.0, initial_value=1.0)
        model = FXRate(
            params=params,
            domestic_rate_model="dom",
            foreign_rate_model="for",
        )
        state = model.init_state()
        dt = 0.5
        shocks = jnp.array([2.0])  # should not matter with σ=0
        deps = {
            "dom": {"ShortRate": jnp.array(0.06, dtype=jnp.float64)},
            "for": {"ShortRate": jnp.array(0.02, dtype=jnp.float64)},
        }

        new_state, outputs = model.step(state, 0.0, dt, shocks, deps)

        expected = jnp.exp((0.06 - 0.02) * dt)
        assert jnp.isclose(new_state.level, expected, atol=1e-12)
        assert jnp.isclose(outputs["ExchangeRate"], expected, atol=1e-12)

    def test_no_deps_zero_rates(self) -> None:
        """Works with zero rates when no dep models configured."""
        params = GBMParams(sigma=0.1, initial_value=1.0)
        model = FXRate(params=params)  # no domestic/foreign
        state = model.init_state()
        dt = 1.0 / 12.0
        shocks = jnp.array([0.0])
        deps: dict = {}

        new_state, _ = model.step(state, 0.0, dt, shocks, deps)

        sigma = 0.1
        expected_log = jnp.log(1.0) + (-0.5 * sigma**2) * dt
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)


class TestFXRegistry:
    def test_registry_lookup(self) -> None:
        cls = get_model("fx")
        assert cls.__name__ == "FXRate"
