"""Tests for the Inflation index model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import GBMParams, SeasonalityParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.core.types import FXState
from hyesg.models.inflation.inflation import Inflation

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the inflation model for each test."""
    clear_registry()

    import importlib

    import hyesg.models.inflation.inflation as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def seasonality() -> SeasonalityParams:
    return SeasonalityParams(a1=1.5, a2=-0.5, b1=0.8, b2=0.3)


@pytest.fixture
def params() -> GBMParams:
    return GBMParams(sigma=0.02, initial_value=100.0)


@pytest.fixture
def model(params: GBMParams, seasonality: SeasonalityParams) -> Inflation:
    return Inflation(
        params=params,
        name="inflation",
        real_rate_model="real_rates",
        seasonality_params=seasonality,
    )


class TestSeasonalityFormula:
    def test_known_values(self, model: Inflation) -> None:
        """Known coefficients → verify seasonal values at specific times."""
        # At t=0: shift = 0.5
        val_0 = model.seasonal_adjustment(0.0)
        s = 0.5
        two_pi = 2.0 * jnp.pi
        expected = 0.01 * (
            1.5 * jnp.cos(two_pi * s)
            + (-0.5) * jnp.cos(2.0 * two_pi * s)
            + 0.8 * jnp.sin(two_pi * s)
            + 0.3 * jnp.sin(2.0 * two_pi * s)
        )
        assert jnp.isclose(val_0, expected, atol=1e-14)

    def test_quarterly_values(self, model: Inflation) -> None:
        """Verify at t=0, 0.25, 0.5, 0.75."""
        for t_val in [0.0, 0.25, 0.5, 0.75]:
            val = model.seasonal_adjustment(t_val)
            shift = t_val + 0.5
            two_pi = 2.0 * jnp.pi
            expected = 0.01 * (
                1.5 * jnp.cos(two_pi * shift)
                + (-0.5) * jnp.cos(2.0 * two_pi * shift)
                + 0.8 * jnp.sin(two_pi * shift)
                + 0.3 * jnp.sin(2.0 * two_pi * shift)
            )
            assert jnp.isclose(val, expected, atol=1e-14)

    def test_periodicity(self, model: Inflation) -> None:
        """seasonal(t) = seasonal(t+1) since cos/sin are 2π-periodic."""
        for t_val in [0.0, 0.1, 0.37, 0.75]:
            val_t = model.seasonal_adjustment(t_val)
            val_t1 = model.seasonal_adjustment(t_val + 1.0)
            assert jnp.isclose(val_t, val_t1, atol=1e-14)

    def test_no_seasonality_zero(self) -> None:
        """None params → zero adjustment."""
        params = GBMParams(sigma=0.02, initial_value=100.0)
        model = Inflation(params=params, seasonality_params=None)
        val = model.seasonal_adjustment(0.25)
        assert jnp.isclose(val, 0.0, atol=1e-14)

    def test_fourier_integral_near_zero(self, model: Inflation) -> None:
        """Sum of seasonal adjustments over 12 months ≈ 0.

        Fourier harmonics integrate to zero over a full period.
        """
        total = jnp.array(0.0, dtype=jnp.float64)
        for month in range(12):
            t_val = month / 12.0
            total = total + model.seasonal_adjustment(t_val)
        assert jnp.abs(total) < 1e-12


class TestInflationInit:
    def test_initial_state(self, model: Inflation) -> None:
        state = model.init_state()
        assert isinstance(state, FXState)
        assert jnp.isclose(state.level, 100.0, atol=1e-12)
        assert jnp.isclose(state.log_level, jnp.log(100.0), atol=1e-12)

    def test_name(self, model: Inflation) -> None:
        assert model.name == "inflation"

    def test_shock_config(self, model: Inflation) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True


class TestInflationStep:
    def test_index_growth_with_real_rate(self, model: Inflation) -> None:
        """Index tracks real rate minus seasonality."""
        state = model.init_state()
        dt = 1.0 / 12.0
        shocks = jnp.array([0.0])
        deps = {
            "real_rates": {"short_rate": jnp.array(0.02, dtype=jnp.float64)},
        }

        new_state, outputs = model.step(state, 0.0, dt, shocks, deps)

        seasonal = model.seasonal_adjustment(0.0)
        adjusted = 0.02 - seasonal
        sigma = 0.02
        expected_log = jnp.log(100.0) + (adjusted - 0.5 * sigma**2) * dt
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)
        assert jnp.isclose(outputs["rate"], adjusted, atol=1e-14)

    def test_zero_vol_deterministic(self) -> None:
        """σ=0 → deterministic index path."""
        params = GBMParams(sigma=0.0, initial_value=100.0)
        model = Inflation(
            params=params,
            real_rate_model="real",
            seasonality_params=None,
        )
        state = model.init_state()
        dt = 0.25
        shocks = jnp.array([3.0])  # should not matter with σ=0
        deps = {"real": {"short_rate": jnp.array(0.03, dtype=jnp.float64)}}

        new_state, _ = model.step(state, 0.0, dt, shocks, deps)

        expected = 100.0 * jnp.exp(0.03 * dt)
        assert jnp.isclose(new_state.level, expected, atol=1e-12)

    def test_no_deps_zero_rate(self) -> None:
        """Works with empty deps."""
        params = GBMParams(sigma=0.02, initial_value=100.0)
        model = Inflation(params=params)
        state = model.init_state()
        dt = 1.0 / 12.0
        shocks = jnp.array([0.0])

        new_state, outputs = model.step(state, 0.0, dt, shocks, {})

        sigma = 0.02
        expected_log = jnp.log(100.0) + (-0.5 * sigma**2) * dt
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)
        assert jnp.isclose(outputs["rate"], 0.0, atol=1e-14)


class TestInflationRegistry:
    def test_registry_lookup(self) -> None:
        cls = get_model("inflation")
        assert cls.__name__ == "Inflation"
