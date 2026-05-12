"""Tests for quanto adjustment functions."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hyesg.engine.quanto import quanto_adjustment, quanto_drift_adjustment


class TestQuantoAdjustment:
    """Tests for quanto_adjustment."""

    def test_zero_correlation_no_adjustment(self):
        """Zero correlation → no adjustment."""
        shock = jnp.array(0.5, dtype=jnp.float64)
        result = quanto_adjustment(
            shock,
            correlations=jnp.array([0.0]),
            fx_vols=jnp.array([0.1]),
            dt=0.25,
        )
        assert jnp.isclose(result, 0.5)

    def test_zero_fx_vol_no_adjustment(self):
        """Zero FX vol → no adjustment."""
        shock = jnp.array(0.5, dtype=jnp.float64)
        result = quanto_adjustment(
            shock,
            correlations=jnp.array([0.5]),
            fx_vols=jnp.array([0.0]),
            dt=0.25,
        )
        assert jnp.isclose(result, 0.5)

    def test_single_fx_adjustment(self):
        """Single FX rate quanto adjustment."""
        shock = jnp.array(1.0, dtype=jnp.float64)
        rho = 0.3
        sigma_fx = 0.15
        dt = 0.25
        expected = 1.0 - rho * sigma_fx * jnp.sqrt(dt)
        result = quanto_adjustment(
            shock,
            correlations=jnp.array([rho]),
            fx_vols=jnp.array([sigma_fx]),
            dt=dt,
        )
        assert jnp.isclose(result, expected, atol=1e-12)

    def test_multiple_fx_adjustment(self):
        """Multiple FX rates: adjustment is sum of products."""
        shock = jnp.array(0.0, dtype=jnp.float64)
        rho = jnp.array([0.3, 0.5])
        sigma_fx = jnp.array([0.1, 0.2])
        dt = 1.0
        expected = 0.0 - (0.3 * 0.1 + 0.5 * 0.2) * jnp.sqrt(1.0)
        result = quanto_adjustment(shock, rho, sigma_fx, dt)
        assert jnp.isclose(result, expected, atol=1e-12)

    def test_negative_correlation(self):
        """Negative correlation increases the shock."""
        shock = jnp.array(0.0, dtype=jnp.float64)
        result = quanto_adjustment(
            shock,
            correlations=jnp.array([-0.5]),
            fx_vols=jnp.array([0.1]),
            dt=1.0,
        )
        # -(-0.5 * 0.1 * 1.0) = +0.05
        assert result > 0.0

    def test_unit_dt(self):
        """dt=1 simplifies sqrt(dt) to 1."""
        shock = jnp.array(0.0, dtype=jnp.float64)
        rho = 0.4
        sigma_fx = 0.2
        result = quanto_adjustment(
            shock,
            correlations=jnp.array([rho]),
            fx_vols=jnp.array([sigma_fx]),
            dt=1.0,
        )
        assert jnp.isclose(result, -rho * sigma_fx, atol=1e-12)


class TestQuantoDriftAdjustment:
    """Tests for quanto_drift_adjustment."""

    def test_zero_correlation_no_drift(self):
        """Zero correlation → zero drift adjustment."""
        result = quanto_drift_adjustment(
            correlations=jnp.array([0.0]),
            fx_vols=jnp.array([0.1]),
            model_vol=0.2,
        )
        assert jnp.isclose(result, 0.0)

    def test_single_fx_drift(self):
        """Single FX drift adjustment."""
        rho = 0.3
        sigma_fx = 0.15
        sigma_model = 0.2
        expected = -sigma_model * rho * sigma_fx
        result = quanto_drift_adjustment(
            correlations=jnp.array([rho]),
            fx_vols=jnp.array([sigma_fx]),
            model_vol=sigma_model,
        )
        assert jnp.isclose(result, expected, atol=1e-12)

    def test_multiple_fx_drift(self):
        """Multiple FX drift adjustments sum."""
        rho = jnp.array([0.3, 0.5])
        sigma_fx = jnp.array([0.1, 0.2])
        sigma_model = 0.25
        expected = -0.25 * (0.3 * 0.1 + 0.5 * 0.2)
        result = quanto_drift_adjustment(rho, sigma_fx, sigma_model)
        assert jnp.isclose(result, expected, atol=1e-12)

    def test_negative_correlation_positive_drift(self):
        """Negative correlation gives positive drift adjustment."""
        result = quanto_drift_adjustment(
            correlations=jnp.array([-0.5]),
            fx_vols=jnp.array([0.1]),
            model_vol=0.2,
        )
        assert result > 0.0
