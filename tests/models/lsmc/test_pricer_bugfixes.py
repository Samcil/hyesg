"""Regression tests for LSMC pricer bug-fixes (Phase 3, Issue #33)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.lsmc.basis import laguerre_basis
from hyesg.models.lsmc.pricer import LSMCConfig, LSMCPricer

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# Bug 1: off-by-one in _build_discount_factors
# ------------------------------------------------------------------


class TestBuildDiscountFactors:
    """D(0, t_0) must be 1.0 (no discounting at t=0)."""

    def test_first_column_is_one(self) -> None:
        short_rates = jnp.array([[0.05, 0.05, 0.05]])
        dt = 1.0
        df = LSMCPricer._build_discount_factors(short_rates, dt)
        assert jnp.isclose(df[0, 0], 1.0, atol=1e-14)

    def test_second_column_discounts_first_rate(self) -> None:
        r = 0.04
        short_rates = jnp.array([[r, r, r]])
        dt = 0.5
        df = LSMCPricer._build_discount_factors(short_rates, dt)
        # D(0, t_1) = exp(-r * dt)
        expected = jnp.exp(-r * dt)
        assert jnp.isclose(df[0, 1], expected, atol=1e-14)

    def test_monotonically_decreasing(self) -> None:
        short_rates = jnp.ones((5, 10)) * 0.03
        dt = 0.25
        df = LSMCPricer._build_discount_factors(short_rates, dt)
        # Each column should be <= previous column
        assert jnp.all(jnp.diff(df, axis=1) <= 0.0)

    def test_multiple_trials(self) -> None:
        short_rates = jnp.array([[0.02, 0.03], [0.05, 0.04]])
        dt = 1.0
        df = LSMCPricer._build_discount_factors(short_rates, dt)
        assert jnp.allclose(df[:, 0], 1.0, atol=1e-14)


# ------------------------------------------------------------------
# Bug 4: LSMCConfig validation
# ------------------------------------------------------------------


class TestLSMCConfigValidation:
    def test_valid_defaults(self) -> None:
        cfg = LSMCConfig()
        assert cfg.degree == 3

    def test_degree_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="degree must be > 0"):
            LSMCConfig(degree=0)

    def test_degree_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="degree must be > 0"):
            LSMCConfig(degree=-2)

    def test_invalid_basis_raises(self) -> None:
        with pytest.raises(ValueError, match="basis must be one of"):
            LSMCConfig(basis="hermite")

    def test_invalid_exercise_type_raises(self) -> None:
        with pytest.raises(ValueError, match="exercise_type must be one of"):
            LSMCConfig(exercise_type="asian")

    def test_valid_laguerre(self) -> None:
        cfg = LSMCConfig(basis="laguerre", degree=5, exercise_type="bermudan")
        assert cfg.basis == "laguerre"


# ------------------------------------------------------------------
# Bug 7: Laguerre overflow for negative inputs
# ------------------------------------------------------------------


class TestLaguerreBasisOverflow:
    def test_large_negative_inputs_no_inf(self) -> None:
        """exp(-u/2) should not overflow when u is very negative."""
        x = jnp.array([-1000.0, -500.0, -100.0, 0.0, 100.0])
        basis = laguerre_basis(x, degree=3)
        assert jnp.all(jnp.isfinite(basis))

    def test_large_positive_inputs_no_inf(self) -> None:
        x = jnp.array([1000.0, 500.0, 100.0])
        basis = laguerre_basis(x, degree=3)
        assert jnp.all(jnp.isfinite(basis))

    def test_normal_inputs_unchanged(self) -> None:
        """Normal-range inputs should produce identical results to before."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        basis = laguerre_basis(x, degree=3)
        assert jnp.all(jnp.isfinite(basis))
        assert basis.shape == (4, 4)
