"""Tests for Gaussian (OU) helper functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.math.gaussian_helpers import b_func, b_over_dt, variance_integral_ou

jax.config.update("jax_enable_x64", True)

ALPHA = 0.5
SIGMA = 0.1


class TestBOverDt:
    """Tests for (1 - e^{-y}) / y with Taylor guard."""

    def test_at_zero(self) -> None:
        """b_over_dt(0) = 1 (via Taylor expansion)."""
        result = b_over_dt(0.0)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_small_y(self) -> None:
        """Taylor expansion should be accurate for small y."""
        y = 1e-10
        result = b_over_dt(y)
        assert float(result) == pytest.approx(1.0, abs=1e-8)

    def test_positive_y(self) -> None:
        """Standard computation for moderate y."""
        y = 1.0
        expected = (1.0 - jnp.exp(-1.0)) / 1.0
        result = b_over_dt(y)
        assert float(result) == pytest.approx(float(expected), abs=1e-12)

    def test_large_y(self) -> None:
        """For large y, result → 1/y."""
        y = 100.0
        result = b_over_dt(y)
        assert float(result) == pytest.approx(1.0 / y, abs=1e-6)

    def test_negative_y(self) -> None:
        """Should work for negative y too."""
        y = -1.0
        expected = (1.0 - jnp.exp(1.0)) / (-1.0)
        result = b_over_dt(y)
        assert float(result) == pytest.approx(float(expected), abs=1e-12)

    def test_vectorized(self) -> None:
        """Should accept array inputs."""
        ys = jnp.array([0.0, 0.5, 1.0, 5.0])
        results = b_over_dt(ys)
        assert results.shape == (4,)
        assert float(results[0]) == pytest.approx(1.0, abs=1e-8)


class TestBFunc:
    """Tests for B(α, τ) = (1 - e^{-ατ}) / α."""

    def test_zero_tau(self) -> None:
        """B(α, 0) = 0."""
        result = b_func(ALPHA, 0.0)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self) -> None:
        """B(α, τ) matches analytic expression."""
        tau = 5.0
        expected = (1.0 - jnp.exp(-ALPHA * tau)) / ALPHA
        result = b_func(ALPHA, tau)
        assert float(result) == pytest.approx(float(expected), abs=1e-12)

    def test_limit_large_tau(self) -> None:
        """B(α, τ) → 1/α as τ → ∞."""
        result = b_func(ALPHA, 1000.0)
        assert float(result) == pytest.approx(1.0 / ALPHA, abs=1e-6)

    def test_small_alpha_tau(self) -> None:
        """For small ατ, B ≈ τ (via Taylor)."""
        tau = 0.001
        result = b_func(ALPHA, tau)
        assert float(result) == pytest.approx(tau, rel=1e-3)

    def test_vectorized(self) -> None:
        """Should accept array tau."""
        taus = jnp.array([0.0, 1.0, 5.0, 10.0])
        results = b_func(ALPHA, taus)
        assert results.shape == (4,)
        assert float(results[0]) == pytest.approx(0.0, abs=1e-12)


class TestVarianceIntegralOU:
    """Tests for V²(σ, α, τ)."""

    def test_zero_tau(self) -> None:
        """V²(σ, α, 0) = 0."""
        result = variance_integral_ou(SIGMA, ALPHA, 0.0)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_positive(self) -> None:
        """V² > 0 for τ > 0."""
        for tau in [0.1, 1.0, 5.0, 10.0]:
            result = variance_integral_ou(SIGMA, ALPHA, tau)
            assert float(result) > 0.0

    def test_monotone_increasing(self) -> None:
        """V² should increase with τ."""
        v1 = float(variance_integral_ou(SIGMA, ALPHA, 1.0))
        v5 = float(variance_integral_ou(SIGMA, ALPHA, 5.0))
        v10 = float(variance_integral_ou(SIGMA, ALPHA, 10.0))
        assert v1 < v5 < v10

    def test_scales_with_sigma_squared(self) -> None:
        """V² ∝ σ²."""
        tau = 5.0
        v1 = float(variance_integral_ou(SIGMA, ALPHA, tau))
        v2 = float(variance_integral_ou(2.0 * SIGMA, ALPHA, tau))
        assert v2 == pytest.approx(4.0 * v1, rel=1e-10)

    def test_vectorized(self) -> None:
        """Should accept array tau."""
        taus = jnp.array([0.0, 1.0, 5.0])
        results = variance_integral_ou(SIGMA, ALPHA, taus)
        assert results.shape == (3,)
        assert float(results[0]) == pytest.approx(0.0, abs=1e-12)

    def test_known_formula(self) -> None:
        """Verify against direct computation."""
        tau = 3.0
        B = float(b_func(ALPHA, tau))
        expected = (SIGMA**2 / ALPHA**2) * (tau - B - 0.5 * ALPHA * B**2)
        result = variance_integral_ou(SIGMA, ALPHA, tau)
        assert float(result) == pytest.approx(expected, abs=1e-12)
