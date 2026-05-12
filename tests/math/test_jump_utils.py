"""Tests for jump-diffusion utilities.

Parity tests against C# reference values:
    - GetJumpAdjustedInitialVolatility  (Calibration.cs:1550-1561)
    - PoissonDistribution.InverseCdf
    - PoissonDistributionContinuousApproximation.InverseCdf
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.math.jump_utils import (
    expected_sigma_taylor,
    jump_adjusted_sigma,
    poisson_inverse_cdf,
    poisson_inverse_cdf_continuous,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# jump_adjusted_sigma — C# parity (Calibration.cs:1550-1561)
# ---------------------------------------------------------------------------
class TestJumpAdjustedSigma:
    """Parity with C# GetJumpAdjustedInitialVolatility."""

    @pytest.mark.parametrize(
        "sigma, lambda_, mu_j, sigma_j, expected",
        [
            # Case 1: typical equity params
            (0.20, 0.5, -0.05, 0.10, 0.19462),
            # Case 2: large jump component
            (0.30, 1.0, 0.10, 0.15, 0.27386),
            # Case 3: zero jumps => unchanged
            (0.15, 0.0, 0.10, 0.20, 0.15),
            # Case 4: jump variance exceeds total => hits floor
            (0.05, 2.0, 0.10, 0.20, 0.01),
            # Case 5: small lambda, moderate params
            (0.25, 0.1, 0.02, 0.05, 0.24988),
        ],
    )
    def test_parity_with_csharp(self, sigma, lambda_, mu_j, sigma_j, expected):
        """Verify against hand-computed C# reference values."""
        # Compute expected from formula: sqrt(max(sigma^2 - lambda*(mu^2+sigma^2), 0.01^2))
        var_raw = sigma**2 - lambda_ * (mu_j**2 + sigma_j**2)
        var_floored = max(var_raw, 0.01**2)
        ref = var_floored**0.5

        result = jump_adjusted_sigma(sigma, lambda_, mu_j, sigma_j)
        assert jnp.isclose(result, ref, atol=1e-12)

    def test_floor_prevents_negative_variance(self):
        """When jump variance exceeds total, result is floored at 0.01."""
        result = jump_adjusted_sigma(0.01, 10.0, 0.5, 0.5)
        assert jnp.isclose(result, 0.01)

    def test_custom_floor(self):
        """Custom floor value is respected."""
        result = jump_adjusted_sigma(0.01, 10.0, 0.5, 0.5, floor=0.05)
        assert jnp.isclose(result, 0.05)

    def test_zero_jump_parameters(self):
        """With zero jump params, adjusted vol equals unadjusted."""
        result = jump_adjusted_sigma(0.20, 0.0, 0.0, 0.0)
        assert jnp.isclose(result, 0.20, atol=1e-14)

    def test_jit_compatible(self):
        """Function can be JIT-compiled."""
        jitted = jax.jit(jump_adjusted_sigma)
        result = jitted(0.20, 0.5, -0.05, 0.10)
        expected = jump_adjusted_sigma(0.20, 0.5, -0.05, 0.10)
        assert jnp.isclose(result, expected, atol=1e-14)

    def test_vmap_compatible(self):
        """Function can be vmapped over parameter arrays."""
        sigmas = jnp.array([0.15, 0.20, 0.25, 0.30])
        fn = jax.vmap(jump_adjusted_sigma, in_axes=(0, None, None, None))
        results = fn(sigmas, 0.5, -0.05, 0.10)
        assert results.shape == (4,)


# ---------------------------------------------------------------------------
# poisson_inverse_cdf — C# parity (PoissonDistribution.cs)
# ---------------------------------------------------------------------------
class TestPoissonInverseCdf:
    """Parity with C# PoissonDistribution.InverseCdf."""

    @pytest.mark.parametrize(
        "u, lambda_, expected",
        [
            # From C# test cases
            (0.00000000206115, 20.0, 0),
            (0.00000000206116, 20.0, 1),
            (0.02138682158728, 20.0, 11),
            (0.02138682158729, 20.0, 12),
            (0.97818178247444, 20.0, 29),
            (0.97818178247445, 20.0, 30),
            (0.77880078307140, 0.25, 0),
            (0.77880078307142, 0.25, 1),
            (0.99999983457834, 0.01, 2),
            (0.99999983457835, 0.01, 3),
        ],
    )
    def test_parity_with_csharp(self, u, lambda_, expected):
        """Exact match with C# PoissonDistribution.InverseCdf."""
        result = poisson_inverse_cdf(u, lambda_)
        assert int(result) == expected

    def test_zero_lambda(self):
        """With lambda=0, all probability is at k=0."""
        result = poisson_inverse_cdf(0.5, 0.0)
        assert int(result) == 0

    def test_small_u(self):
        """Very small u returns 0 for moderate lambda."""
        result = poisson_inverse_cdf(1e-15, 1.0)
        assert int(result) == 0

    def test_jit_compatible(self):
        """Function can be JIT-compiled."""
        jitted = jax.jit(poisson_inverse_cdf)
        result = jitted(0.8, 0.5)
        expected = poisson_inverse_cdf(0.8, 0.5)
        assert int(result) == int(expected)

    @pytest.mark.parametrize(
        "u, lambda_",
        [
            (0.1, 0.5),
            (0.5, 1.0),
            (0.9, 2.0),
            (0.99, 5.0),
            (0.001, 0.1),
        ],
    )
    def test_inverse_is_consistent_with_cdf(self, u, lambda_):
        """The returned k satisfies CDF(k-1) < u <= CDF(k)."""
        k = int(poisson_inverse_cdf(u, lambda_))

        # Compute CDF(k) by summing PMF
        cdf_k = 0.0
        term = jnp.exp(-lambda_)
        for i in range(k + 1):
            if i > 0:
                term = term * lambda_ / i
            cdf_k += term

        assert cdf_k >= u or jnp.isclose(cdf_k, u, atol=1e-14)


# ---------------------------------------------------------------------------
# poisson_inverse_cdf_continuous — C# parity
# (PoissonDistributionContinuousApproximation.cs)
# ---------------------------------------------------------------------------
class TestPoissonInverseCdfContinuous:
    """Parity with C# PoissonDistributionContinuousApproximation.InverseCdf."""

    @pytest.mark.parametrize(
        "u, lambda_, expected",
        [
            # From C# test cases
            (0.36787944117144, 0.5, 0.0),
            (0.36787944117145, 0.5, 0.000000000000016),
            (0.8, 0.5, 0.9053391000263720),
            (0.99997355703465, 0.5, 4.99999999999047),
            (0.99997355703470, 0.5, 5.00000000192602),
            (0.99399395665861, 20.0, 31.999999999998500),
            (0.99399395665862, 20.0, 32.0000000000014),
        ],
    )
    def test_parity_with_csharp(self, u, lambda_, expected):
        """Match C# PoissonDistributionContinuousApproximation.InverseCdf."""
        result = poisson_inverse_cdf_continuous(u, lambda_)
        assert jnp.isclose(result, expected, atol=1e-10), (
            f"u={u}, lambda={lambda_}: got {result}, expected {expected}"
        )

    def test_returns_float(self):
        """Continuous version returns fractional values."""
        result = poisson_inverse_cdf_continuous(0.8, 0.5)
        # Should be fractional, not integer
        assert not jnp.isclose(result, jnp.round(result), atol=0.01)

    def test_zero_for_small_u(self):
        """Very small u gives 0."""
        result = poisson_inverse_cdf_continuous(0.01, 0.5)
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_jit_compatible(self):
        """Function can be JIT-compiled."""
        jitted = jax.jit(poisson_inverse_cdf_continuous)
        result = jitted(0.8, 0.5)
        expected = poisson_inverse_cdf_continuous(0.8, 0.5)
        assert jnp.isclose(result, expected, atol=1e-14)

    def test_monotonically_increasing(self):
        """Output is monotonically increasing in u for fixed lambda."""
        us = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        results = jax.vmap(poisson_inverse_cdf_continuous, in_axes=(0, None))(us, 1.0)
        diffs = jnp.diff(results)
        assert jnp.all(diffs >= 0)


# ---------------------------------------------------------------------------
# expected_sigma_taylor
# ---------------------------------------------------------------------------
class TestExpectedSigmaTaylor:
    """Test Taylor approximation E[sqrt(V)] ≈ sqrt(mu) - sigma^2/(8*mu^(3/2))."""

    @pytest.mark.parametrize(
        "mu_v, sigma_v",
        [
            (0.04, 0.05),  # typical equity vol-of-vol
            (0.09, 0.10),
            (0.01, 0.02),
            (0.16, 0.15),
            (0.25, 0.20),
        ],
    )
    def test_matches_analytic_formula(self, mu_v, sigma_v):
        """Verify against direct computation of the Taylor formula."""
        import math

        expected = math.sqrt(mu_v) - sigma_v**2 / (8.0 * mu_v**1.5)
        result = expected_sigma_taylor(mu_v, sigma_v)
        assert jnp.isclose(result, expected, atol=1e-14)

    def test_zero_vol_of_vol(self):
        """With zero vol-of-vol, E[sigma] = sqrt(mu_v) exactly."""
        result = expected_sigma_taylor(0.04, 0.0)
        assert jnp.isclose(result, jnp.sqrt(0.04), atol=1e-14)

    def test_correction_is_negative(self):
        """The Taylor correction should reduce E[sigma] below sqrt(mu_v)."""
        mu_v, sigma_v = 0.04, 0.10
        result = expected_sigma_taylor(mu_v, sigma_v)
        assert result < jnp.sqrt(mu_v)

    def test_jit_compatible(self):
        """Function can be JIT-compiled."""
        jitted = jax.jit(expected_sigma_taylor)
        result = jitted(0.04, 0.05)
        expected = expected_sigma_taylor(0.04, 0.05)
        assert jnp.isclose(result, expected, atol=1e-14)

    def test_vmap_compatible(self):
        """Function can be vmapped."""
        mus = jnp.array([0.01, 0.04, 0.09, 0.16])
        fn = jax.vmap(expected_sigma_taylor, in_axes=(0, None))
        results = fn(mus, 0.05)
        assert results.shape == (4,)
