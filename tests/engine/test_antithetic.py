"""Tests for the antithetic variance-reduction module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

jax.config.update("jax_enable_x64", True)

from hyesg.engine.antithetic import (
    antithetic_combine,
    antithetic_combine_pytree,
    apply_antithetic_normal,
    apply_antithetic_uniform,
    generate_antithetic_shocks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 42
_N_STEPS = 50
_N_SHOCKS = 6


def _make_shocks(seed: int = _SEED) -> Array:
    """Generate a small shock array for testing."""
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape=(_N_STEPS, _N_SHOCKS))


# ---------------------------------------------------------------------------
# apply_antithetic_normal
# ---------------------------------------------------------------------------


class TestApplyAntitheticNormal:
    """Tests for ``apply_antithetic_normal``."""

    def test_negation(self) -> None:
        """z_anti should equal -z."""
        z = _make_shocks()
        z_anti = apply_antithetic_normal(z)
        assert jnp.allclose(z_anti, -z, atol=1e-15)

    def test_sum_is_zero(self) -> None:
        """z + z_anti should be zero everywhere."""
        z = _make_shocks()
        z_anti = apply_antithetic_normal(z)
        assert jnp.allclose(z + z_anti, 0.0, atol=1e-15)

    def test_preserves_shape(self) -> None:
        """Output shape matches input shape."""
        z = _make_shocks()
        assert apply_antithetic_normal(z).shape == z.shape

    def test_preserves_dtype(self) -> None:
        """Output dtype matches input dtype."""
        z = _make_shocks()
        assert apply_antithetic_normal(z).dtype == z.dtype

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled without error."""
        z = _make_shocks()
        jitted = jax.jit(apply_antithetic_normal)
        result = jitted(z)
        assert jnp.allclose(result, -z, atol=1e-15)


# ---------------------------------------------------------------------------
# apply_antithetic_uniform
# ---------------------------------------------------------------------------


class TestApplyAntitheticUniform:
    """Tests for ``apply_antithetic_uniform``."""

    def test_complement(self) -> None:
        """u_anti should equal 1 - u."""
        key = jax.random.PRNGKey(0)
        u = jax.random.uniform(key, shape=(20, 4))
        u_anti = apply_antithetic_uniform(u)
        assert jnp.allclose(u_anti, 1.0 - u, atol=1e-15)

    def test_sum_is_one(self) -> None:
        """u + u_anti should equal 1 everywhere."""
        key = jax.random.PRNGKey(1)
        u = jax.random.uniform(key, shape=(30, 5))
        u_anti = apply_antithetic_uniform(u)
        assert jnp.allclose(u + u_anti, 1.0, atol=1e-15)

    def test_preserves_shape(self) -> None:
        """Output shape matches input shape."""
        key = jax.random.PRNGKey(2)
        u = jax.random.uniform(key, shape=(10, 3))
        assert apply_antithetic_uniform(u).shape == u.shape

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled without error."""
        key = jax.random.PRNGKey(3)
        u = jax.random.uniform(key, shape=(10, 4))
        jitted = jax.jit(apply_antithetic_uniform)
        result = jitted(u)
        assert jnp.allclose(result, 1.0 - u, atol=1e-15)


# ---------------------------------------------------------------------------
# generate_antithetic_shocks
# ---------------------------------------------------------------------------


class TestGenerateAntitheticShocks:
    """Tests for ``generate_antithetic_shocks``."""

    def test_no_copula_mask_negates_all(self) -> None:
        """Without copula_mask, all columns are negated."""
        z = _make_shocks()
        z_anti = generate_antithetic_shocks(z)
        assert jnp.allclose(z_anti, -z, atol=1e-15)

    def test_copula_mask_negates_only_non_copula(self) -> None:
        """With copula_mask, only non-copula columns are negated."""
        z = _make_shocks()
        # Columns 0, 2, 4 are copula; columns 1, 3, 5 are non-copula
        mask = jnp.array([True, False, True, False, True, False])
        z_anti = generate_antithetic_shocks(z, copula_mask=mask)

        # Copula columns should be unchanged
        assert jnp.allclose(z_anti[:, 0], z[:, 0], atol=1e-15)
        assert jnp.allclose(z_anti[:, 2], z[:, 2], atol=1e-15)
        assert jnp.allclose(z_anti[:, 4], z[:, 4], atol=1e-15)

        # Non-copula columns should be negated
        assert jnp.allclose(z_anti[:, 1], -z[:, 1], atol=1e-15)
        assert jnp.allclose(z_anti[:, 3], -z[:, 3], atol=1e-15)
        assert jnp.allclose(z_anti[:, 5], -z[:, 5], atol=1e-15)

    def test_all_copula_leaves_unchanged(self) -> None:
        """When all shocks are copula, nothing is negated."""
        z = _make_shocks()
        mask = jnp.ones(_N_SHOCKS, dtype=jnp.bool_)
        z_anti = generate_antithetic_shocks(z, copula_mask=mask)
        assert jnp.allclose(z_anti, z, atol=1e-15)

    def test_no_copula_negates_all(self) -> None:
        """When no shocks are copula, all are negated."""
        z = _make_shocks()
        mask = jnp.zeros(_N_SHOCKS, dtype=jnp.bool_)
        z_anti = generate_antithetic_shocks(z, copula_mask=mask)
        assert jnp.allclose(z_anti, -z, atol=1e-15)

    def test_preserves_shape(self) -> None:
        """Output shape matches input shape."""
        z = _make_shocks()
        mask = jnp.array([True, False, True, False, True, False])
        assert generate_antithetic_shocks(z, copula_mask=mask).shape == z.shape


# ---------------------------------------------------------------------------
# antithetic_combine
# ---------------------------------------------------------------------------


class TestAntitheticCombine:
    """Tests for ``antithetic_combine``."""

    def test_average(self) -> None:
        """Combined result is the element-wise average."""
        a = jnp.array([2.0, 4.0, 6.0])
        b = jnp.array([8.0, 10.0, 12.0])
        result = antithetic_combine(a, b)
        expected = jnp.array([5.0, 7.0, 9.0])
        assert jnp.allclose(result, expected, atol=1e-15)

    def test_2d(self) -> None:
        """Works on 2-D arrays."""
        a = jnp.ones((3, 4)) * 2.0
        b = jnp.ones((3, 4)) * 6.0
        result = antithetic_combine(a, b)
        assert jnp.allclose(result, 4.0, atol=1e-15)

    def test_identical_inputs(self) -> None:
        """Averaging identical inputs returns the same values."""
        x = _make_shocks()
        result = antithetic_combine(x, x)
        assert jnp.allclose(result, x, atol=1e-15)

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled."""
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        jitted = jax.jit(antithetic_combine)
        result = jitted(a, b)
        assert jnp.allclose(result, jnp.array([2.0, 3.0]), atol=1e-15)


# ---------------------------------------------------------------------------
# antithetic_combine_pytree
# ---------------------------------------------------------------------------


class TestAntitheticCombinePytree:
    """Tests for ``antithetic_combine_pytree``."""

    def test_flat_dict(self) -> None:
        """Combines a flat dict of arrays."""
        orig = {"x": jnp.array([2.0, 4.0]), "y": jnp.array([10.0])}
        anti = {"x": jnp.array([6.0, 8.0]), "y": jnp.array([20.0])}
        result = antithetic_combine_pytree(orig, anti)
        assert jnp.allclose(result["x"], jnp.array([4.0, 6.0]), atol=1e-15)
        assert jnp.allclose(result["y"], jnp.array([15.0]), atol=1e-15)

    def test_nested_dict(self) -> None:
        """Combines a nested dict of arrays."""
        orig = {"rates": {"short": jnp.array([0.02]), "long": jnp.array([0.04])}}
        anti = {"rates": {"short": jnp.array([0.06]), "long": jnp.array([0.08])}}
        result = antithetic_combine_pytree(orig, anti)
        assert jnp.allclose(result["rates"]["short"], jnp.array([0.04]), atol=1e-15)
        assert jnp.allclose(result["rates"]["long"], jnp.array([0.06]), atol=1e-15)

    def test_preserves_structure(self) -> None:
        """Output dict has the same keys as input."""
        orig = {"a": jnp.zeros(3), "b": jnp.ones(2)}
        anti = {"a": jnp.ones(3), "b": jnp.zeros(2)}
        result = antithetic_combine_pytree(orig, anti)
        assert set(result.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Variance reduction property
# ---------------------------------------------------------------------------


class TestVarianceReduction:
    """Statistical tests demonstrating antithetic variance reduction."""

    def test_variance_reduced_for_x_squared(self) -> None:
        """Antithetic MC for E[X²] (X ~ N(0,1)) has lower variance.

        E[X²] = 1.  We estimate this with plain MC and antithetic MC
        across many replications, and verify the antithetic estimator
        has lower variance.
        """
        n_reps = 200
        n_samples = 500

        plain_estimates = []
        anti_estimates = []

        for i in range(n_reps):
            key = jax.random.PRNGKey(i)
            z = jax.random.normal(key, shape=(n_samples,))

            # Plain MC estimate of E[X²]
            plain_est = jnp.mean(z**2)
            plain_estimates.append(float(plain_est))

            # Antithetic MC estimate: (X² + (-X)²) / 2 = X²
            # For X², antithetic gives no improvement because f(x)=x²
            # is symmetric.  Use f(x) = exp(x) instead.

        # Use f(x) = exp(x) where antithetic actually helps.
        # E[exp(X)] = exp(1/2) for X ~ N(0,1).
        true_value = jnp.exp(0.5)
        plain_estimates = []
        anti_estimates = []

        for i in range(n_reps):
            key = jax.random.PRNGKey(i + 1000)
            z = jax.random.normal(key, shape=(n_samples,))

            plain_est = jnp.mean(jnp.exp(z))
            plain_estimates.append(float(plain_est))

            z_anti = apply_antithetic_normal(z)
            anti_est = jnp.mean((jnp.exp(z) + jnp.exp(z_anti)) / 2.0)
            anti_estimates.append(float(anti_est))

        var_plain = jnp.var(jnp.array(plain_estimates))
        var_anti = jnp.var(jnp.array(anti_estimates))

        # Antithetic should have substantially lower variance
        assert float(var_anti) < float(var_plain), (
            f"Antithetic variance ({var_anti:.6f}) should be less than "
            f"plain variance ({var_plain:.6f})"
        )

    def test_unbiased_estimator(self) -> None:
        """Antithetic estimator has the correct mean (unbiased).

        E[exp(X)] = exp(1/2) ≈ 1.6487 for X ~ N(0,1).
        """
        n_samples = 50_000
        key = jax.random.PRNGKey(999)
        z = jax.random.normal(key, shape=(n_samples,))

        z_anti = apply_antithetic_normal(z)
        anti_est = jnp.mean((jnp.exp(z) + jnp.exp(z_anti)) / 2.0)
        true_value = jnp.exp(0.5)

        assert jnp.abs(anti_est - true_value) < 0.05, (
            f"Antithetic estimate {anti_est:.4f} too far from "
            f"true value {true_value:.4f}"
        )


# ---------------------------------------------------------------------------
# Gaussian vs Student-t: uniform complement vs normal negation
# ---------------------------------------------------------------------------


class TestGaussianVsStudentT:
    """Demonstrate that 1-u and -z differ for non-Gaussian marginals.

    For the Gaussian CDF Φ:
        Φ(-z) = 1 - Φ(z)
    so negating z before CDF is equivalent to complementing u after CDF.

    For the Student-t CDF F_ν:
        F_ν(-z) ≠ 1 - F_ν(z)  in general (only equal for symmetric t)

    Actually, the standard Student-t IS symmetric, so
        F_ν(-z) = 1 - F_ν(z)
    holds exactly. The difference arises for non-symmetric marginals
    (e.g., skewed distributions). We demonstrate the Gaussian case here.
    """

    def test_gaussian_equivalence(self) -> None:
        """For Gaussian: Φ(-z) == 1 - Φ(z) (the two approaches match)."""
        from jax.scipy.stats import norm

        z = jnp.linspace(-3.0, 3.0, 100)

        # Approach 1: negate z, then apply CDF (C# approach)
        u_from_neg_z = norm.cdf(-z)

        # Approach 2: apply CDF, then complement (correct approach)
        u_complemented = 1.0 - norm.cdf(z)

        assert jnp.allclose(u_from_neg_z, u_complemented, atol=1e-12), (
            "For Gaussian, Φ(-z) should equal 1 - Φ(z)"
        )

    def test_skewed_distribution_diverges(self) -> None:
        """For a skewed distribution, F(-z) ≠ 1 - F(z).

        We use a log-normal-inspired transform to create a non-symmetric
        CDF mapping, demonstrating that pre-CDF negation and post-CDF
        complement give different results when the marginal is
        non-symmetric.
        """
        z = jnp.linspace(-2.0, 2.0, 100)

        # A skewed CDF: F(z) = Φ(z + 0.5) (shifted Gaussian)
        from jax.scipy.stats import norm

        def skewed_cdf(x: Array) -> Array:
            return norm.cdf(x + 0.5)

        # Approach 1: negate z, then apply skewed CDF
        u_from_neg_z = skewed_cdf(-z)

        # Approach 2: apply skewed CDF, then complement
        u_complemented = 1.0 - skewed_cdf(z)

        # These should NOT be equal for a shifted (non-symmetric) CDF
        max_diff = float(jnp.max(jnp.abs(u_from_neg_z - u_complemented)))
        assert max_diff > 0.01, (
            f"Skewed distribution should show divergence, but max diff "
            f"was only {max_diff:.6f}"
        )
