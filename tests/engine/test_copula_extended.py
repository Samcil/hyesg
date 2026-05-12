"""Tests for chi-squared and Student-t sampling in the copula module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.engine.copula import chi_squared_sample, student_t_sample

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Chi-squared sampling
# ---------------------------------------------------------------------------


class TestChiSquaredSample:
    """Tests for chi_squared_sample."""

    def test_positive_samples(self) -> None:
        """All chi-squared samples are positive."""
        key = jax.random.PRNGKey(0)
        samples = chi_squared_sample(key, df=5, shape=(10000,))
        assert jnp.all(samples > 0)

    def test_mean_approx_df(self) -> None:
        """Mean of chi-squared samples ≈ df (statistical test)."""
        key = jax.random.PRNGKey(1)
        df = 10
        samples = chi_squared_sample(key, df=df, shape=(50000,))
        mean = jnp.mean(samples)
        # With 50k samples, mean should be within ~0.2 of df
        assert abs(mean - df) < 0.5, f"Mean {mean} too far from df={df}"

    def test_variance_approx_2df(self) -> None:
        """Variance of chi-squared samples ≈ 2*df."""
        key = jax.random.PRNGKey(2)
        df = 8
        samples = chi_squared_sample(key, df=df, shape=(50000,))
        var = jnp.var(samples)
        expected_var = 2.0 * df
        assert abs(var - expected_var) < 2.0, (
            f"Var {var} too far from 2*df={expected_var}"
        )

    def test_shape_scalar(self) -> None:
        """Empty shape produces a scalar."""
        key = jax.random.PRNGKey(3)
        result = chi_squared_sample(key, df=5, shape=())
        assert result.shape == ()

    def test_shape_batch(self) -> None:
        """Batch shape is respected."""
        key = jax.random.PRNGKey(4)
        result = chi_squared_sample(key, df=5, shape=(3, 7))
        assert result.shape == (3, 7)


# ---------------------------------------------------------------------------
# Student-t sampling
# ---------------------------------------------------------------------------


class TestStudentTSample:
    """Tests for student_t_sample."""

    def test_heavier_tails_than_normal(self) -> None:
        """Student-t samples have more extreme values than normal."""
        normal_key = jax.random.PRNGKey(10)
        chi2_key = jax.random.PRNGKey(11)

        n = 50000
        t_samples = student_t_sample(normal_key, chi2_key, df=5, shape=(n,))
        normal_samples = jax.random.normal(normal_key, shape=(n,))

        # Kurtosis of Student-t(5) = 6/(5-4) + 3 = 9 (excess = 6)
        # Kurtosis of normal = 3 (excess = 0)
        # More values beyond ±3 for Student-t
        t_extreme = jnp.sum(jnp.abs(t_samples) > 3)
        n_extreme = jnp.sum(jnp.abs(normal_samples) > 3)
        assert t_extreme > n_extreme

    def test_variance_approx_df_over_df_minus_2(self) -> None:
        """Variance of Student-t(df) ≈ df/(df-2) for df > 2."""
        normal_key = jax.random.PRNGKey(20)
        chi2_key = jax.random.PRNGKey(21)
        df = 10
        samples = student_t_sample(normal_key, chi2_key, df=df, shape=(100000,))
        var = jnp.var(samples)
        expected_var = df / (df - 2)
        assert abs(var - expected_var) < 0.15, (
            f"Var {var} too far from {expected_var}"
        )

    def test_mean_approx_zero(self) -> None:
        """Mean of Student-t samples ≈ 0."""
        normal_key = jax.random.PRNGKey(30)
        chi2_key = jax.random.PRNGKey(31)
        samples = student_t_sample(normal_key, chi2_key, df=5, shape=(100000,))
        mean = jnp.mean(samples)
        assert abs(mean) < 0.05, f"Mean {mean} too far from 0"

    def test_shape_respected(self) -> None:
        """Output shape matches requested shape."""
        normal_key = jax.random.PRNGKey(40)
        chi2_key = jax.random.PRNGKey(41)
        result = student_t_sample(normal_key, chi2_key, df=5, shape=(4, 6))
        assert result.shape == (4, 6)

    def test_different_keys_different_samples(self) -> None:
        """Different keys produce different samples."""
        k1 = jax.random.PRNGKey(50)
        k2 = jax.random.PRNGKey(51)
        k3 = jax.random.PRNGKey(52)
        k4 = jax.random.PRNGKey(53)
        s1 = student_t_sample(k1, k2, df=5, shape=(100,))
        s2 = student_t_sample(k3, k4, df=5, shape=(100,))
        assert not jnp.allclose(s1, s2)
