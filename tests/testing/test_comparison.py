"""Tests for statistical comparison functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyesg.testing.comparison import (
    compare_distributions,
    compare_exact,
    compare_moments,
    compare_quantiles,
)

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_sample(mean: float, std: float, n: int, seed: int) -> jax.Array:
    """Generate a normal sample as a JAX array."""
    rng = np.random.default_rng(seed)
    return jnp.array(rng.normal(mean, std, n))


def _uniform_sample(low: float, high: float, n: int, seed: int) -> jax.Array:
    """Generate a uniform sample as a JAX array."""
    rng = np.random.default_rng(seed)
    return jnp.array(rng.uniform(low, high, n))


# ---------------------------------------------------------------------------
# compare_moments
# ---------------------------------------------------------------------------


class TestCompareMoments:
    """Tests for compare_moments."""

    def test_identical_arrays_pass(self):
        """Identical arrays produce zero error and pass."""
        arr = _normal_sample(0.0, 1.0, 5000, seed=10)
        result = compare_moments(arr, arr, rtol=1e-6)
        assert result.passed
        assert result.metric == pytest.approx(0.0, abs=1e-15)

    def test_same_distribution_passes(self):
        """Samples from the same distribution pass with reasonable rtol."""
        a = _normal_sample(0.0, 1.0, 10000, seed=10)
        b = _normal_sample(0.0, 1.0, 10000, seed=11)
        result = compare_moments(a, b, rtol=0.1)
        assert result.passed

    def test_different_mean_fails(self):
        """Significantly shifted mean is detected."""
        a = _normal_sample(0.0, 1.0, 5000, seed=10)
        b = _normal_sample(5.0, 1.0, 5000, seed=11)
        result = compare_moments(a, b, rtol=0.01)
        assert not result.passed

    def test_different_variance_fails(self):
        """Different variance is detected."""
        a = _normal_sample(0.0, 1.0, 5000, seed=10)
        b = _normal_sample(0.0, 5.0, 5000, seed=11)
        result = compare_moments(a, b, rtol=0.1)
        assert not result.passed

    def test_details_contain_moments(self):
        """Result details include all four moments."""
        a = _normal_sample(0.0, 1.0, 1000, seed=10)
        result = compare_moments(a, a)
        assert "actual_moments" in result.details
        assert "expected_moments" in result.details
        assert "relative_errors" in result.details
        assert len(result.details["actual_moments"]) == 4

    def test_test_name(self):
        """Result has correct test name."""
        a = _normal_sample(0.0, 1.0, 100, seed=10)
        result = compare_moments(a, a)
        assert result.test_name == "moment_comparison"

    def test_2d_arrays_flattened(self):
        """2D arrays are flattened before comparison."""
        rng = np.random.default_rng(10)
        a = jnp.array(rng.normal(0.0, 1.0, (50, 100)))
        result = compare_moments(a, a, rtol=1e-6)
        assert result.passed


# ---------------------------------------------------------------------------
# compare_distributions
# ---------------------------------------------------------------------------


class TestCompareDistributions:
    """Tests for compare_distributions (KS and AD tests)."""

    def test_same_sample_passes_ks(self):
        """Identical samples pass KS test."""
        arr = _normal_sample(0.0, 1.0, 1000, seed=20)
        result = compare_distributions(arr, arr, test="ks")
        assert result.passed
        assert result.metric >= 0.01  # p-value should be high

    def test_same_distribution_passes_ks(self):
        """Two samples from the same distribution pass KS test."""
        a = _normal_sample(0.0, 1.0, 5000, seed=20)
        b = _normal_sample(0.0, 1.0, 5000, seed=21)
        result = compare_distributions(a, b, test="ks", significance=0.01)
        assert result.passed

    def test_different_distributions_fail_ks(self):
        """Normal vs uniform fails KS test."""
        a = _normal_sample(0.0, 1.0, 2000, seed=20)
        b = _uniform_sample(-3.0, 3.0, 2000, seed=21)
        result = compare_distributions(a, b, test="ks", significance=0.01)
        assert not result.passed

    def test_shifted_distribution_fails_ks(self):
        """Shifted mean is detected by KS test."""
        a = _normal_sample(0.0, 1.0, 2000, seed=20)
        b = _normal_sample(2.0, 1.0, 2000, seed=21)
        result = compare_distributions(a, b, test="ks")
        assert not result.passed

    def test_ks_details_contain_statistic(self):
        """KS result details contain the test statistic."""
        a = _normal_sample(0.0, 1.0, 500, seed=20)
        result = compare_distributions(a, a, test="ks")
        assert "statistic" in result.details
        assert "p_value" in result.details

    def test_anderson_darling_same_passes(self):
        """Same sample passes Anderson-Darling test."""
        arr = _normal_sample(0.0, 1.0, 1000, seed=30)
        result = compare_distributions(arr, arr, test="ad")
        assert result.passed

    def test_anderson_darling_different_fails(self):
        """Different distributions fail Anderson-Darling test."""
        a = _normal_sample(0.0, 1.0, 2000, seed=30)
        b = _normal_sample(3.0, 1.0, 2000, seed=31)
        result = compare_distributions(a, b, test="ad", significance=0.01)
        assert not result.passed

    def test_invalid_test_raises(self):
        """Unknown test name raises ValueError."""
        a = _normal_sample(0.0, 1.0, 100, seed=20)
        with pytest.raises(ValueError, match="Unknown test"):
            compare_distributions(a, a, test="chi2")

    def test_ks_test_name(self):
        """KS result has correct test name."""
        a = _normal_sample(0.0, 1.0, 100, seed=20)
        result = compare_distributions(a, a, test="ks")
        assert result.test_name == "ks_test"

    def test_ad_test_name(self):
        """AD result has correct test name."""
        a = _normal_sample(0.0, 1.0, 100, seed=20)
        result = compare_distributions(a, a, test="ad")
        assert result.test_name == "anderson_darling_test"


# ---------------------------------------------------------------------------
# compare_quantiles
# ---------------------------------------------------------------------------


class TestCompareQuantiles:
    """Tests for compare_quantiles."""

    def test_identical_arrays_pass(self):
        """Identical arrays pass quantile comparison."""
        arr = _normal_sample(0.0, 1.0, 5000, seed=40)
        result = compare_quantiles(arr, arr, rtol=1e-6)
        assert result.passed

    def test_same_distribution_passes(self):
        """Same-distribution samples pass with reasonable rtol."""
        a = _normal_sample(0.0, 1.0, 10000, seed=40)
        b = _normal_sample(0.0, 1.0, 10000, seed=41)
        result = compare_quantiles(a, b, rtol=0.1)
        assert result.passed

    def test_shifted_distribution_fails(self):
        """Shifted distribution fails quantile comparison."""
        a = _normal_sample(0.0, 1.0, 5000, seed=40)
        b = _normal_sample(5.0, 1.0, 5000, seed=41)
        result = compare_quantiles(a, b, rtol=0.01)
        assert not result.passed

    def test_custom_quantiles(self):
        """Custom quantile list is used."""
        a = _normal_sample(0.0, 1.0, 1000, seed=40)
        quantiles = [0.1, 0.5, 0.9]
        result = compare_quantiles(a, a, quantiles=quantiles)
        assert result.passed
        assert result.details["quantiles"] == quantiles

    def test_default_quantiles(self):
        """Default quantile list has 7 entries."""
        a = _normal_sample(0.0, 1.0, 1000, seed=40)
        result = compare_quantiles(a, a)
        assert len(result.details["quantiles"]) == 7

    def test_details_contain_quantile_values(self):
        """Details include actual and expected quantiles."""
        a = _normal_sample(0.0, 1.0, 1000, seed=40)
        result = compare_quantiles(a, a)
        assert "actual_quantiles" in result.details
        assert "expected_quantiles" in result.details
        assert "relative_errors" in result.details

    def test_test_name(self):
        """Quantile comparison has correct test name."""
        a = _normal_sample(0.0, 1.0, 100, seed=40)
        result = compare_quantiles(a, a)
        assert result.test_name == "quantile_comparison"


# ---------------------------------------------------------------------------
# compare_exact
# ---------------------------------------------------------------------------


class TestCompareExact:
    """Tests for compare_exact."""

    def test_identical_arrays_pass(self):
        """Identical arrays pass with any tolerance."""
        arr = jnp.array([1.0, 2.0, 3.0])
        result = compare_exact(arr, arr, atol=0.0)
        assert result.passed
        assert result.metric == 0.0

    def test_within_tolerance_passes(self):
        """Arrays within atol pass."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0 + 1e-13, 2.0 - 1e-13, 3.0])
        result = compare_exact(a, b, atol=1e-12)
        assert result.passed

    def test_outside_tolerance_fails(self):
        """Arrays outside atol fail."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.1, 2.0, 3.0])
        result = compare_exact(a, b, atol=1e-3)
        assert not result.passed

    def test_metric_is_max_diff(self):
        """Metric equals the maximum absolute difference."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.01, 2.02, 3.03])
        result = compare_exact(a, b, atol=1.0)
        assert result.metric == pytest.approx(0.03, abs=1e-10)

    def test_details_contain_counts(self):
        """Details include failing element counts."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.1, 2.0, 3.1])
        result = compare_exact(a, b, atol=0.05)
        assert result.details["n_failing_elements"] == 2
        assert result.details["n_total_elements"] == 3

    def test_zero_tolerance_exact_match(self):
        """Zero tolerance requires bit-exact match."""
        a = jnp.array([1.0, 2.0])
        result = compare_exact(a, a, atol=0.0)
        assert result.passed
        assert result.metric == 0.0

    def test_2d_arrays(self):
        """Works with multi-dimensional arrays."""
        a = jnp.ones((5, 10))
        b = jnp.ones((5, 10)) + 1e-14
        result = compare_exact(a, b, atol=1e-12)
        assert result.passed

    def test_test_name(self):
        """Exact comparison has correct test name."""
        a = jnp.array([1.0])
        result = compare_exact(a, a)
        assert result.test_name == "exact_comparison"
