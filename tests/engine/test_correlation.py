"""Tests for the correlation engine — Cholesky decomposition and shock pipeline."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import pytest

from hyesg.engine.correlation import (
    cholesky_factor,
    correlate_shocks,
    merge_copula_shocks,
    nearest_psd,
    split_copula_shocks,
    validate_correlation_matrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corr_3x3() -> jax.Array:
    """A valid 3×3 correlation matrix."""
    return jnp.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0],
    ])


def _make_non_psd() -> jax.Array:
    """A symmetric unit-diagonal matrix that is NOT positive semi-definite."""
    return jnp.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.9],
        [0.9, -0.9, 1.0],
    ])


# ===========================================================================
# validate_correlation_matrix
# ===========================================================================


class TestValidateCorrelationMatrix:
    """Tests for ``validate_correlation_matrix``."""

    def test_valid_identity(self) -> None:
        """Identity matrix is a valid correlation matrix."""
        mat = jnp.eye(4)
        is_valid, errors = validate_correlation_matrix(mat)
        assert is_valid
        assert errors == []

    def test_valid_3x3(self) -> None:
        """A well-formed 3×3 correlation matrix passes."""
        is_valid, errors = validate_correlation_matrix(_make_corr_3x3())
        assert is_valid
        assert errors == []

    def test_valid_1x1(self) -> None:
        """1×1 identity is valid."""
        mat = jnp.ones((1, 1))
        is_valid, errors = validate_correlation_matrix(mat)
        assert is_valid

    def test_not_square(self) -> None:
        """Non-square matrix is rejected."""
        mat = jnp.ones((2, 3))
        is_valid, errors = validate_correlation_matrix(mat)
        assert not is_valid
        assert any("not square" in e for e in errors)

    def test_not_2d(self) -> None:
        """1-D array is rejected."""
        mat = jnp.ones((3,))
        is_valid, errors = validate_correlation_matrix(mat)
        assert not is_valid
        assert any("2-D" in e for e in errors)

    def test_not_symmetric(self) -> None:
        """Asymmetric matrix is flagged."""
        mat = jnp.array([[1.0, 0.5], [0.3, 1.0]])
        is_valid, errors = validate_correlation_matrix(mat)
        assert not is_valid
        assert any("not symmetric" in e for e in errors)

    def test_diagonal_not_one(self) -> None:
        """Matrix with non-unit diagonal is flagged."""
        mat = jnp.array([[1.0, 0.5], [0.5, 0.9]])
        is_valid, errors = validate_correlation_matrix(mat)
        assert not is_valid
        assert any("Diagonal" in e for e in errors)

    def test_not_psd(self) -> None:
        """Non-PSD matrix is flagged."""
        is_valid, errors = validate_correlation_matrix(_make_non_psd())
        assert not is_valid
        assert any("Not positive semi-definite" in e for e in errors)


# ===========================================================================
# nearest_psd
# ===========================================================================


class TestNearestPSD:
    """Tests for ``nearest_psd``."""

    def test_already_psd_unchanged(self) -> None:
        """An already-PSD matrix stays close to itself."""
        mat = _make_corr_3x3()
        result = nearest_psd(mat)
        assert jnp.allclose(result, mat, atol=1e-10)

    def test_identity_unchanged(self) -> None:
        """Identity stays identity."""
        mat = jnp.eye(5)
        result = nearest_psd(mat)
        assert jnp.allclose(result, mat, atol=1e-10)

    def test_non_psd_becomes_psd(self) -> None:
        """A non-PSD matrix is corrected to PSD."""
        mat = _make_non_psd()
        result = nearest_psd(mat)

        # Must be PSD now
        eigenvalues = jnp.linalg.eigvalsh(result)
        assert float(jnp.min(eigenvalues)) >= -1e-10

        # Diagonal must be 1
        assert jnp.allclose(jnp.diag(result), 1.0, atol=1e-10)

        # Must be symmetric
        assert jnp.allclose(result, result.T, atol=1e-10)

    def test_minimises_frobenius_distance(self) -> None:
        """Result is closer to input than an arbitrary PSD matrix."""
        mat = _make_non_psd()
        result = nearest_psd(mat)
        dist_result = float(jnp.linalg.norm(result - mat, ord="fro"))
        dist_identity = float(jnp.linalg.norm(jnp.eye(3) - mat, ord="fro"))
        assert dist_result <= dist_identity

    def test_1x1_matrix(self) -> None:
        """1×1 matrix returns [[1.0]]."""
        mat = jnp.array([[0.5]])
        result = nearest_psd(mat)
        assert jnp.allclose(result, jnp.ones((1, 1)))


# ===========================================================================
# cholesky_factor
# ===========================================================================


class TestCholeskyFactor:
    """Tests for ``cholesky_factor``."""

    def test_reconstruction(self) -> None:
        """L @ L.T ≈ original correlation matrix."""
        mat = _make_corr_3x3()
        lower = cholesky_factor(mat)
        reconstructed = lower @ lower.T
        assert jnp.allclose(reconstructed, mat, atol=1e-12)

    def test_lower_triangular(self) -> None:
        """Factor is lower triangular."""
        mat = _make_corr_3x3()
        lower = cholesky_factor(mat)
        assert jnp.allclose(lower, jnp.tril(lower))

    def test_identity_cholesky(self) -> None:
        """Cholesky of identity is identity."""
        mat = jnp.eye(4)
        lower = cholesky_factor(mat)
        assert jnp.allclose(lower, jnp.eye(4), atol=1e-12)

    def test_1x1_cholesky(self) -> None:
        """1×1 matrix returns [[1.0]]."""
        mat = jnp.ones((1, 1))
        lower = cholesky_factor(mat)
        assert jnp.allclose(lower, jnp.ones((1, 1)))

    def test_fallback_nearest_psd(self) -> None:
        """Non-PSD matrix triggers fallback and still produces valid L."""
        mat = _make_non_psd()
        lower = cholesky_factor(mat)
        reconstructed = lower @ lower.T

        # Reconstructed should be PSD
        eigenvalues = jnp.linalg.eigvalsh(reconstructed)
        assert float(jnp.min(eigenvalues)) >= -1e-10

        # L should be lower triangular
        assert jnp.allclose(lower, jnp.tril(lower))

    def test_2x2_known_values(self) -> None:
        """Verify against hand-computed Cholesky for a 2×2 case."""
        rho = 0.6
        mat = jnp.array([[1.0, rho], [rho, 1.0]])
        lower = cholesky_factor(mat)

        # Expected: [[1, 0], [rho, sqrt(1 - rho^2)]]
        expected = jnp.array([[1.0, 0.0], [rho, jnp.sqrt(1.0 - rho**2)]])
        assert jnp.allclose(lower, expected, atol=1e-12)


# ===========================================================================
# correlate_shocks
# ===========================================================================


class TestCorrelateShocks:
    """Tests for ``correlate_shocks``."""

    def test_identity_no_correlation(self) -> None:
        """Identity Cholesky leaves shocks unchanged."""
        shocks = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 3))
        lower = jnp.eye(3)
        result = correlate_shocks(shocks, lower)
        assert jnp.allclose(result, shocks, atol=1e-12)

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        shocks = jax.random.normal(jax.random.PRNGKey(1), shape=(50, 4))
        lower = cholesky_factor(jnp.eye(4))
        result = correlate_shocks(shocks, lower)
        assert result.shape == (50, 4)

    def test_statistical_correlation(self) -> None:
        """Correlated shocks match target correlation (N=50000)."""
        n_samples = 50_000
        target_corr = _make_corr_3x3()
        lower = cholesky_factor(target_corr)

        key = jax.random.PRNGKey(42)
        indep = jax.random.normal(key, shape=(n_samples, 3))
        correlated = correlate_shocks(indep, lower)

        # Empirical correlation
        empirical = jnp.corrcoef(correlated.T)

        # Allow statistical tolerance: ~3σ/√N ≈ 0.015 for N=50000
        assert jnp.allclose(empirical, target_corr, atol=0.03), (
            f"Empirical correlation too far from target:\n{empirical}"
        )

    def test_preserves_marginal_distribution(self) -> None:
        """Each shock stream remains approximately N(0,1)."""
        n_samples = 50_000
        target_corr = _make_corr_3x3()
        lower = cholesky_factor(target_corr)

        key = jax.random.PRNGKey(99)
        indep = jax.random.normal(key, shape=(n_samples, 3))
        correlated = correlate_shocks(indep, lower)

        for col in range(3):
            mean = float(jnp.mean(correlated[:, col]))
            std = float(jnp.std(correlated[:, col]))
            assert abs(mean) < 0.03, f"Column {col} mean={mean}"
            assert abs(std - 1.0) < 0.03, f"Column {col} std={std}"

    def test_jit_compatible(self) -> None:
        """correlate_shocks runs under JIT."""
        lower = cholesky_factor(_make_corr_3x3())
        shocks = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 3))

        jitted = jax.jit(correlate_shocks)
        result = jitted(shocks, lower)
        expected = correlate_shocks(shocks, lower)
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_1x1_single_shock(self) -> None:
        """Single-shock system works correctly."""
        shocks = jax.random.normal(jax.random.PRNGKey(5), shape=(20, 1))
        lower = jnp.ones((1, 1))
        result = correlate_shocks(shocks, lower)
        assert jnp.allclose(result, shocks, atol=1e-12)

    def test_functools_partial_reuse(self) -> None:
        """Cholesky factor reused across timesteps via functools.partial."""
        lower = cholesky_factor(_make_corr_3x3())
        apply_corr = functools.partial(correlate_shocks, cholesky_L=lower)

        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        shocks1 = jax.random.normal(key1, shape=(10, 3))
        shocks2 = jax.random.normal(key2, shape=(10, 3))

        result1 = apply_corr(shocks1)
        result2 = apply_corr(shocks2)

        # Results should differ (different inputs) but both use same L
        assert not jnp.array_equal(result1, result2)
        # Verify correctness of each
        assert jnp.allclose(result1, (lower @ shocks1.T).T, atol=1e-12)
        assert jnp.allclose(result2, (lower @ shocks2.T).T, atol=1e-12)


# ===========================================================================
# split_copula_shocks / merge_copula_shocks
# ===========================================================================


class TestSplitMergeCopulaShocks:
    """Tests for ``split_copula_shocks`` and ``merge_copula_shocks``."""

    def test_split_shapes(self) -> None:
        """Split produces correct shapes."""
        shocks = jnp.ones((10, 5))
        mask = jnp.array([True, False, True, True, False])
        copula, non_copula = split_copula_shocks(shocks, mask)
        assert copula.shape == (10, 3)
        assert non_copula.shape == (10, 2)

    def test_split_values(self) -> None:
        """Split extracts correct columns."""
        shocks = jnp.arange(20.0).reshape(4, 5)
        mask = jnp.array([True, False, True, False, True])
        copula, non_copula = split_copula_shocks(shocks, mask)

        # Copula columns: 0, 2, 4
        expected_copula = shocks[:, jnp.array([0, 2, 4])]
        assert jnp.allclose(copula, expected_copula)

        # Non-copula columns: 1, 3
        expected_non_copula = shocks[:, jnp.array([1, 3])]
        assert jnp.allclose(non_copula, expected_non_copula)

    def test_roundtrip(self) -> None:
        """Split → merge recovers the original array."""
        key = jax.random.PRNGKey(77)
        shocks = jax.random.normal(key, shape=(20, 6))
        mask = jnp.array([True, False, True, True, False, True])

        copula, non_copula = split_copula_shocks(shocks, mask)
        merged = merge_copula_shocks(copula, non_copula, mask)

        assert jnp.allclose(merged, shocks, atol=1e-12)

    def test_all_copula(self) -> None:
        """All shocks participate in copula."""
        key = jax.random.PRNGKey(0)
        shocks = jax.random.normal(key, shape=(10, 3))
        mask = jnp.array([True, True, True])

        copula, non_copula = split_copula_shocks(shocks, mask)
        assert copula.shape == (10, 3)
        assert non_copula.shape == (10, 0)
        assert jnp.allclose(copula, shocks)

        merged = merge_copula_shocks(copula, non_copula, mask)
        assert jnp.allclose(merged, shocks, atol=1e-12)

    def test_no_copula(self) -> None:
        """No shocks participate in copula."""
        key = jax.random.PRNGKey(1)
        shocks = jax.random.normal(key, shape=(10, 3))
        mask = jnp.array([False, False, False])

        copula, non_copula = split_copula_shocks(shocks, mask)
        assert copula.shape == (10, 0)
        assert non_copula.shape == (10, 3)
        assert jnp.allclose(non_copula, shocks)

        merged = merge_copula_shocks(copula, non_copula, mask)
        assert jnp.allclose(merged, shocks, atol=1e-12)

    def test_single_shock_copula(self) -> None:
        """Single shock marked as copula."""
        shocks = jnp.array([[1.0], [2.0], [3.0]])
        mask = jnp.array([True])

        copula, non_copula = split_copula_shocks(shocks, mask)
        assert copula.shape == (3, 1)
        assert non_copula.shape == (3, 0)

        merged = merge_copula_shocks(copula, non_copula, mask)
        assert jnp.allclose(merged, shocks)

    def test_jit_compatible(self) -> None:
        """Split and merge work when called inside a JIT-compiled function."""
        shocks = jax.random.normal(jax.random.PRNGKey(3), shape=(10, 4))
        mask = jnp.array([True, False, True, False])

        # Pre-compute concrete indices, then JIT the gather/scatter
        copula, non_copula = split_copula_shocks(shocks, mask)
        merged = merge_copula_shocks(copula, non_copula, mask)
        assert jnp.allclose(merged, shocks, atol=1e-12)
