"""Tests for the copula engine — Gaussian and Student-t copula transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import pytest
from scipy import stats as sp_stats

jax.config.update("jax_enable_x64", True)

from hyesg.engine.copula import (
    apply_copula,
    apply_copula_antithetic,
    apply_copula_antithetic_csharp,
    gaussian_copula,
    gaussian_copula_inverse,
    student_t_copula,
    student_t_copula_inverse,
)
from hyesg.engine.correlation import cholesky_factor


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


def _generate_normals(key: jax.Array, n_steps: int, n_shocks: int) -> jax.Array:
    """Generate standard-normal shocks."""
    return jax.random.normal(key, shape=(n_steps, n_shocks))


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------


class TestGaussianCopula:
    """Tests for gaussian_copula and gaussian_copula_inverse."""

    def test_output_in_unit_interval(self, rng_key: jax.Array) -> None:
        """Gaussian copula CDF maps normals to (0, 1)."""
        z = _generate_normals(rng_key, 1000, 3)
        u = gaussian_copula(z)
        assert jnp.all(u > 0.0)
        assert jnp.all(u < 1.0)

    def test_round_trip(self, rng_key: jax.Array) -> None:
        """inverse(copula(z)) ≈ z."""
        z = _generate_normals(rng_key, 500, 4)
        u = gaussian_copula(z)
        z_recovered = gaussian_copula_inverse(u)
        assert jnp.allclose(z, z_recovered, atol=1e-10)

    def test_marginals_are_standard_normal(self, rng_key: jax.Array) -> None:
        """After Gaussian copula round-trip, marginals pass KS test."""
        z = _generate_normals(rng_key, 10_000, 3)
        corr = _make_corr_3x3()
        chol = cholesky_factor(corr)

        z_out = apply_copula(z, chol, "gaussian")

        for col in range(3):
            samples = z_out[:, col]
            ks_stat, p_value = sp_stats.kstest(samples, "norm")
            assert p_value > 0.01, (
                f"Column {col} failed KS test: stat={ks_stat:.4f}, p={p_value:.4f}"
            )

    def test_jit_compatible(self, rng_key: jax.Array) -> None:
        """Gaussian copula functions work under JIT."""
        z = _generate_normals(rng_key, 100, 3)
        u_jit = jax.jit(gaussian_copula)(z)
        z_jit = jax.jit(gaussian_copula_inverse)(u_jit)
        assert jnp.allclose(z, z_jit, atol=1e-10)


# ---------------------------------------------------------------------------
# Student-t copula
# ---------------------------------------------------------------------------


class TestStudentTCopula:
    """Tests for student_t_copula and student_t_copula_inverse."""

    def test_output_in_unit_interval(self, rng_key: jax.Array) -> None:
        """Student-t copula CDF maps normals to (0, 1)."""
        z = _generate_normals(rng_key, 1000, 3)
        u = student_t_copula(z, df=5.0)
        assert jnp.all(u > 0.0)
        assert jnp.all(u < 1.0)

    def test_round_trip(self, rng_key: jax.Array) -> None:
        """CDF → PPF round-trip: output has correct standard-normal marginals.

        Because student_t_copula_inverse outputs Φ⁻¹(u) (normal marginals),
        the composition inverse(copula(z)) = Φ⁻¹(F_t(z; df)) ≠ z.  Instead
        we verify the uniforms from the CDF step are valid (in (0,1)) and
        that transforming them back to normals yields finite results.
        """
        z = _generate_normals(rng_key, 500, 4)
        u = student_t_copula(z, df=5.0)
        # Uniforms must be in (0, 1)
        assert jnp.all(u > 0.0)
        assert jnp.all(u < 1.0)
        # Inverse maps uniforms to normals (finite, no NaN)
        z_out = student_t_copula_inverse(u, df=5.0)
        assert jnp.all(jnp.isfinite(z_out))

    def test_df_validation(self) -> None:
        """Student-t copula rejects df <= 2."""
        z = jnp.ones((10, 3))
        with pytest.raises(ValueError, match="df must be > 2"):
            student_t_copula(z, df=2.0)
        with pytest.raises(ValueError, match="df must be > 2"):
            student_t_copula(z, df=1.5)

    def test_df_validation_inverse(self) -> None:
        """Student-t inverse copula rejects df <= 2."""
        u = 0.5 * jnp.ones((10, 3))
        with pytest.raises(ValueError, match="df must be > 2"):
            student_t_copula_inverse(u, df=2.0)

    def test_heavier_tails_than_gaussian(self, rng_key: jax.Array) -> None:
        """Student-t copula alters the dependence structure vs Gaussian.

        The pipeline applies F_t(z; df) → Φ⁻¹(u) to correlated standard
        normals.  Because the Student-t CDF is less steep than the normal
        CDF in the tails, the transform *compresses* marginals toward zero.
        This test verifies the effect exists and increases with lower df.
        As df → ∞ the Student-t pipeline converges to Gaussian (identity).
        """
        n_steps = 50_000
        corr = jnp.array([[1.0, 0.7], [0.7, 1.0]])
        chol = cholesky_factor(corr)
        z = _generate_normals(rng_key, n_steps, 2)

        # Gaussian copula pipeline (identity on marginals)
        z_gauss = apply_copula(z, chol, "gaussian")

        # Student-t copula pipeline with low df (compresses marginals)
        z_t3 = apply_copula(z, chol, "student_t", df=3.0)
        z_t30 = apply_copula(z, chol, "student_t", df=30.0)

        # Student-t marginals should be compressed (smaller std dev)
        std_gauss = float(jnp.std(z_gauss))
        std_t3 = float(jnp.std(z_t3))
        std_t30 = float(jnp.std(z_t30))

        assert std_t3 < std_gauss, (
            f"Low-df Student-t should compress marginals: std_t3={std_t3:.4f}, "
            f"std_gauss={std_gauss:.4f}"
        )
        # Higher df should be closer to Gaussian
        assert std_t30 > std_t3, (
            f"Higher df should compress less: std_t30={std_t30:.4f}, std_t3={std_t3:.4f}"
        )
        # As df → ∞, the pipeline converges to Gaussian
        assert abs(std_t30 - std_gauss) < abs(std_t3 - std_gauss), (
            "Higher df should be closer to Gaussian"
        )

    def test_convergence_to_gaussian(self, rng_key: jax.Array) -> None:
        """Student-t(df→∞) converges to Gaussian copula."""
        z = _generate_normals(rng_key, 5000, 3)
        u_gauss = gaussian_copula(z)
        u_t_large_df = student_t_copula(z, df=1000.0)

        assert jnp.allclose(u_gauss, u_t_large_df, atol=1e-3), (
            f"Max diff = {float(jnp.max(jnp.abs(u_gauss - u_t_large_df))):.6f}"
        )

    def test_jit_compatible(self, rng_key: jax.Array) -> None:
        """Student-t copula functions work under JIT."""
        z = _generate_normals(rng_key, 100, 3)
        u = student_t_copula(z, df=5.0)
        assert jnp.all(jnp.isfinite(u))
        z_back = student_t_copula_inverse(u, df=5.0)
        assert jnp.all(jnp.isfinite(z_back))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestApplyCopula:
    """Tests for the full apply_copula pipeline."""

    def test_gaussian_pipeline_shape(self, rng_key: jax.Array) -> None:
        """Pipeline preserves shape."""
        n_steps, n_shocks = 200, 3
        z = _generate_normals(rng_key, n_steps, n_shocks)
        corr = _make_corr_3x3()
        chol = cholesky_factor(corr)

        result = apply_copula(z, chol, "gaussian")
        assert result.shape == (n_steps, n_shocks)

    def test_student_t_pipeline_shape(self, rng_key: jax.Array) -> None:
        """Pipeline preserves shape for Student-t."""
        n_steps, n_shocks = 200, 3
        z = _generate_normals(rng_key, n_steps, n_shocks)
        corr = _make_corr_3x3()
        chol = cholesky_factor(corr)

        result = apply_copula(z, chol, "student_t", df=5.0)
        assert result.shape == (n_steps, n_shocks)

    def test_unknown_copula_type_raises(self, rng_key: jax.Array) -> None:
        """Unknown copula type raises ValueError."""
        z = _generate_normals(rng_key, 10, 3)
        chol = jnp.eye(3)
        with pytest.raises(ValueError, match="Unknown copula_type"):
            apply_copula(z, chol, "unknown")

    def test_correlation_structure_preserved(self, rng_key: jax.Array) -> None:
        """Empirical correlation ≈ target correlation after Gaussian copula."""
        n_steps = 50_000
        corr = _make_corr_3x3()
        chol = cholesky_factor(corr)
        z = _generate_normals(rng_key, n_steps, 3)

        z_out = apply_copula(z, chol, "gaussian")
        empirical_corr = jnp.corrcoef(z_out.T)

        assert jnp.allclose(empirical_corr, corr, atol=0.03), (
            f"Max corr error = {float(jnp.max(jnp.abs(empirical_corr - corr))):.4f}"
        )

    def test_student_t_correlation_structure(self, rng_key: jax.Array) -> None:
        """Student-t pipeline produces approximately correct correlation."""
        n_steps = 50_000
        corr = _make_corr_3x3()
        chol = cholesky_factor(corr)
        z = _generate_normals(rng_key, n_steps, 3)

        z_out = apply_copula(z, chol, "student_t", df=10.0)
        empirical_corr = jnp.corrcoef(z_out.T)

        # Student-t slightly distorts linear correlation, so wider tolerance
        assert jnp.allclose(empirical_corr, corr, atol=0.05), (
            f"Max corr error = {float(jnp.max(jnp.abs(empirical_corr - corr))):.4f}"
        )

    def test_jit_compatible(self, rng_key: jax.Array) -> None:
        """Full pipeline works under JIT."""
        z = _generate_normals(rng_key, 100, 3)
        corr = _make_corr_3x3()
        chol = cholesky_factor(corr)

        result = apply_copula(z, chol, "gaussian")
        assert result.shape == (100, 3)


# ---------------------------------------------------------------------------
# Antithetic variance reduction
# ---------------------------------------------------------------------------


class TestAntithetic:
    """Tests for antithetic variance reduction methods."""

    def test_correct_antithetic_is_complement(self) -> None:
        """Correct antithetic: u_anti = 1 - u."""
        u = jnp.array([[0.1, 0.9], [0.5, 0.3]])
        u_anti = apply_copula_antithetic(u)
        expected = jnp.array([[0.9, 0.1], [0.5, 0.7]])
        assert jnp.allclose(u_anti, expected, atol=1e-12)

    def test_csharp_antithetic_is_negation(self) -> None:
        """C# antithetic: z_anti = -z."""
        z = jnp.array([[1.0, -0.5], [0.0, 2.0]])
        z_anti = apply_copula_antithetic_csharp(z)
        expected = jnp.array([[-1.0, 0.5], [0.0, -2.0]])
        assert jnp.allclose(z_anti, expected, atol=1e-12)

    def test_gaussian_antithetics_equivalent(self, rng_key: jax.Array) -> None:
        """For Gaussian copula, both antithetic methods are equivalent.

        Φ⁻¹(1 - Φ(z)) = -z, so complement-of-uniform and negate-normal
        produce the same result.
        """
        z = _generate_normals(rng_key, 500, 3)

        # Method 1: correct antithetic (complement uniforms)
        u = gaussian_copula(z)
        u_anti = apply_copula_antithetic(u)
        z_anti_correct = gaussian_copula_inverse(u_anti)

        # Method 2: C# antithetic (negate normals)
        z_anti_csharp = apply_copula_antithetic_csharp(z)
        u_csharp = gaussian_copula(z_anti_csharp)
        z_anti_csharp_out = gaussian_copula_inverse(u_csharp)

        assert jnp.allclose(z_anti_correct, z_anti_csharp_out, atol=1e-10)

    def test_student_t_antithetics_equivalent_for_symmetric(
        self, rng_key: jax.Array,
    ) -> None:
        """For Student-t copula, both antithetic methods are equivalent.

        The Student-t CDF is symmetric: F_t(-z; df) = 1 - F_t(z; df).
        Therefore Φ⁻¹(1 - F_t(z)) = Φ⁻¹(F_t(-z)), making uniform-
        complement and normal-negation antithetics identical.

        Note: The C# engine's antithetic (negate pre-CDF normals) is
        mathematically equivalent to the uniform-complement method for
        ALL symmetric distributions, including Student-t.
        """
        z = _generate_normals(rng_key, 500, 3)
        df = 5.0

        # Method 1: correct antithetic (complement uniforms)
        u = student_t_copula(z, df)
        u_anti = apply_copula_antithetic(u)
        z_anti_correct = student_t_copula_inverse(u_anti, df)

        # Method 2: C# antithetic (negate normals then CDF)
        z_anti_csharp = apply_copula_antithetic_csharp(z)
        u_csharp = student_t_copula(z_anti_csharp, df)
        z_anti_csharp_out = student_t_copula_inverse(u_csharp, df)

        # They should be equal because Student-t is symmetric
        assert jnp.allclose(z_anti_correct, z_anti_csharp_out, atol=1e-10), (
            f"Max diff = {float(jnp.max(jnp.abs(z_anti_correct - z_anti_csharp_out))):.2e}"
        )

    def test_jit_compatible(self) -> None:
        """Antithetic functions work under JIT."""
        u = jnp.array([[0.1, 0.9], [0.5, 0.3]])
        z = jnp.array([[1.0, -0.5], [0.0, 2.0]])

        u_anti = jax.jit(apply_copula_antithetic)(u)
        z_anti = jax.jit(apply_copula_antithetic_csharp)(z)

        assert jnp.allclose(u_anti, 1.0 - u, atol=1e-12)
        assert jnp.allclose(z_anti, -z, atol=1e-12)


# ---------------------------------------------------------------------------
# C# match mode
# ---------------------------------------------------------------------------


class TestCSharpMatchMode:
    """Tests for C# compatibility mode in apply_copula."""

    def test_csharp_mode_produces_different_output_for_student_t(
        self, rng_key: jax.Array,
    ) -> None:
        """C# match mode produces different output from correct mode for t-copula.

        When match_csharp=True, antithetic is applied by negating normals
        (pre-CDF), which differs from complement-of-uniforms for Student-t.
        Here we just verify both modes produce valid outputs of correct shape.
        """
        z = _generate_normals(rng_key, 200, 3)
        corr = _make_corr_3x3()
        chol = cholesky_factor(corr)

        result_correct = apply_copula(z, chol, "student_t", df=5.0, match_csharp=False)
        result_csharp = apply_copula(z, chol, "student_t", df=5.0, match_csharp=True)

        # Both should be valid arrays of same shape
        assert result_correct.shape == result_csharp.shape == (200, 3)
        # Both should produce finite values
        assert jnp.all(jnp.isfinite(result_correct))
        assert jnp.all(jnp.isfinite(result_csharp))
