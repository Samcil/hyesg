"""Tests for correlation matrix repair — hyperspherical parameterisation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.core.matrix import SymmetricLabelledMatrix
from hyesg.engine.correlation import (
    _angles_to_correlation,
    _correlation_residuals,
    nearest_psd,
    repair_correlation_hyperspherical,
    validate_and_repair,
    validate_correlation_matrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_non_psd_3x3() -> jax.Array:
    """A known non-PSD 3×3 matrix."""
    return jnp.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.9],
        [0.9, -0.9, 1.0],
    ])


def _make_valid_5x5() -> jax.Array:
    """A valid 5×5 correlation matrix built from Cholesky."""
    key = jax.random.PRNGKey(123)
    L = jnp.tril(jax.random.normal(key, (5, 5)))
    C = L @ L.T
    # Normalise to correlation
    d = jnp.sqrt(jnp.diag(C))
    C = C / jnp.outer(d, d)
    return C


def _make_realistic_10x10() -> jax.Array:
    """A realistic 10×10 financial correlation matrix (non-PSD).

    Constructed by perturbing a valid matrix to break PSD property.
    """
    key = jax.random.PRNGKey(42)
    L = jnp.tril(jax.random.normal(key, (10, 10)))
    C = L @ L.T
    d = jnp.sqrt(jnp.diag(C))
    C = C / jnp.outer(d, d)

    # Perturb to break PSD
    perturbation = jax.random.normal(jax.random.PRNGKey(99), (10, 10)) * 0.3
    perturbation = 0.5 * (perturbation + perturbation.T)
    C_bad = C + perturbation
    # Force unit diagonal
    C_bad = C_bad.at[jnp.diag_indices(10)].set(1.0)
    # Clamp off-diag to [-1, 1]
    C_bad = jnp.clip(C_bad, -1.0, 1.0)
    C_bad = C_bad.at[jnp.diag_indices(10)].set(1.0)
    return C_bad


# ===========================================================================
# _angles_to_correlation
# ===========================================================================


class TestAnglesToCorrelation:
    """Tests for ``_angles_to_correlation``."""

    def test_pi_half_produces_identity(self) -> None:
        """All angles = pi/2 gives the identity matrix."""
        n = 4
        n_angles = n * (n - 1) // 2
        angles = jnp.full(n_angles, jnp.pi / 2)
        C = _angles_to_correlation(angles, n)
        assert jnp.allclose(C, jnp.eye(n), atol=1e-12)

    def test_result_is_psd(self) -> None:
        """Result is always positive semi-definite for random angles."""
        key = jax.random.PRNGKey(7)
        n = 5
        n_angles = n * (n - 1) // 2
        angles = jax.random.uniform(key, (n_angles,), minval=0.1, maxval=jnp.pi - 0.1)
        C = _angles_to_correlation(angles, n)
        eigenvalues = jnp.linalg.eigvalsh(C)
        assert float(jnp.min(eigenvalues)) >= -1e-12

    def test_unit_diagonal(self) -> None:
        """Diagonal entries are always 1."""
        key = jax.random.PRNGKey(11)
        n = 6
        n_angles = n * (n - 1) // 2
        angles = jax.random.uniform(key, (n_angles,), minval=0.0, maxval=jnp.pi)
        C = _angles_to_correlation(angles, n)
        assert jnp.allclose(jnp.diag(C), 1.0, atol=1e-12)

    def test_symmetric(self) -> None:
        """Result is symmetric."""
        key = jax.random.PRNGKey(13)
        n = 4
        n_angles = n * (n - 1) // 2
        angles = jax.random.uniform(key, (n_angles,), minval=0.0, maxval=jnp.pi)
        C = _angles_to_correlation(angles, n)
        assert jnp.allclose(C, C.T, atol=1e-14)

    def test_1x1(self) -> None:
        """1×1 returns [[1.0]]."""
        angles = jnp.array([], dtype=jnp.float64)
        C = _angles_to_correlation(angles, 1)
        assert C.shape == (1, 1)
        assert jnp.allclose(C, jnp.ones((1, 1)))

    def test_2x2_known_value(self) -> None:
        """2×2 with one angle = theta gives rho = cos(theta)."""
        theta = jnp.array([jnp.pi / 3])
        C = _angles_to_correlation(theta, 2)
        rho_expected = jnp.cos(jnp.pi / 3)
        assert C.shape == (2, 2)
        assert jnp.allclose(C[0, 1], rho_expected, atol=1e-12)
        assert jnp.allclose(C[1, 0], rho_expected, atol=1e-12)

    def test_3x3_identity_from_pi_half(self) -> None:
        """3×3 with all pi/2 angles gives identity."""
        angles = jnp.full(3, jnp.pi / 2)
        C = _angles_to_correlation(angles, 3)
        assert jnp.allclose(C, jnp.eye(3), atol=1e-12)


# ===========================================================================
# repair_correlation_hyperspherical
# ===========================================================================


class TestRepairCorrelationHyperspherical:
    """Tests for ``repair_correlation_hyperspherical``."""

    def test_repairs_non_psd(self) -> None:
        """Non-PSD matrix is repaired to a valid PSD matrix."""
        target = _make_non_psd_3x3()
        repaired = repair_correlation_hyperspherical(target)

        # Must be PSD
        eigenvalues = jnp.linalg.eigvalsh(repaired)
        assert float(jnp.min(eigenvalues)) >= -1e-10

        # Unit diagonal
        assert jnp.allclose(jnp.diag(repaired), 1.0, atol=1e-10)

        # Symmetric
        assert jnp.allclose(repaired, repaired.T, atol=1e-12)

    def test_already_psd_close_to_input(self) -> None:
        """Already-PSD matrix stays close to itself."""
        target = _make_valid_5x5()
        repaired = repair_correlation_hyperspherical(target, max_iterations=100)

        dist = float(jnp.linalg.norm(repaired - target, ord="fro"))
        assert dist < 0.5, f"Repaired too far from valid input: {dist}"

    def test_1x1_edge_case(self) -> None:
        """1×1 matrix returns [[1.0]]."""
        target = jnp.array([[0.5]])
        repaired = repair_correlation_hyperspherical(target)
        assert jnp.allclose(repaired, jnp.ones((1, 1)))

    def test_identity_stays_identity(self) -> None:
        """Identity matrix stays close to identity."""
        n = 4
        target = jnp.eye(n)
        repaired = repair_correlation_hyperspherical(target)
        assert jnp.allclose(repaired, jnp.eye(n), atol=1e-4)

    def test_10x10_realistic(self) -> None:
        """10×10 non-PSD financial matrix is repaired."""
        target = _make_realistic_10x10()
        repaired = repair_correlation_hyperspherical(target, max_iterations=100)

        # Must be PSD
        eigenvalues = jnp.linalg.eigvalsh(repaired)
        assert float(jnp.min(eigenvalues)) >= -1e-10

        # Unit diagonal
        assert jnp.allclose(jnp.diag(repaired), 1.0, atol=1e-8)

        # Symmetric
        assert jnp.allclose(repaired, repaired.T, atol=1e-12)

    def test_2x2_perfect_correlation(self) -> None:
        """2×2 with rho=1 (singular but PSD)."""
        target = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        repaired = repair_correlation_hyperspherical(target)
        eigenvalues = jnp.linalg.eigvalsh(repaired)
        assert float(jnp.min(eigenvalues)) >= -1e-10


# ===========================================================================
# validate_and_repair
# ===========================================================================


class TestValidateAndRepair:
    """Tests for ``validate_and_repair``."""

    def test_psd_no_op(self) -> None:
        """PSD matrix passes through unchanged."""
        mat = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        result = validate_and_repair(mat)
        assert jnp.allclose(result, mat, atol=1e-12)

    def test_psd_with_labels(self) -> None:
        """PSD matrix with labels returns SymmetricLabelledMatrix."""
        mat = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        result = validate_and_repair(mat, labels=["a", "b"])
        assert isinstance(result, SymmetricLabelledMatrix)
        assert result.row_labels == ("a", "b")
        assert result["a", "b"] == pytest.approx(0.5)

    def test_non_psd_repairs_higham(self) -> None:
        """Non-PSD matrix repaired via Higham method."""
        target = _make_non_psd_3x3()
        result = validate_and_repair(target, method="higham")
        is_valid, _ = validate_correlation_matrix(result)
        assert is_valid

    def test_non_psd_repairs_hyperspherical(self) -> None:
        """Non-PSD matrix repaired via hyperspherical method."""
        target = _make_non_psd_3x3()
        result = validate_and_repair(target, method="hyperspherical")
        eigenvalues = jnp.linalg.eigvalsh(result)
        assert float(jnp.min(eigenvalues)) >= -1e-10
        assert jnp.allclose(jnp.diag(result), 1.0, atol=1e-8)

    def test_non_psd_with_labels_returns_labelled(self) -> None:
        """Non-PSD with labels returns SymmetricLabelledMatrix after repair."""
        target = _make_non_psd_3x3()
        labels = ["UK_Equity", "US_Equity", "Gilts"]
        result = validate_and_repair(target, labels=labels, method="higham")
        assert isinstance(result, SymmetricLabelledMatrix)
        assert result.row_labels == tuple(labels)

    def test_identity_no_op(self) -> None:
        """Identity matrix passes through unchanged."""
        mat = jnp.eye(5)
        result = validate_and_repair(mat)
        assert jnp.allclose(result, jnp.eye(5), atol=1e-12)

    def test_1x1_no_op(self) -> None:
        """1×1 matrix passes through."""
        mat = jnp.ones((1, 1))
        result = validate_and_repair(mat)
        assert jnp.allclose(result, jnp.ones((1, 1)))

    def test_invalid_method_raises(self) -> None:
        """Unknown method name raises ValueError."""
        mat = jnp.eye(2)
        with pytest.raises(ValueError, match="Unknown repair method"):
            validate_and_repair(mat, method="unknown")

    def test_no_labels_returns_array(self) -> None:
        """Without labels, returns a plain array."""
        mat = jnp.eye(3)
        result = validate_and_repair(mat)
        assert isinstance(result, jax.Array)
        assert not isinstance(result, SymmetricLabelledMatrix)


# ===========================================================================
# _correlation_residuals
# ===========================================================================


class TestCorrelationResiduals:
    """Tests for ``_correlation_residuals``."""

    def test_zero_residuals_at_optimum(self) -> None:
        """Residuals are zero when angles exactly match the target."""
        n = 3
        n_angles = n * (n - 1) // 2
        angles = jnp.full(n_angles, jnp.pi / 2)
        target = jnp.eye(n)
        residuals = _correlation_residuals(angles, target=target, n=n)
        assert jnp.allclose(residuals, 0.0, atol=1e-12)

    def test_residual_count(self) -> None:
        """Number of residuals is n*(n+1)/2 (lower-triangular elements)."""
        n = 4
        n_angles = n * (n - 1) // 2
        angles = jnp.ones(n_angles)
        target = jnp.eye(n)
        residuals = _correlation_residuals(angles, target=target, n=n)
        expected_count = n * (n + 1) // 2
        assert residuals.shape == (expected_count,)
