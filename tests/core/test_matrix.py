"""Tests for LabelledMatrix and SymmetricLabelledMatrix."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.core.matrix import LabelledMatrix, SymmetricLabelledMatrix


# ---------------------------------------------------------------------------
# LabelledMatrix
# ---------------------------------------------------------------------------


class TestLabelledMatrix:
    """Tests for ``LabelledMatrix``."""

    def test_construction_basic(self) -> None:
        """Basic construction stores data and labels."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mat = LabelledMatrix(data, ["a", "b"], ["x", "y"])
        assert mat.shape == (2, 2)
        assert mat.row_labels == ("a", "b")
        assert mat.col_labels == ("x", "y")
        assert jnp.allclose(mat.data, data)

    def test_label_based_access(self) -> None:
        """Element access by label pair."""
        data = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        mat = LabelledMatrix(data, ["UK_Equity", "US_Equity"], ["UK_Equity", "US_Equity"])
        assert mat["UK_Equity", "US_Equity"] == pytest.approx(0.5)
        assert mat["UK_Equity", "UK_Equity"] == pytest.approx(1.0)

    def test_label_access_asymmetric(self) -> None:
        """Label access on non-square matrix."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mat = LabelledMatrix(data, ["r1", "r2"], ["c1", "c2", "c3"])
        assert mat["r1", "c3"] == pytest.approx(3.0)
        assert mat["r2", "c1"] == pytest.approx(4.0)

    def test_invalid_row_label_raises_key_error(self) -> None:
        """Missing row label raises KeyError."""
        data = jnp.eye(2)
        mat = LabelledMatrix(data, ["a", "b"], ["a", "b"])
        with pytest.raises(KeyError, match="not_a_label"):
            _ = mat["not_a_label", "a"]

    def test_invalid_col_label_raises_key_error(self) -> None:
        """Missing column label raises KeyError."""
        data = jnp.eye(2)
        mat = LabelledMatrix(data, ["a", "b"], ["a", "b"])
        with pytest.raises(KeyError, match="bad_col"):
            _ = mat["a", "bad_col"]

    def test_dimension_mismatch_rows(self) -> None:
        """Wrong number of row labels raises ValueError."""
        data = jnp.eye(3)
        with pytest.raises(ValueError, match="Row label count"):
            LabelledMatrix(data, ["a", "b"], ["x", "y", "z"])

    def test_dimension_mismatch_cols(self) -> None:
        """Wrong number of column labels raises ValueError."""
        data = jnp.eye(3)
        with pytest.raises(ValueError, match="Column label count"):
            LabelledMatrix(data, ["a", "b", "c"], ["x", "y"])

    def test_1d_array_rejected(self) -> None:
        """1-D array raises ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            LabelledMatrix(jnp.ones(3), ["a", "b", "c"], ["x"])

    def test_1x1_matrix(self) -> None:
        """1×1 matrix edge case."""
        data = jnp.array([[42.0]])
        mat = LabelledMatrix(data, ["only"], ["only"])
        assert mat.shape == (1, 1)
        assert mat["only", "only"] == pytest.approx(42.0)

    def test_submatrix_extraction(self) -> None:
        """Submatrix extracts correct rows and columns."""
        data = jnp.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ])
        mat = LabelledMatrix(data, ["a", "b", "c"], ["a", "b", "c"])
        sub = mat.submatrix(["a", "c"], ["b", "c"])
        assert sub.shape == (2, 2)
        assert sub.row_labels == ("a", "c")
        assert sub.col_labels == ("b", "c")
        assert sub["a", "b"] == pytest.approx(0.5)
        assert sub["c", "c"] == pytest.approx(1.0)

    def test_submatrix_invalid_label(self) -> None:
        """Submatrix with invalid label raises KeyError."""
        data = jnp.eye(2)
        mat = LabelledMatrix(data, ["a", "b"], ["a", "b"])
        with pytest.raises(KeyError):
            mat.submatrix(["a", "missing"], ["a"])

    def test_repr(self) -> None:
        """repr includes shape and labels."""
        data = jnp.eye(2)
        mat = LabelledMatrix(data, ["r1", "r2"], ["c1", "c2"])
        r = repr(mat)
        assert "LabelledMatrix" in r
        assert "(2, 2)" in r

    def test_data_is_jax_array(self) -> None:
        """Data property returns a JAX array."""
        data = jnp.eye(2)
        mat = LabelledMatrix(data, ["a", "b"], ["a", "b"])
        assert isinstance(mat.data, jax.Array)

    def test_labels_are_tuples(self) -> None:
        """Labels are stored as tuples (immutable)."""
        data = jnp.eye(2)
        mat = LabelledMatrix(data, ["a", "b"], ["x", "y"])
        assert isinstance(mat.row_labels, tuple)
        assert isinstance(mat.col_labels, tuple)


# ---------------------------------------------------------------------------
# SymmetricLabelledMatrix
# ---------------------------------------------------------------------------


class TestSymmetricLabelledMatrix:
    """Tests for ``SymmetricLabelledMatrix``."""

    def test_construction_symmetric(self) -> None:
        """Symmetric matrix is accepted."""
        data = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        mat = SymmetricLabelledMatrix(data, ["a", "b"])
        assert mat.shape == (2, 2)
        assert mat.row_labels == ("a", "b")
        assert mat.col_labels == ("a", "b")

    def test_symmetry_enforcement(self) -> None:
        """Tiny asymmetry within tolerance is symmetrised."""
        eps = 1e-16
        data = jnp.array([[1.0, 0.5 + eps], [0.5 - eps, 1.0]])
        mat = SymmetricLabelledMatrix(data, ["a", "b"], symmetry_tol=1e-14)
        # Result should be exactly symmetric
        assert jnp.allclose(mat.data, mat.data.T, atol=0.0)

    def test_rejects_asymmetric(self) -> None:
        """Clearly asymmetric matrix is rejected."""
        data = jnp.array([[1.0, 0.5], [0.9, 1.0]])
        with pytest.raises(ValueError, match="not symmetric"):
            SymmetricLabelledMatrix(data, ["a", "b"])

    def test_rejects_non_square(self) -> None:
        """Non-square matrix is rejected."""
        data = jnp.ones((2, 3))
        with pytest.raises(ValueError, match="square"):
            SymmetricLabelledMatrix(data, ["a", "b"])

    def test_label_count_mismatch(self) -> None:
        """Wrong number of labels raises ValueError."""
        data = jnp.eye(3)
        with pytest.raises(ValueError, match="Label count"):
            SymmetricLabelledMatrix(data, ["a", "b"])

    def test_1x1_symmetric(self) -> None:
        """1×1 matrix edge case."""
        data = jnp.array([[1.0]])
        mat = SymmetricLabelledMatrix(data, ["only"])
        assert mat.shape == (1, 1)
        assert mat["only", "only"] == pytest.approx(1.0)

    def test_label_access(self) -> None:
        """Label-based access works on symmetric matrix."""
        data = jnp.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ])
        mat = SymmetricLabelledMatrix(data, ["UK_Equity", "US_Equity", "Gilts"])
        assert mat["UK_Equity", "US_Equity"] == pytest.approx(0.5)
        assert mat["US_Equity", "UK_Equity"] == pytest.approx(0.5)
        assert mat["Gilts", "Gilts"] == pytest.approx(1.0)

    def test_submatrix_inherits_type(self) -> None:
        """Submatrix from SymmetricLabelledMatrix returns LabelledMatrix."""
        data = jnp.eye(3)
        mat = SymmetricLabelledMatrix(data, ["a", "b", "c"])
        sub = mat.submatrix(["a", "c"], ["a", "c"])
        # submatrix returns LabelledMatrix (parent type)
        assert isinstance(sub, LabelledMatrix)
        assert sub.shape == (2, 2)

    def test_repr_symmetric(self) -> None:
        """repr includes class name and dimension."""
        data = jnp.eye(3)
        mat = SymmetricLabelledMatrix(data, ["a", "b", "c"])
        r = repr(mat)
        assert "SymmetricLabelledMatrix" in r
        assert "n=3" in r

    def test_custom_tolerance(self) -> None:
        """Custom symmetry tolerance is respected."""
        data = jnp.array([[1.0, 0.5], [0.4, 1.0]])
        # Default tolerance should reject
        with pytest.raises(ValueError):
            SymmetricLabelledMatrix(data, ["a", "b"])
        # Relaxed tolerance should accept
        mat = SymmetricLabelledMatrix(data, ["a", "b"], symmetry_tol=0.2)
        assert mat.shape == (2, 2)
