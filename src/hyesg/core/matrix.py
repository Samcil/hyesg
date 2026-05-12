"""Labelled matrix types for correlation and covariance structures.

Provides ``LabelledMatrix`` for generic label-based access and
``SymmetricLabelledMatrix`` for correlation matrices that enforce
symmetry on construction.
"""

from __future__ import annotations

from typing import Generic, Sequence, TypeVar

import jax.numpy as jnp
from jax import Array

T = TypeVar("T")


class LabelledMatrix(Generic[T]):
    """Matrix with typed row and column labels.

    Provides label-based access to matrix elements, used for
    correlation matrices where labels are asset/model names.
    """

    def __init__(
        self,
        data: Array,
        row_labels: Sequence[T],
        col_labels: Sequence[T],
    ) -> None:
        """Initialize with data and labels.

        Args:
            data: 2D array of shape (n_rows, n_cols).
            row_labels: Labels for each row.
            col_labels: Labels for each column.

        Raises:
            ValueError: If dimensions don't match labels.
        """
        data = jnp.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {data.ndim}-D")

        n_rows, n_cols = data.shape
        row_labels_tup = tuple(row_labels)
        col_labels_tup = tuple(col_labels)

        if len(row_labels_tup) != n_rows:
            raise ValueError(
                f"Row label count ({len(row_labels_tup)}) does not match "
                f"number of rows ({n_rows})"
            )
        if len(col_labels_tup) != n_cols:
            raise ValueError(
                f"Column label count ({len(col_labels_tup)}) does not match "
                f"number of columns ({n_cols})"
            )

        self._data = data
        self._row_labels = row_labels_tup
        self._col_labels = col_labels_tup
        self._row_index: dict[T, int] = {
            label: i for i, label in enumerate(row_labels_tup)
        }
        self._col_index: dict[T, int] = {
            label: i for i, label in enumerate(col_labels_tup)
        }

    @property
    def data(self) -> Array:
        """Underlying 2-D array."""
        return self._data

    @property
    def row_labels(self) -> tuple[T, ...]:
        """Row labels as a tuple."""
        return self._row_labels

    @property
    def col_labels(self) -> tuple[T, ...]:
        """Column labels as a tuple."""
        return self._col_labels

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying array."""
        return (self._data.shape[0], self._data.shape[1])

    def __getitem__(self, key: tuple[T, T]) -> float:
        """Access element by label pair: ``matrix[row_label, col_label]``.

        Args:
            key: Tuple of (row_label, col_label).

        Returns:
            Scalar value at the given position.

        Raises:
            KeyError: If a label is not found.
        """
        row_label, col_label = key
        if row_label not in self._row_index:
            raise KeyError(f"Row label {row_label!r} not found")
        if col_label not in self._col_index:
            raise KeyError(f"Column label {col_label!r} not found")
        row_idx = self._row_index[row_label]
        col_idx = self._col_index[col_label]
        return float(self._data[row_idx, col_idx])

    def submatrix(
        self, rows: Sequence[T], cols: Sequence[T]
    ) -> LabelledMatrix[T]:
        """Extract submatrix for given row and column labels.

        Args:
            rows: Row labels to include.
            cols: Column labels to include.

        Returns:
            A new ``LabelledMatrix`` containing only the selected rows
            and columns.

        Raises:
            KeyError: If any label is not found.
        """
        row_indices = []
        for label in rows:
            if label not in self._row_index:
                raise KeyError(f"Row label {label!r} not found")
            row_indices.append(self._row_index[label])

        col_indices = []
        for label in cols:
            if label not in self._col_index:
                raise KeyError(f"Column label {label!r} not found")
            col_indices.append(self._col_index[label])

        row_idx_arr = jnp.array(row_indices)
        col_idx_arr = jnp.array(col_indices)
        sub_data = self._data[jnp.ix_(row_idx_arr, col_idx_arr)]
        return LabelledMatrix(sub_data, list(rows), list(cols))

    def __repr__(self) -> str:
        return (
            f"LabelledMatrix(shape={self.shape}, "
            f"row_labels={self._row_labels}, "
            f"col_labels={self._col_labels})"
        )


class SymmetricLabelledMatrix(LabelledMatrix[T]):
    """Symmetric matrix with identical row and column labels.

    Enforces symmetry on construction. Used for correlation matrices.
    """

    def __init__(
        self,
        data: Array,
        labels: Sequence[T],
        symmetry_tol: float = 1e-15,
    ) -> None:
        """Initialize and enforce symmetry.

        The matrix is symmetrised as ``0.5 * (A + A.T)`` after the
        tolerance check passes.

        Args:
            data: 2D symmetric array of shape ``(n, n)``.
            labels: Labels for both rows and columns.
            symmetry_tol: Tolerance for symmetry check.

        Raises:
            ValueError: If matrix is not square or not symmetric
                within tolerance.
        """
        data = jnp.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {data.ndim}-D")

        n, m = data.shape
        if n != m:
            raise ValueError(
                f"Symmetric matrix must be square, got shape ({n}, {m})"
            )

        labels_tup = tuple(labels)
        if len(labels_tup) != n:
            raise ValueError(
                f"Label count ({len(labels_tup)}) does not match "
                f"matrix dimension ({n})"
            )

        max_asym = float(jnp.max(jnp.abs(data - data.T)))
        if max_asym > symmetry_tol:
            raise ValueError(
                f"Matrix is not symmetric: max |A - A.T| = {max_asym:.2e} "
                f"(tolerance = {symmetry_tol:.2e})"
            )

        # Enforce exact symmetry
        data = 0.5 * (data + data.T)
        super().__init__(data, labels_tup, labels_tup)

    def __repr__(self) -> str:
        return (
            f"SymmetricLabelledMatrix(n={self.shape[0]}, "
            f"labels={self._row_labels})"
        )
