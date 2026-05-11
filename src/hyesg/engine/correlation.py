"""Correlation engine — Cholesky decomposition and shock pipeline.

Provides correlation matrix validation, Cholesky factorisation with
automatic nearest-PSD fallback, and a split pipeline for correlating
shocks (copula vs non-copula).

All functions are pure and compatible with ``jax.jit``.

Shock convention
~~~~~~~~~~~~~~~~
Models receive raw N(0,1) shocks and multiply by √dt themselves.
The Cholesky factor *L* is pre-computed **once** and reused across
all timesteps via ``functools.partial``.

Split pipeline
~~~~~~~~~~~~~~
Copula-participating shocks go through CDF transform BEFORE
correlation (handled in the copula module).  Non-copula shocks
have Cholesky applied directly to raw normals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Correlation matrix validation
# ---------------------------------------------------------------------------

_SYMMETRY_TOL = 1e-10
_DIAG_TOL = 1e-12
_EIGENVALUE_TOL = -1e-10


def validate_correlation_matrix(matrix: Array) -> tuple[bool, list[str]]:
    """Validate that a matrix is a proper correlation matrix.

    Checks:
    * Square.
    * Symmetric (within tolerance).
    * Diagonal entries are all 1.
    * Positive semi-definite (eigenvalues ≥ 0).

    Args:
        matrix: 2-D array to validate.

    Returns:
        Tuple of ``(is_valid, error_messages)`` where *is_valid* is
        ``True`` when no issues are found and *error_messages* is a
        list of human-readable descriptions of any failures.
    """
    errors: list[str] = []

    if matrix.ndim != 2:
        errors.append(f"Expected 2-D matrix, got {matrix.ndim}-D")
        return False, errors

    n, m = matrix.shape
    if n != m:
        errors.append(f"Matrix is not square: shape ({n}, {m})")
        return False, errors

    # Symmetry
    max_asym = float(jnp.max(jnp.abs(matrix - matrix.T)))
    if max_asym > _SYMMETRY_TOL:
        errors.append(f"Matrix is not symmetric: max |A - A.T| = {max_asym:.2e}")

    # Diagonal = 1
    diag = jnp.diag(matrix)
    max_diag_err = float(jnp.max(jnp.abs(diag - 1.0)))
    if max_diag_err > _DIAG_TOL:
        errors.append(f"Diagonal not all 1: max |diag - 1| = {max_diag_err:.2e}")

    # Positive semi-definite
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    min_eig = float(jnp.min(eigenvalues))
    if min_eig < _EIGENVALUE_TOL:
        errors.append(f"Not positive semi-definite: min eigenvalue = {min_eig:.2e}")

    is_valid = len(errors) == 0
    return is_valid, errors


# ---------------------------------------------------------------------------
# Nearest PSD — Higham's alternating projections
# ---------------------------------------------------------------------------

_HIGHAM_MAX_ITER = 100
_HIGHAM_TOL = 1e-12


def _project_psd(matrix: Array) -> Array:
    """Project onto the positive semi-definite cone.

    Eigendecomposes, clamps negative eigenvalues to zero, and
    reconstructs.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    eigenvalues_clamped = jnp.maximum(eigenvalues, 0.0)
    return eigenvectors @ jnp.diag(eigenvalues_clamped) @ eigenvectors.T


def _project_unit_diagonal(matrix: Array) -> Array:
    """Project onto the set of matrices with unit diagonal."""
    return matrix.at[jnp.diag_indices(matrix.shape[0])].set(1.0)


def nearest_psd(matrix: Array) -> Array:
    """Compute the nearest positive semi-definite correlation matrix.

    Uses Higham's alternating projections algorithm, iterating between
    projections onto the PSD cone and the unit-diagonal constraint set.

    Args:
        matrix: A symmetric matrix (not necessarily PSD).

    Returns:
        The nearest PSD correlation matrix in the Frobenius norm sense.
    """
    n = matrix.shape[0]
    if n == 1:
        return jnp.ones((1, 1), dtype=matrix.dtype)

    # Symmetrise input
    y = 0.5 * (matrix + matrix.T)
    delta_s = jnp.zeros_like(y)

    for _ in range(_HIGHAM_MAX_ITER):
        r = y - delta_s
        x = _project_psd(r)
        delta_s = x - r
        y_new = _project_unit_diagonal(x)

        # Convergence check
        diff = float(jnp.linalg.norm(y_new - y, ord="fro"))
        y = y_new
        if diff < _HIGHAM_TOL:
            break

    # Final PSD projection + unit diagonal to guarantee both constraints
    y = _project_psd(y)
    y = _project_unit_diagonal(y)
    # Symmetrise for numerical safety
    y = 0.5 * (y + y.T)

    # Eigenvalue repair: ensure strictly positive eigenvalues for Cholesky
    eigenvalues, eigenvectors = jnp.linalg.eigh(y)
    min_eig = float(jnp.min(eigenvalues))
    if min_eig < 1e-10:
        eigenvalues_repaired = jnp.maximum(eigenvalues, 1e-10)
        y = eigenvectors @ jnp.diag(eigenvalues_repaired) @ eigenvectors.T
        y = _project_unit_diagonal(y)
        y = 0.5 * (y + y.T)

    return y


# ---------------------------------------------------------------------------
# Cholesky factorisation
# ---------------------------------------------------------------------------


def cholesky_factor(correlation_matrix: Array) -> Array:
    """Compute the lower-triangular Cholesky factor of a correlation matrix.

    If the matrix is not positive definite (Cholesky produces NaN),
    falls back to :func:`nearest_psd` and retries.

    The factor *L* satisfies ``L @ L.T ≈ correlation_matrix``.

    Args:
        correlation_matrix: A symmetric positive (semi-)definite
            correlation matrix of shape ``(n, n)``.

    Returns:
        Lower-triangular Cholesky factor *L* of shape ``(n, n)``.
    """
    n = correlation_matrix.shape[0]
    if n == 1:
        return jnp.ones((1, 1), dtype=correlation_matrix.dtype)

    # Try direct Cholesky first (JAX returns NaN on failure, not an exception)
    lower = jnp.linalg.cholesky(correlation_matrix)
    if not bool(jnp.any(jnp.isnan(lower))):
        return lower

    # Fall back to nearest PSD
    psd_matrix = nearest_psd(correlation_matrix)
    lower = jnp.linalg.cholesky(psd_matrix)
    return lower


# ---------------------------------------------------------------------------
# Shock correlation
# ---------------------------------------------------------------------------


@jax.jit
def correlate_shocks(independent_shocks: Array, cholesky_L: Array) -> Array:
    """Apply correlation to independent standard-normal shocks.

    Computes ``z_corr = (L @ z_indep.T).T`` so that each row of the
    output has the target correlation structure.

    The Cholesky factor *L* should be pre-computed once and passed via
    ``functools.partial`` for reuse across timesteps.

    Args:
        independent_shocks: Array of shape ``(n_steps, n_shocks)``
            containing i.i.d. N(0, 1) samples.
        cholesky_L: Lower-triangular Cholesky factor of shape
            ``(n_shocks, n_shocks)``.

    Returns:
        Correlated shocks of the same shape ``(n_steps, n_shocks)``.
    """
    return (cholesky_L @ independent_shocks.T).T


# ---------------------------------------------------------------------------
# Split / merge pipeline for copula shocks
# ---------------------------------------------------------------------------


def split_copula_shocks(
    shocks: Array,
    copula_mask: Array,
) -> tuple[Array, Array]:
    """Split shocks into copula-participating and non-copula subsets.

    The *copula_mask* is a configuration-level quantity (fixed for the
    entire simulation) so the split is performed outside the inner
    ``jax.lax.scan`` loop.  The function is pure and uses only JAX ops.

    Args:
        shocks: Array of shape ``(n_steps, n_shocks)``.
        copula_mask: Boolean array of shape ``(n_shocks,)`` where
            ``True`` indicates the shock participates in the copula.

    Returns:
        Tuple ``(copula_shocks, non_copula_shocks)`` where each has
        shape ``(n_steps, n_selected)`` with the number of selected
        columns determined by the mask.
    """
    # Concrete mask → concrete indices (safe outside JIT trace)
    mask_np = jnp.asarray(copula_mask, dtype=jnp.bool_)
    copula_idx = jnp.nonzero(mask_np, size=int(jnp.sum(mask_np)))[0]
    non_copula_idx = jnp.nonzero(~mask_np, size=int(jnp.sum(~mask_np)))[0]

    copula_shocks = shocks[:, copula_idx]
    non_copula_shocks = shocks[:, non_copula_idx]

    return copula_shocks, non_copula_shocks


def merge_copula_shocks(
    copula_shocks: Array,
    non_copula_shocks: Array,
    copula_mask: Array,
) -> Array:
    """Merge copula and non-copula shocks back into a single array.

    Inverse of :func:`split_copula_shocks`.

    Args:
        copula_shocks: Array of shape ``(n_steps, n_copula)``.
        non_copula_shocks: Array of shape ``(n_steps, n_non_copula)``.
        copula_mask: Boolean array of shape ``(n_shocks,)`` matching
            the mask used in :func:`split_copula_shocks`.

    Returns:
        Merged array of shape ``(n_steps, n_shocks)`` with columns
        placed back in their original positions.
    """
    n_steps = copula_shocks.shape[0]
    n_shocks = copula_mask.shape[0]

    mask_np = jnp.asarray(copula_mask, dtype=jnp.bool_)
    n_copula = int(jnp.sum(mask_np))
    copula_idx = jnp.nonzero(mask_np, size=n_copula)[0]
    non_copula_idx = jnp.nonzero(~mask_np, size=n_shocks - n_copula)[0]

    merged = jnp.zeros((n_steps, n_shocks), dtype=copula_shocks.dtype)
    merged = merged.at[:, copula_idx].set(copula_shocks)
    merged = merged.at[:, non_copula_idx].set(non_copula_shocks)

    return merged
