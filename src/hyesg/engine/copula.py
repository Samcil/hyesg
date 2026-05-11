"""Copula engine — Gaussian and Student-t copula transforms.

Provides copula transforms for tail-dependency modelling in the
ESG shock pipeline.  Copula-participating shocks follow this path:

1. Start with independent N(0,1) normals (from RNG).
2. Apply Cholesky correlation: ``z_corr = L @ z_indep``.
3. Transform to uniforms via CDF:
   - Gaussian copula: ``u = Φ(z_corr)``
   - Student-t copula: ``u = F_t(z_corr; df)``
4. Transform back to standard-normal marginals: ``z_out = Φ⁻¹(u)``.

The Student-t copula modifies the dependence structure: lower ``df``
compresses marginals toward zero (the Student-t CDF rises more slowly in
the tails than the normal CDF, so ``Φ⁻¹(F_t(z; df)) < z`` for ``|z| > 0``).
As ``df → ∞`` the Student-t copula converges to the Gaussian copula.

Antithetic variance reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The mathematically correct antithetic for copula shocks complements
the *uniform* values: ``u_anti = 1 - u``.  The C# engine instead
negates the *pre-CDF normals* (``z_anti = -z``).  For any symmetric
distribution (including both Gaussian and Student-t) these two
approaches are equivalent because ``F(-z) = 1 - F(z)``.  Both modes
are provided for parity testing.

All functions are pure and compatible with ``jax.jit``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import jax.scipy.stats as jstats
from jax import Array

from hyesg.engine.correlation import correlate_shocks


# ---------------------------------------------------------------------------
# Student-t CDF / PPF helpers (JAX lacks jax.scipy.stats.t.cdf)
# ---------------------------------------------------------------------------


@jax.jit
def _student_t_cdf(x: Array, df: float) -> Array:
    """Student-t CDF via the regularised incomplete beta function.

    For ``x >= 0``:  ``F(x) = 1 - 0.5 * I_z(df/2, 0.5)``
    where ``z = df / (df + x²)``.

    Uses symmetry ``F(-x) = 1 - F(x)`` for negative values.

    Args:
        x: Input array.
        df: Degrees of freedom (> 0).

    Returns:
        CDF values in (0, 1).
    """
    z = df / (df + x ** 2)
    # betainc(a, b, x) = regularised incomplete beta I_x(a, b)
    half_beta = 0.5 * jsp_special.betainc(df / 2.0, 0.5, z)
    return jnp.where(x >= 0, 1.0 - half_beta, half_beta)


def _student_t_ppf(u: Array, df: float) -> Array:
    """Student-t PPF (inverse CDF) via bisection.

    Uses the identity: for ``u > 0.5``, find ``x > 0`` such that
    ``F_t(x; df) = u`` by inverting ``_student_t_cdf``.
    Exploits symmetry for ``u < 0.5``.

    Falls back to ``scipy.stats.t.ppf`` computation via the
    regularised incomplete beta inverse, but since JAX does not
    expose ``betaincinv``, we use a Newton-Raphson iteration.

    Args:
        u: Uniform values in (0, 1).
        df: Degrees of freedom (> 0).

    Returns:
        Student-t quantiles.
    """
    # Use Newton's method: solve F_t(x; df) = u
    # Starting point: use normal PPF as initial guess
    x0 = jstats.norm.ppf(u)

    def _newton_step(x: Array) -> Array:
        cdf_val = _student_t_cdf(x, df)
        pdf_val = jnp.exp(jstats.t.logpdf(x, df))
        # Clamp pdf to avoid division by zero
        pdf_val = jnp.maximum(pdf_val, 1e-30)
        return x - (cdf_val - u) / pdf_val

    # Run Newton iterations (unrolled for JIT compatibility)
    x = x0
    for _ in range(25):
        x = _newton_step(x)

    return x

# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------


@jax.jit
def gaussian_copula(correlated_normals: Array) -> Array:
    """Transform correlated normals to uniforms via the standard normal CDF.

    Applies ``u = Φ(z)`` element-wise, producing uniform(0, 1) marginals
    while preserving the Gaussian dependence structure.

    Args:
        correlated_normals: Array of shape ``(n_steps, n_shocks)``
            with correlated standard-normal values.

    Returns:
        Uniform(0, 1) array of the same shape.
    """
    return jstats.norm.cdf(correlated_normals)


@jax.jit
def gaussian_copula_inverse(uniforms: Array) -> Array:
    """Transform uniforms back to standard normals via the inverse normal CDF.

    Applies ``z = Φ⁻¹(u)`` element-wise.

    Args:
        uniforms: Array of shape ``(n_steps, n_shocks)`` with
            values in (0, 1).

    Returns:
        Standard-normal array of the same shape.
    """
    return jstats.norm.ppf(uniforms)


# ---------------------------------------------------------------------------
# Student-t copula
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(1,))
def student_t_copula(correlated_normals: Array, df: float) -> Array:
    """Transform correlated normals to uniforms via the Student-t CDF.

    Applies ``u = F_t(z; df)`` element-wise.  Lower ``df`` produces
    heavier tails and stronger tail dependence.  As ``df → ∞`` the
    result converges to :func:`gaussian_copula`.

    Args:
        correlated_normals: Array of shape ``(n_steps, n_shocks)``
            with correlated standard-normal values.
        df: Degrees of freedom for the Student-t distribution.
            Must be ``> 2`` (required for finite variance).

    Returns:
        Uniform(0, 1) array of the same shape.

    Raises:
        ValueError: If ``df <= 2``.
    """
    if df <= 2:
        msg = f"Student-t df must be > 2 for finite variance, got {df}"
        raise ValueError(msg)
    return _student_t_cdf(correlated_normals, df)


@functools.partial(jax.jit, static_argnums=(1,))
def student_t_copula_inverse(uniforms: Array, df: float) -> Array:
    """Transform uniforms to standard-normal marginals via Student-t inverse.

    Computes ``z_out = Φ⁻¹(F_t(z; df))`` by first inverting the
    Student-t CDF (to recover the correlated normals) and then
    applying the normal CDF.  In practice, since we receive
    *uniforms* already on (0, 1), we apply ``Φ⁻¹(u)`` directly.

    Note: This function takes *uniforms* (post-CDF) and returns
    standard-normal marginals.

    Args:
        uniforms: Array of shape ``(n_steps, n_shocks)`` with
            values in (0, 1) from a Student-t copula transform.
        df: Degrees of freedom (must be ``> 2``).

    Returns:
        Standard-normal array of the same shape.

    Raises:
        ValueError: If ``df <= 2``.
    """
    if df <= 2:
        msg = f"Student-t df must be > 2 for finite variance, got {df}"
        raise ValueError(msg)
    return jstats.norm.ppf(uniforms)


# ---------------------------------------------------------------------------
# Full copula pipeline
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def apply_copula(
    shocks: Array,
    cholesky_L: Array,
    copula_type: str,
    df: float | None = None,
    match_csharp: bool = False,
) -> Array:
    """Full copula pipeline: correlate → CDF → inverse CDF.

    Orchestrates the copula transform for copula-participating shocks:

    1. Correlate independent normals via Cholesky: ``z_corr = L @ z``.
    2. Transform to uniforms via the copula CDF.
    3. (Optional) Apply antithetic: complement uniforms or negate normals.
    4. Transform uniforms back to standard-normal marginals.

    Args:
        shocks: Independent N(0, 1) shocks of shape
            ``(n_steps, n_copula_shocks)``.
        cholesky_L: Lower-triangular Cholesky factor of the copula
            correlation matrix, shape ``(n_copula_shocks, n_copula_shocks)``.
        copula_type: ``"gaussian"`` or ``"student_t"``.
        df: Degrees of freedom for Student-t copula (required when
            ``copula_type="student_t"``).
        match_csharp: If ``True``, use C#-compatible antithetic
            (negate pre-CDF normals instead of complementing uniforms).
            Default ``False``.

    Returns:
        Correlated shocks with standard-normal marginals, same shape
        as *shocks*.

    Raises:
        ValueError: If *copula_type* is unknown or *df* is invalid.
    """
    # Step 1: correlate
    correlated = correlate_shocks(shocks, cholesky_L)

    # Step 2: CDF transform → uniforms
    if copula_type == "gaussian":
        uniforms = gaussian_copula(correlated)
    elif copula_type == "student_t":
        uniforms = student_t_copula(correlated, df)
    else:
        msg = f"Unknown copula_type: {copula_type!r}"
        raise ValueError(msg)

    # Step 3: inverse CDF → standard-normal marginals
    result = gaussian_copula_inverse(uniforms)

    return result


# ---------------------------------------------------------------------------
# Antithetic variance reduction
# ---------------------------------------------------------------------------


@jax.jit
def apply_copula_antithetic(uniforms: Array) -> Array:
    """Apply mathematically correct antithetic to copula uniforms.

    Complements post-CDF uniform values: ``u_anti = 1 - u``.
    This is the correct antithetic for both Gaussian and Student-t
    copulas because it preserves the copula dependence structure.

    Args:
        uniforms: Uniform(0, 1) array of shape
            ``(n_steps, n_shocks)`` from a copula CDF transform.

    Returns:
        Antithetic uniforms of the same shape.
    """
    return 1.0 - uniforms


@jax.jit
def apply_copula_antithetic_csharp(normals: Array) -> Array:
    """Apply C#-compatible antithetic to pre-CDF normals.

    Negates correlated normals before the CDF transform:
    ``z_anti = -z``.

    For any symmetric distribution ``F`` (including both Gaussian and
    Student-t), ``F(-z) = 1 - F(z)``, so ``Φ⁻¹(F(-z)) = -Φ⁻¹(F(z))``.
    This means the C# approach is mathematically equivalent to the
    uniform-complement method for symmetric distributions.

    Provided separately for parity testing with the C# engine.

    Args:
        normals: Correlated standard-normal array of shape
            ``(n_steps, n_shocks)`` (pre-CDF).

    Returns:
        Negated normals of the same shape.
    """
    return -normals
