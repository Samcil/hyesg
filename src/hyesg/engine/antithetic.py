"""Antithetic variance reduction for Monte Carlo simulation.

Provides three modes of antithetic sampling:

Non-copula (standard)
~~~~~~~~~~~~~~~~~~~~~
Negate the standard-normal shocks: ``z_anti = -z``.  Run the
simulation twice (once with *z*, once with *-z*) and average the
results.  By the symmetry of the normal distribution, this produces
an unbiased estimator with lower variance.

Copula — Deviation D1 (mathematically correct)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Complement the **post-CDF** uniforms: ``u_anti = 1 - u``.  This
preserves the copula dependence structure in the uniform space where
it is defined.  After the complement, the uniforms are passed back
through the inverse CDF (quantile function) of the marginal
distribution — which may be Student-t, not Gaussian.

C# compatibility mode
~~~~~~~~~~~~~~~~~~~~~
The C# ESG engine applies antithetic as ``z_anti = -z`` **before**
the CDF transform (pre-CDF negation).  This is correct when the
marginal is Gaussian (because ``Φ(-z) = 1 - Φ(z)``), but **wrong**
for Student-t or any non-symmetric marginal.  Use
``match_csharp=True`` when exact parity with the C# engine is
required.

All functions are pure and compatible with ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Primitive antithetic transforms
# ---------------------------------------------------------------------------


@jax.jit
def apply_antithetic_normal(normals: Array) -> Array:
    """Negate normal shocks: ``z_anti = -z``.

    Used for non-copula antithetic sampling, or for C# compatibility
    mode (pre-CDF negation).

    Args:
        normals: Array of standard-normal samples (any shape).

    Returns:
        Negated array of the same shape and dtype.
    """
    return -normals


@jax.jit
def apply_antithetic_uniform(uniforms: Array) -> Array:
    """Complement uniforms: ``u_anti = 1 - u``.

    Used in the copula pipeline for post-CDF antithetic (Deviation D1).
    This is the mathematically correct approach for non-Gaussian
    marginals.

    Args:
        uniforms: Array of U(0, 1) samples (any shape).

    Returns:
        Complemented array of the same shape and dtype.
    """
    return 1.0 - uniforms


# ---------------------------------------------------------------------------
# Shock-level antithetic generation
# ---------------------------------------------------------------------------


def generate_antithetic_shocks(
    shocks: Array,
    copula_mask: Array | None = None,
) -> Array:
    """Generate the antithetic version of a shock array.

    For non-copula shocks (or when *copula_mask* is ``None``), all
    columns are negated (``z_anti = -z``).

    When a *copula_mask* is provided, only the **non-copula** columns
    are negated.  Copula-participating columns are left unchanged
    because their antithetic pairing is handled separately in the
    copula pipeline via :func:`apply_antithetic_uniform` after the
    CDF transform.

    Args:
        shocks: Array of shape ``(n_steps, n_shocks)`` containing
            standard-normal samples.
        copula_mask: Optional boolean array of shape ``(n_shocks,)``
            where ``True`` indicates the shock participates in the
            copula.  When ``None``, all shocks are negated.

    Returns:
        Antithetic shock array of the same shape.
    """
    if copula_mask is None:
        return apply_antithetic_normal(shocks)

    mask = jnp.asarray(copula_mask, dtype=jnp.bool_)
    # Negate non-copula columns; leave copula columns unchanged
    negate = jnp.where(mask, 1.0, -1.0)
    return shocks * negate


# ---------------------------------------------------------------------------
# Result combination
# ---------------------------------------------------------------------------


@jax.jit
def antithetic_combine(
    results_original: Array,
    results_antithetic: Array,
) -> Array:
    """Element-wise average of original and antithetic results.

    ``combined = (original + antithetic) / 2``

    Args:
        results_original: Array of simulation results from the
            original shocks (any shape).
        results_antithetic: Array of simulation results from the
            antithetic shocks (same shape as *results_original*).

    Returns:
        Averaged array of the same shape.
    """
    return (results_original + results_antithetic) / 2.0


def antithetic_combine_pytree(
    original: dict,
    antithetic: dict,
) -> dict:
    """Combine two pytree dicts of arrays element-wise.

    Uses ``jax.tree.map`` to average each leaf array, so the
    function works with arbitrarily nested dicts of arrays.

    Args:
        original: Pytree dict of arrays from the original simulation.
        antithetic: Pytree dict of arrays from the antithetic
            simulation (same structure as *original*).

    Returns:
        Pytree dict with each leaf replaced by the element-wise
        average of the corresponding leaves.
    """
    return jax.tree.map(antithetic_combine, original, antithetic)
