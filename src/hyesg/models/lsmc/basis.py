"""Basis function constructors for LSMC regression."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def polynomial_basis(x: Array, degree: int = 3) -> Array:
    """Polynomial basis [1, x, x², x³, ...].

    Args:
        x: State variable values, shape (n_paths,).
        degree: Maximum polynomial degree.

    Returns:
        Basis matrix, shape (n_paths, degree + 1).
    """
    return jnp.column_stack([x**k for k in range(degree + 1)])


def laguerre_basis(x: Array, degree: int = 3) -> Array:
    """Weighted Laguerre polynomial basis exp(-u/2) * L_n(u).

    The input is normalised by its mean before applying the Laguerre
    polynomials and exponential weighting, so that the basis remains
    well-conditioned for arbitrary-scale inputs (e.g. stock prices).

    Uses the three-term recurrence relation:
        L_0(u) = 1
        L_1(u) = 1 - u
        L_{n+1}(u) = ((2n + 1 - u) * L_n(u) - n * L_{n-1}(u)) / (n + 1)

    Args:
        x: State variable values, shape (n_paths,).
        degree: Maximum Laguerre degree.

    Returns:
        Basis matrix, shape (n_paths, degree + 1).
    """
    # Normalise to unit-mean so exp(-u/2) stays well-conditioned
    x_mean = jnp.mean(x)
    u = jnp.where(x_mean > 0, x / x_mean, x)
    weight = jnp.exp(-u / 2.0)

    columns: list[Array] = []
    if degree >= 0:
        l_prev = jnp.ones_like(u)
        columns.append(weight * l_prev)
    if degree >= 1:
        l_curr = 1.0 - u
        columns.append(weight * l_curr)

    for n in range(1, degree):
        l_next = ((2.0 * n + 1.0 - u) * l_curr - n * l_prev) / (n + 1.0)
        columns.append(weight * l_next)
        l_prev = l_curr
        l_curr = l_next

    return jnp.column_stack(columns)
