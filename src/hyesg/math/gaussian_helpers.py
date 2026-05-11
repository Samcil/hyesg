"""Gaussian (OU-based) short rate model helpers.

Shared pure-JAX functions for Vasicek and G1++/G2++ models.
All functions use float64 precision and are JIT-compatible.

Key formulas:
    B(α, τ) = (1 - e^{-ατ}) / α   (duration-like coefficient)
    V²(σ, α, τ) = (σ²/α²)[τ - B(τ) - ½αB(τ)²]   (OU variance integral)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, Float


def b_over_dt(y: Float[Array, ...] | float) -> Float[Array, ...]:
    """Compute (1 - e^{-y}) / y with Taylor guard for small |y|.

    For |y| < 1e-8, uses the Taylor expansion:
        (1 - e^{-y}) / y ≈ 1 - y/2 + y²/6 - y³/24

    This avoids the 0/0 indeterminate form at y = 0.

    Args:
        y: Input value (typically α·τ).

    Returns:
        The ratio (1 - e^{-y}) / y.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    safe_y = jnp.where(jnp.abs(y) < 1e-8, jnp.ones_like(y), y)
    standard = (1.0 - jnp.exp(-safe_y)) / safe_y
    taylor = 1.0 - y / 2.0 + y**2 / 6.0 - y**3 / 24.0
    return jnp.where(jnp.abs(y) < 1e-8, taylor, standard)


def b_func(
    alpha: float,
    tau: Float[Array, ...] | float,
) -> Float[Array, ...]:
    """Compute B(α, τ) = (1 - e^{-ατ}) / α = τ · b_over_dt(ατ).

    This is the standard duration coefficient appearing in
    Vasicek/Hull-White/G++ ZCB pricing formulas.

    Args:
        alpha: Mean-reversion speed.
        tau: Time to maturity T - t.

    Returns:
        B(α, τ) coefficient.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    return tau * b_over_dt(alpha * tau)


def variance_integral_ou(
    sigma: float,
    alpha: float,
    tau: Float[Array, ...] | float,
) -> Float[Array, ...]:
    """OU variance integral V²(σ, α, τ) for bond pricing.

    V²(t, T) = (σ²/α²)[τ - B(τ) - ½αB(τ)²]

    where B(τ) = (1 - e^{-ατ}) / α and τ = T - t.

    This quantity appears in the exponent of the G1++/Vasicek
    ZCB pricing formula.

    Args:
        sigma: OU volatility.
        alpha: Mean-reversion speed.
        tau: Time to maturity T - t.

    Returns:
        V²(σ, α, τ).
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    B = b_func(alpha, tau)
    return (sigma**2 / alpha**2) * (tau - B - 0.5 * alpha * B**2)
