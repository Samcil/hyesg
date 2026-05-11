"""Financial pricing formulas.

Pure JAX implementations of Black model, SABR, and bond pricing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float

# ─── Black Model ───


def black_call(
    F: Float[Array, ...] | float,
    K: Float[Array, ...] | float,
    sigma: float,
    tau: float,
    df: float,
) -> Float[Array, ...]:
    """Black's model call option price.

    C = df × [F·N(d₁) - K·N(d₂)]
    where d₁ = [ln(F/K) + σ²τ/2] / (σ√τ), d₂ = d₁ - σ√τ

    Args:
        F: Forward price.
        K: Strike price.
        sigma: Implied volatility.
        tau: Time to expiry.
        df: Discount factor.

    Returns:
        Call option price.
    """
    F = jnp.asarray(F, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    sqrt_tau = jnp.sqrt(tau)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return df * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black_put(
    F: Float[Array, ...] | float,
    K: Float[Array, ...] | float,
    sigma: float,
    tau: float,
    df: float,
) -> Float[Array, ...]:
    """Black's model put option price via put-call parity.

    P = df × [K·N(-d₂) - F·N(-d₁)]

    Args:
        F: Forward price.
        K: Strike price.
        sigma: Implied volatility.
        tau: Time to expiry.
        df: Discount factor.

    Returns:
        Put option price.
    """
    F = jnp.asarray(F, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    sqrt_tau = jnp.sqrt(tau)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black_implied_vol(
    price: float,
    F: float,
    K: float,
    tau: float,
    df: float,
    is_call: bool,
    tol: float = 1e-15,
    max_iter: int = 100,
) -> Float[Array, ""]:
    """Black implied volatility via Newton-Raphson.

    Uses vega (∂C/∂σ) for faster convergence.

    Args:
        price: Observed option price.
        F: Forward price.
        K: Strike price.
        tau: Time to expiry.
        df: Discount factor.
        is_call: True for call, False for put.
        tol: Convergence tolerance for price error.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility.
    """
    F = jnp.asarray(F, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    price = jnp.asarray(price, dtype=jnp.float64)

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = jnp.sqrt(2.0 * jnp.pi / tau) * price / (df * F)
    sigma = jnp.clip(sigma, 0.001, 5.0)
    diff = jnp.asarray(jnp.inf, dtype=jnp.float64)

    def body_fn(carry: tuple) -> tuple:
        sigma, diff, i = carry
        sqrt_tau = jnp.sqrt(tau)
        d1 = (jnp.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau

        model_price = jnp.where(
            is_call,
            df * (F * norm.cdf(d1) - K * norm.cdf(d2)),
            df * (K * norm.cdf(-d2) - F * norm.cdf(-d1)),
        )

        vega = df * F * sqrt_tau * norm.pdf(d1)
        diff_new = model_price - price
        sigma_new = sigma - diff_new / jnp.maximum(vega, 1e-30)
        sigma_new = jnp.clip(sigma_new, 1e-6, 10.0)
        return sigma_new, diff_new, i + 1

    def cond_fn(carry: tuple) -> bool:
        _, diff, i = carry
        return (i < max_iter) & (jnp.abs(diff) > tol)

    sigma, _, _ = jax.lax.while_loop(cond_fn, body_fn, (sigma, diff, 0))
    return sigma


# ─── SABR Model ───


class SabrParams(NamedTuple):
    """SABR model parameters.

    Attributes:
        alpha: Initial vol.
        beta: CEV exponent (usually fixed).
        rho: Correlation.
        nu: Vol of vol.
    """

    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> Float[Array, ""]:
    """Hagan's SABR implied volatility approximation.

    σ_B(K,F) ≈ α / (FK)^((1-β)/2) × z/x(z) × (1 + corrections)

    Args:
        F: Forward price.
        K: Strike price.
        T: Time to expiry.
        alpha: Initial vol.
        beta: CEV exponent.
        rho: Correlation.
        nu: Vol of vol.

    Returns:
        SABR implied Black volatility.
    """
    F = jnp.asarray(F, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)

    FK = F * K
    one_minus_beta = 1.0 - beta
    FK_beta = jnp.power(FK, one_minus_beta / 2.0)

    logFK = jnp.log(F / K)
    logFK2 = logFK**2

    # Correction terms
    A = one_minus_beta**2 / 24.0 * alpha**2 / FK_beta**2
    B = 0.25 * rho * beta * nu * alpha / FK_beta
    C = (2.0 - 3.0 * rho**2) * nu**2 / 24.0

    # z/x(z) ratio
    z = nu / alpha * FK_beta * logFK
    sqrt_term = jnp.sqrt(1.0 - 2.0 * rho * z + z**2)
    x_z = jnp.log((sqrt_term + z - rho) / (1.0 - rho))

    # Handle ATM (z→0): z/x(z) → 1
    zx_ratio = jnp.where(jnp.abs(z) < 1e-10, 1.0, z / x_z)

    # Base vol with log(F/K) correction
    vol_base = alpha / (
        FK_beta
        * (
            1.0
            + one_minus_beta**2 / 24.0 * logFK2
            + one_minus_beta**4 / 1920.0 * logFK2**2
        )
    )

    # Time correction factor
    correction = 1.0 + (A + B + C) * T

    return vol_base * zx_ratio * correction


# ─── Bond Pricing ───


def bond_price(
    cashflows: Float[Array, n],
    times: Float[Array, n],
    discount_fn: Callable,
) -> Float[Array, ""]:
    """Price a bond given cashflows and a discount function.

    P = Σᵢ cᵢ × D(tᵢ)

    Args:
        cashflows: Array of cashflow amounts.
        times: Array of cashflow times.
        discount_fn: Function mapping time → discount factor.

    Returns:
        Present value of cashflows.
    """
    discounts = jax.vmap(discount_fn)(times)
    return jnp.sum(cashflows * discounts)


def bond_yield(
    price: float,
    cashflows: Float[Array, n],
    times: Float[Array, n],
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Float[Array, ""]:
    """Solve for yield-to-maturity using Newton-Raphson.

    Finds y such that Σᵢ cᵢ × exp(-y×tᵢ) = price.

    Args:
        price: Target bond price.
        cashflows: Array of cashflow amounts.
        times: Array of cashflow times.
        max_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance for price error.

    Returns:
        Continuously compounded yield to maturity.
    """
    cashflows = jnp.asarray(cashflows, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)
    price = jnp.asarray(price, dtype=jnp.float64)
    y = jnp.asarray(0.05, dtype=jnp.float64)
    diff = jnp.asarray(jnp.inf, dtype=jnp.float64)

    def body_fn(carry: tuple) -> tuple:
        y, diff, i = carry
        discounts = jnp.exp(-y * times)
        pv = jnp.sum(cashflows * discounts)
        dpv = -jnp.sum(cashflows * times * discounts)
        diff_new = pv - price
        y_new = y - diff_new / dpv
        return y_new, diff_new, i + 1

    def cond_fn(carry: tuple) -> bool:
        _, diff, i = carry
        return (i < max_iter) & (jnp.abs(diff) > tol)

    y, _, _ = jax.lax.while_loop(cond_fn, body_fn, (y, diff, 0))
    return y


def bond_duration(
    price: float,
    cashflows: Float[Array, n],
    times: Float[Array, n],
    yield_: float,
) -> Float[Array, ""]:
    """Macaulay duration (continuously compounded).

    D = (1/P) × Σᵢ tᵢ × cᵢ × exp(-y×tᵢ)

    Args:
        price: Bond price.
        cashflows: Array of cashflow amounts.
        times: Array of cashflow times.
        yield_: Continuously compounded yield.

    Returns:
        Macaulay duration.
    """
    cashflows = jnp.asarray(cashflows, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)
    discounts = jnp.exp(-yield_ * times)
    return jnp.sum(times * cashflows * discounts) / price


def bond_convexity(
    price: float,
    cashflows: Float[Array, n],
    times: Float[Array, n],
    yield_: float,
) -> Float[Array, ""]:
    """Bond convexity (continuously compounded).

    Conv = (1/P) × Σᵢ tᵢ² × cᵢ × exp(-y×tᵢ)

    Args:
        price: Bond price.
        cashflows: Array of cashflow amounts.
        times: Array of cashflow times.
        yield_: Continuously compounded yield.

    Returns:
        Bond convexity.
    """
    cashflows = jnp.asarray(cashflows, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)
    discounts = jnp.exp(-yield_ * times)
    return jnp.sum(times**2 * cashflows * discounts) / price
