"""Bond pricing engine — pure JAX functions.

All functions are pure, stateless, and operate on JAX arrays.
They are designed to be JIT-compiled via ``jax.jit``.
"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def zcb_price(spot_rate: jax.Array, maturity: float) -> jax.Array:
    """Zero-coupon bond price from a continuously-compounded spot rate.

    P(T) = exp(-r * T)

    Args:
        spot_rate: Continuously-compounded spot rate to maturity.
        maturity: Time to maturity in years.

    Returns:
        ZCB price as a JAX scalar.
    """
    return jnp.exp(-jnp.asarray(spot_rate, dtype=jnp.float64) * maturity)


def coupon_bond_price(
    spot_curve_fn: Callable[[float], jax.Array],
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
) -> jax.Array:
    """Risk-free coupon bond price by discounting each cash-flow.

    Args:
        spot_curve_fn: Maps time-to-maturity to spot rate.
        coupon_rate: Annual coupon rate.
        maturity: Time to maturity in years.
        frequency: Coupon payments per year.

    Returns:
        Bond dirty price as a JAX scalar.
    """
    if frequency <= 0 or coupon_rate == 0.0:
        rate = spot_curve_fn(maturity)
        return zcb_price(rate, maturity)

    coupon_per_period = coupon_rate / frequency
    dt = 1.0 / frequency
    n_periods = int(maturity * frequency)

    price = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(1, n_periods + 1):
        t_i = i * dt
        rate_i = spot_curve_fn(t_i)
        df_i = jnp.exp(-jnp.asarray(rate_i, dtype=jnp.float64) * t_i)
        price = price + coupon_per_period * df_i

    # Add principal repayment at maturity
    rate_m = spot_curve_fn(maturity)
    df_m = jnp.exp(-jnp.asarray(rate_m, dtype=jnp.float64) * maturity)
    price = price + df_m

    return price


def credit_bond_price(
    spot_curve_fn: Callable[[float], jax.Array],
    survival_fn: Callable[[float], jax.Array],
    coupon_rate: float,
    maturity: float,
    recovery_rate: float,
    frequency: int = 2,
) -> jax.Array:
    """Credit-risky coupon bond price.

    Discounts each cash-flow by both the risk-free rate and survival
    probability.  Adds a recovery leg for default events.

    Args:
        spot_curve_fn: Maps time-to-maturity to risk-free spot rate.
        survival_fn: Maps time-to-maturity to survival probability Q(0, t).
        coupon_rate: Annual coupon rate.
        maturity: Time to maturity in years.
        recovery_rate: Expected recovery fraction on default.
        frequency: Coupon payments per year.

    Returns:
        Credit-risky bond price.
    """
    if frequency <= 0 or coupon_rate == 0.0:
        rate = spot_curve_fn(maturity)
        surv = survival_fn(maturity)
        df = jnp.exp(-jnp.asarray(rate, dtype=jnp.float64) * maturity)
        return jnp.asarray(surv, dtype=jnp.float64) * df + recovery_rate * (
            1.0 - jnp.asarray(surv, dtype=jnp.float64)
        ) * df

    coupon_per_period = coupon_rate / frequency
    dt = 1.0 / frequency
    n_periods = int(maturity * frequency)

    price = jnp.asarray(0.0, dtype=jnp.float64)
    prev_surv = jnp.asarray(1.0, dtype=jnp.float64)

    for i in range(1, n_periods + 1):
        t_i = i * dt
        rate_i = spot_curve_fn(t_i)
        df_i = jnp.exp(-jnp.asarray(rate_i, dtype=jnp.float64) * t_i)
        surv_i = jnp.asarray(survival_fn(t_i), dtype=jnp.float64)

        # Coupon paid conditional on survival
        price = price + coupon_per_period * df_i * surv_i

        # Recovery on default in this period
        default_prob = prev_surv - surv_i
        price = price + recovery_rate * df_i * default_prob

        prev_surv = surv_i

    # Principal at maturity conditional on survival
    rate_m = spot_curve_fn(maturity)
    df_m = jnp.exp(-jnp.asarray(rate_m, dtype=jnp.float64) * maturity)
    surv_m = jnp.asarray(survival_fn(maturity), dtype=jnp.float64)
    price = price + df_m * surv_m

    return price


def index_linked_bond_price(
    real_spot_curve_fn: Callable[[float], jax.Array],
    inflation_index_fn: Callable[[float], jax.Array],
    survival_fn: Callable[[float], jax.Array],
    coupon_rate: float,
    maturity: float,
    recovery_rate: float,
    frequency: int = 2,
) -> jax.Array:
    """Index-linked credit bond price.

    Each cash-flow is inflation-adjusted via the inflation index ratio
    and discounted using the real spot curve.

    Args:
        real_spot_curve_fn: Maps time to real spot rate.
        inflation_index_fn: Maps time to inflation index level I(t).
        survival_fn: Maps time to survival probability.
        coupon_rate: Annual real coupon rate.
        maturity: Time to maturity in years.
        recovery_rate: Expected recovery fraction on default.
        frequency: Coupon payments per year.

    Returns:
        Index-linked bond price.
    """
    base_index = jnp.asarray(inflation_index_fn(0.0), dtype=jnp.float64)

    if frequency <= 0 or coupon_rate == 0.0:
        rate = real_spot_curve_fn(maturity)
        df = jnp.exp(-jnp.asarray(rate, dtype=jnp.float64) * maturity)
        surv = jnp.asarray(survival_fn(maturity), dtype=jnp.float64)
        idx_ratio = (
            jnp.asarray(inflation_index_fn(maturity), dtype=jnp.float64) / base_index
        )
        return idx_ratio * (surv * df + recovery_rate * (1.0 - surv) * df)

    coupon_per_period = coupon_rate / frequency
    dt = 1.0 / frequency
    n_periods = int(maturity * frequency)

    price = jnp.asarray(0.0, dtype=jnp.float64)
    prev_surv = jnp.asarray(1.0, dtype=jnp.float64)

    for i in range(1, n_periods + 1):
        t_i = i * dt
        rate_i = real_spot_curve_fn(t_i)
        df_i = jnp.exp(-jnp.asarray(rate_i, dtype=jnp.float64) * t_i)
        surv_i = jnp.asarray(survival_fn(t_i), dtype=jnp.float64)
        idx_ratio = (
            jnp.asarray(inflation_index_fn(t_i), dtype=jnp.float64) / base_index
        )

        # Inflation-adjusted coupon
        price = price + coupon_per_period * idx_ratio * df_i * surv_i

        # Recovery on default
        default_prob = prev_surv - surv_i
        price = price + recovery_rate * idx_ratio * df_i * default_prob

        prev_surv = surv_i

    # Inflation-adjusted principal at maturity
    rate_m = real_spot_curve_fn(maturity)
    df_m = jnp.exp(-jnp.asarray(rate_m, dtype=jnp.float64) * maturity)
    surv_m = jnp.asarray(survival_fn(maturity), dtype=jnp.float64)
    idx_ratio_m = (
        jnp.asarray(inflation_index_fn(maturity), dtype=jnp.float64) / base_index
    )
    price = price + idx_ratio_m * df_m * surv_m

    return price


def basket_yield(
    bond_prices: jax.Array,
    weights: jax.Array,
    maturities: jax.Array,
) -> jax.Array:
    """Portfolio-weighted basket yield.

    Computes the weighted-average implied yield from ZCB-equivalent
    prices: y_i = -ln(P_i) / T_i.

    Args:
        bond_prices: Array of bond prices.  shape: (n_bonds,)
        weights: Array of portfolio weights.  shape: (n_bonds,)
        maturities: Array of maturities in years.  shape: (n_bonds,)

    Returns:
        Basket yield as a JAX scalar.
    """
    bond_prices = jnp.asarray(bond_prices, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    maturities = jnp.asarray(maturities, dtype=jnp.float64)

    # Clamp prices to avoid log(0)
    safe_prices = jnp.clip(bond_prices, min=1e-15)
    implied_yields = -jnp.log(safe_prices) / jnp.clip(maturities, min=1e-10)

    return jnp.sum(weights * implied_yields)


__all__ = [
    "basket_yield",
    "coupon_bond_price",
    "credit_bond_price",
    "index_linked_bond_price",
    "zcb_price",
]
