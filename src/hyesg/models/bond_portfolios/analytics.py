"""Bond-specific analytics — pure JAX functions.

Duration, convexity, yield-to-maturity, z-spread, and total return
calculations. All functions are pure, stateless, and JIT-compatible.
"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    from collections.abc import Callable

from hyesg.models.bond_portfolios import pricing as _pricing


def macaulay_duration(
    spot_curve_fn: Callable[[float], jax.Array],
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
) -> jax.Array:
    """Macaulay duration of a coupon bond.

    Weighted-average time of cash-flows, where weights are the
    present-value fractions.

    Args:
        spot_curve_fn: Maps time-to-maturity to spot rate.
        coupon_rate: Annual coupon rate.
        maturity: Time to maturity in years.
        frequency: Coupon payments per year.

    Returns:
        Macaulay duration in years.
    """
    if frequency <= 0 or coupon_rate == 0.0:
        return jnp.asarray(maturity, dtype=jnp.float64)

    coupon_per_period = coupon_rate / frequency
    dt = 1.0 / frequency
    n_periods = int(maturity * frequency)

    pv_weighted_time = jnp.asarray(0.0, dtype=jnp.float64)
    total_pv = jnp.asarray(0.0, dtype=jnp.float64)

    for i in range(1, n_periods + 1):
        t_i = i * dt
        rate_i = spot_curve_fn(t_i)
        df_i = jnp.exp(-jnp.asarray(rate_i, dtype=jnp.float64) * t_i)

        cf = coupon_per_period
        if i == n_periods:
            cf = cf + 1.0  # principal repayment

        pv_i = cf * df_i
        pv_weighted_time = pv_weighted_time + t_i * pv_i
        total_pv = total_pv + pv_i

    return pv_weighted_time / total_pv


def modified_duration(
    spot_curve_fn: Callable[[float], jax.Array],
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
) -> jax.Array:
    """Modified duration = Macaulay / (1 + y / freq).

    Uses the bond's yield-to-maturity for the adjustment.

    Args:
        spot_curve_fn: Maps time-to-maturity to spot rate.
        coupon_rate: Annual coupon rate.
        maturity: Time to maturity in years.
        frequency: Coupon payments per year.

    Returns:
        Modified duration in years.
    """
    mac_dur = macaulay_duration(spot_curve_fn, coupon_rate, maturity, frequency)
    price = _pricing.coupon_bond_price(spot_curve_fn, coupon_rate, maturity, frequency)
    ytm = yield_to_maturity(price, coupon_rate, maturity, frequency=frequency)
    freq = max(frequency, 1)
    return mac_dur / (1.0 + ytm / freq)


def convexity(
    spot_curve_fn: Callable[[float], jax.Array],
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
) -> jax.Array:
    """Bond convexity.

    Sum of t_i * (t_i + dt) * PV(CF_i) / (Price * (1 + y/freq)^2).

    Args:
        spot_curve_fn: Maps time-to-maturity to spot rate.
        coupon_rate: Annual coupon rate.
        maturity: Time to maturity in years.
        frequency: Coupon payments per year.

    Returns:
        Convexity as a JAX scalar.
    """
    if frequency <= 0 or coupon_rate == 0.0:
        return jnp.asarray(maturity * maturity, dtype=jnp.float64)

    coupon_per_period = coupon_rate / frequency
    dt = 1.0 / frequency
    n_periods = int(maturity * frequency)

    price = _pricing.coupon_bond_price(spot_curve_fn, coupon_rate, maturity, frequency)
    ytm = yield_to_maturity(price, coupon_rate, maturity, frequency=frequency)
    freq = max(frequency, 1)
    discount_factor = (1.0 + ytm / freq) ** 2

    conv = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(1, n_periods + 1):
        t_i = i * dt
        rate_i = spot_curve_fn(t_i)
        df_i = jnp.exp(-jnp.asarray(rate_i, dtype=jnp.float64) * t_i)

        cf = coupon_per_period
        if i == n_periods:
            cf = cf + 1.0

        pv_i = cf * df_i
        conv = conv + t_i * (t_i + dt) * pv_i

    return conv / (price * discount_factor)


def yield_to_maturity(
    price: jax.Array,
    coupon_rate: float,
    maturity: float,
    face: float = 1.0,
    frequency: int = 2,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> jax.Array:
    """Yield to maturity via Newton's method.

    Finds y such that the present-value of cash-flows equals ``price``.

    Args:
        price: Observed bond price.
        coupon_rate: Annual coupon rate.
        maturity: Time to maturity in years.
        face: Face / par value.
        frequency: Coupon payments per year.
        tol: Convergence tolerance.
        max_iter: Maximum Newton iterations.

    Returns:
        Yield to maturity (annualised, continuously compounded).
    """
    price = jnp.asarray(price, dtype=jnp.float64)

    if frequency <= 0 or coupon_rate == 0.0:
        # ZCB: P = face * exp(-y * T)  =>  y = -ln(P / face) / T
        safe_price = jnp.clip(price, min=1e-15)
        return -jnp.log(safe_price / face) / maturity

    coupon_per_period = coupon_rate * face / frequency
    dt = 1.0 / frequency
    n_periods = int(maturity * frequency)

    # Initial guess from ZCB-equivalent yield
    safe_price = jnp.clip(price, min=1e-15)
    y = -jnp.log(safe_price / face) / maturity

    for _ in range(max_iter):
        pv = jnp.asarray(0.0, dtype=jnp.float64)
        dpv = jnp.asarray(0.0, dtype=jnp.float64)

        for i in range(1, n_periods + 1):
            t_i = i * dt
            df_i = jnp.exp(-y * t_i)

            cf = coupon_per_period
            if i == n_periods:
                cf = cf + face

            pv = pv + cf * df_i
            dpv = dpv - t_i * cf * df_i

        residual = pv - price
        # Avoid division by zero
        dpv = jnp.where(jnp.abs(dpv) < 1e-30, jnp.asarray(-1e-30), dpv)
        step = residual / dpv
        y = y - step

        converged = jnp.abs(residual) < tol
        y = jnp.where(converged, y, y)

    return y


def z_spread(
    bond_price: jax.Array,
    spot_curve_fn: Callable[[float], jax.Array],
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
    face: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> jax.Array:
    """Z-spread over the risk-free curve.

    Finds z such that discounting cash-flows at r(t_i) + z recovers
    the observed bond price.

    Args:
        bond_price: Observed bond price.
        spot_curve_fn: Maps time-to-maturity to risk-free spot rate.
        coupon_rate: Annual coupon rate.
        maturity: Time to maturity in years.
        frequency: Coupon payments per year.
        face: Face / par value.
        tol: Convergence tolerance.
        max_iter: Maximum Newton iterations.

    Returns:
        Z-spread as a JAX scalar.
    """
    bond_price = jnp.asarray(bond_price, dtype=jnp.float64)

    if frequency <= 0 or coupon_rate == 0.0:
        rate_m = spot_curve_fn(maturity)
        implied_rate = -jnp.log(jnp.clip(bond_price, min=1e-15) / face) / maturity
        return implied_rate - jnp.asarray(rate_m, dtype=jnp.float64)

    coupon_per_period = coupon_rate * face / frequency
    dt = 1.0 / frequency
    n_periods = int(maturity * frequency)

    # Cache spot rates
    spot_rates: list[jax.Array] = []
    for i in range(1, n_periods + 1):
        t_i = i * dt
        spot_rates.append(jnp.asarray(spot_curve_fn(t_i), dtype=jnp.float64))

    z = jnp.asarray(0.01, dtype=jnp.float64)  # initial guess

    for _ in range(max_iter):
        pv = jnp.asarray(0.0, dtype=jnp.float64)
        dpv = jnp.asarray(0.0, dtype=jnp.float64)

        for i in range(n_periods):
            t_i = (i + 1) * dt
            total_rate = spot_rates[i] + z
            df_i = jnp.exp(-total_rate * t_i)

            cf = coupon_per_period
            if i == n_periods - 1:
                cf = cf + face

            pv = pv + cf * df_i
            dpv = dpv - t_i * cf * df_i

        residual = pv - bond_price
        dpv = jnp.where(jnp.abs(dpv) < 1e-30, jnp.asarray(-1e-30), dpv)
        step = residual / dpv
        z = z - step

    return z


def total_return(
    price_start: jax.Array,
    price_end: jax.Array,
    coupon_income: jax.Array,
) -> jax.Array:
    """Total return = (price_end - price_start + income) / price_start.

    Args:
        price_start: Bond price at start of period.
        price_end: Bond price at end of period.
        coupon_income: Coupon income received during the period.

    Returns:
        Total return as a JAX scalar.
    """
    price_start = jnp.asarray(price_start, dtype=jnp.float64)
    price_end = jnp.asarray(price_end, dtype=jnp.float64)
    coupon_income = jnp.asarray(coupon_income, dtype=jnp.float64)

    safe_start = jnp.clip(price_start, min=1e-15)
    return (price_end - price_start + coupon_income) / safe_start


__all__ = [
    "convexity",
    "macaulay_duration",
    "modified_duration",
    "total_return",
    "yield_to_maturity",
    "z_spread",
]
