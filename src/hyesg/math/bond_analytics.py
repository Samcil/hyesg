"""Bond analytics — coupon schedules, YTM, duration, convexity, DV01.

Discrete (periodically compounded) bond analytics for use in the portfolio
system.  These complement the continuously-compounded helpers in
``hyesg.math.pricing`` and are designed for coupon-bearing bonds with
standard market conventions (semi-annual, quarterly, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Sequence


class BondMetrics(NamedTuple):
    """Container for bond analytics.

    Attributes:
        price: Clean bond price.
        ytm: Yield to maturity (periodically compounded).
        duration: Modified duration.
        convexity: Bond convexity.
        spread: Spread over risk-free rate.
        dv01: Dollar value of one basis point.
    """

    price: Array
    ytm: Array
    duration: Array
    convexity: Array
    spread: Array
    dv01: Array


# ─── Coupon Schedule Helpers ───


def future_coupons(maturity: float, coupon_freq: int, t: float) -> list[float]:
    """Coupon payment dates strictly after time *t*.

    Works backwards from *maturity* in steps of ``1/coupon_freq``.

    Args:
        maturity: Bond maturity in years.
        coupon_freq: Number of coupon payments per year.
        t: Current time in years.

    Returns:
        Sorted list of coupon dates after *t*.
    """
    period = 1.0 / coupon_freq
    dates: list[float] = []
    d = maturity
    while d > t + 1e-10:
        dates.append(d)
        d -= period
    return sorted(dates)


def elapsed_coupons(maturity: float, coupon_freq: int, t: float) -> list[float]:
    """Coupon payment dates on or before time *t*.

    Args:
        maturity: Bond maturity in years.
        coupon_freq: Number of coupon payments per year.
        t: Current time in years.

    Returns:
        Sorted list of coupon dates at or before *t*.
    """
    period = 1.0 / coupon_freq
    dates: list[float] = []
    d = maturity
    while d > 1e-10:
        if d <= t + 1e-10:
            dates.append(d)
        d -= period
    return sorted(dates)


def next_coupon(maturity: float, coupon_freq: int, t: float) -> float:
    """Next coupon date strictly after *t*.

    Args:
        maturity: Bond maturity in years.
        coupon_freq: Number of coupon payments per year.
        t: Current time in years.

    Returns:
        The nearest future coupon date, or *maturity* if none remain.
    """
    fc = future_coupons(maturity, coupon_freq, t)
    return fc[0] if fc else maturity


# ─── Cash-Flow Construction ───


def _build_cashflows(
    face: float,
    coupon: float,
    maturity: float,
    freq: int,
    t: float,
) -> tuple[list[float], list[float]]:
    """Build remaining cash-flow amounts and times from *t*.

    Args:
        face: Face / par value.
        coupon: Annual coupon rate (e.g. 0.05 for 5%).
        maturity: Bond maturity in years.
        freq: Coupon payments per year.
        t: Current time.

    Returns:
        ``(cash_flows, times)`` — parallel lists.
    """
    cpn_amount = face * coupon / freq
    dates = future_coupons(maturity, freq, t)
    cfs: list[float] = []
    for d in dates:
        cf = cpn_amount
        if abs(d - maturity) < 1e-10:
            cf += face
        cfs.append(cf)
    return cfs, dates


# ─── Price From Yield ───


def _price_from_yield(
    ytm: float,
    cash_flows: Sequence[float],
    times: Sequence[float],
    t: float = 0.0,
) -> float:
    """Dirty price given a periodically-compounded yield.

    P = Σ cf_i / (1 + y)^(t_i - t)

    Args:
        ytm: Yield to maturity (annualised, periodically compounded).
        cash_flows: Cash-flow amounts.
        times: Cash-flow dates.
        t: Valuation time.

    Returns:
        Present value of cash flows.
    """
    pv = 0.0
    for cf, ti in zip(cash_flows, times, strict=True):
        pv += cf / (1.0 + ytm) ** (ti - t)
    return pv


# ─── YTM Solver ───


def yield_to_maturity(
    price: float,
    face: float,
    coupon: float,
    maturity: float,
    freq: int,
    t: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson yield-to-maturity solver.

    Finds *y* such that ``price = Σ cf_i / (1+y)^(t_i - t)``.

    Args:
        price: Observed bond price.
        face: Face / par value.
        coupon: Annual coupon rate.
        maturity: Bond maturity in years.
        freq: Coupon payments per year.
        t: Current time.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Yield to maturity (annualised).
    """
    cfs, dates = _build_cashflows(face, coupon, maturity, freq, t)
    if not cfs:
        return 0.0

    y = coupon if coupon > 0 else 0.05  # initial guess

    for _ in range(max_iter):
        pv = 0.0
        dpv = 0.0
        for cf, ti in zip(cfs, dates, strict=True):
            tau = ti - t
            disc = (1.0 + y) ** tau
            pv += cf / disc
            dpv -= tau * cf / ((1.0 + y) * disc)

        diff = pv - price
        if abs(diff) < tol:
            break
        if abs(dpv) < 1e-30:
            break
        y -= diff / dpv

    return y


# ─── Duration ───


def macaulay_duration(
    ytm: float,
    cash_flows: Sequence[float],
    times: Sequence[float],
) -> float:
    """Macaulay duration.

    ``D_mac = Σ(t_i × PV(cf_i)) / price``

    Args:
        ytm: Yield to maturity.
        cash_flows: Cash-flow amounts.
        times: Cash-flow dates.

    Returns:
        Macaulay duration in years.
    """
    price = 0.0
    weighted = 0.0
    for cf, ti in zip(cash_flows, times, strict=True):
        pv = cf / (1.0 + ytm) ** ti
        price += pv
        weighted += ti * pv
    if abs(price) < 1e-30:
        return 0.0
    return weighted / price


def modified_duration(
    ytm: float,
    cash_flows: Sequence[float],
    times: Sequence[float],
) -> float:
    """Modified duration.

    ``D_mod = D_mac / (1 + y)``

    Args:
        ytm: Yield to maturity.
        cash_flows: Cash-flow amounts.
        times: Cash-flow dates.

    Returns:
        Modified duration in years.
    """
    mac = macaulay_duration(ytm, cash_flows, times)
    return mac / (1.0 + ytm)


# ─── Convexity ───


def convexity(
    ytm: float,
    cash_flows: Sequence[float],
    times: Sequence[float],
) -> float:
    """Bond convexity.

    ``Conv = Σ(t_i × (t_i + 1) × PV(cf_i)) / (P × (1+y)²)``

    Args:
        ytm: Yield to maturity.
        cash_flows: Cash-flow amounts.
        times: Cash-flow dates.

    Returns:
        Bond convexity.
    """
    price = 0.0
    weighted = 0.0
    for cf, ti in zip(cash_flows, times, strict=True):
        pv = cf / (1.0 + ytm) ** ti
        price += pv
        weighted += ti * (ti + 1.0) * pv
    if abs(price) < 1e-30:
        return 0.0
    return weighted / (price * (1.0 + ytm) ** 2)


# ─── DV01 ───


def dv01(
    ytm: float,
    cash_flows: Sequence[float],
    times: Sequence[float],
) -> float:
    """Dollar value of one basis point.

    ``DV01 = D_mod × P × 0.0001``

    Args:
        ytm: Yield to maturity.
        cash_flows: Cash-flow amounts.
        times: Cash-flow dates.

    Returns:
        DV01 (price change per 1 bp yield move).
    """
    price = _price_from_yield(ytm, cash_flows, times)
    mod_dur = modified_duration(ytm, cash_flows, times)
    return mod_dur * price * 0.0001


# ─── Convenience: full metrics ───


def compute_bond_metrics(
    price: float,
    face: float,
    coupon: float,
    maturity: float,
    freq: int,
    risk_free_yield: float = 0.0,
    t: float = 0.0,
) -> BondMetrics:
    """Compute all bond analytics in one call.

    Args:
        price: Market price.
        face: Face / par value.
        coupon: Annual coupon rate.
        maturity: Bond maturity.
        freq: Coupon frequency.
        risk_free_yield: Risk-free rate for spread calculation.
        t: Current time.

    Returns:
        A ``BondMetrics`` named tuple.
    """
    cfs, dates = _build_cashflows(face, coupon, maturity, freq, t)
    ytm_val = yield_to_maturity(price, face, coupon, maturity, freq, t)
    mac_dur = macaulay_duration(ytm_val, cfs, dates)
    mod_dur = mac_dur / (1.0 + ytm_val)
    conv = convexity(ytm_val, cfs, dates)
    spread = ytm_val - risk_free_yield
    dv01_val = mod_dur * price * 0.0001

    return BondMetrics(
        price=jnp.asarray(price, dtype=jnp.float64),
        ytm=jnp.asarray(ytm_val, dtype=jnp.float64),
        duration=jnp.asarray(mod_dur, dtype=jnp.float64),
        convexity=jnp.asarray(conv, dtype=jnp.float64),
        spread=jnp.asarray(spread, dtype=jnp.float64),
        dv01=jnp.asarray(dv01_val, dtype=jnp.float64),
    )
