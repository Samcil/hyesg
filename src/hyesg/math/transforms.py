"""Yield curve transformations.

All transforms take ParametricCurve and return ParametricCurve,
using the algebraic composition from the curve system.

Relationships:
    spot(t) = (1/t) ∫₀ᵗ f(u)du
    zcbp(t) = exp(-∫₀ᵗ f(u)du) = exp(-t × spot(t))
    forward(t) = -d/dt [ln P(t)] = spot(t) + t × spot'(t)
"""

from __future__ import annotations

import math

from hyesg.math.curves import (
    LinearCurve,
    ParametricCurve,
)


class _GuardedForwardToSpot(ParametricCurve):
    """Spot rate from forward: spot(t) = (1/t)∫₀ᵗ f(u)du, spot(0) = f(0)."""

    def __init__(self, fwd: ParametricCurve) -> None:
        self._fwd = fwd
        self._integrated = fwd.integrate_curve()

    def evaluate(self, x: float) -> float:
        if abs(x) < 1e-12:
            return self._fwd.evaluate(0.0)
        return self._integrated.evaluate(x) / x


class _GuardedZcbpToSpot(ParametricCurve):
    """Spot rate from ZCB prices: spot(t) = -ln(P(t))/t, spot(0) = f(0)."""

    def __init__(self, zcbp: ParametricCurve) -> None:
        self._zcbp = zcbp
        self._neg_log = -(zcbp.log())

    def evaluate(self, x: float) -> float:
        if abs(x) < 1e-12:
            # spot(0) = lim_{t→0} -ln(P(t))/t = f(0)
            # Use numerical derivative of -ln(P) at 0
            return self._neg_log.derivative(0.0)
        return self._neg_log.evaluate(x) / x


class _GuardedInvZcbpToSpot(ParametricCurve):
    """Spot rate from accumulation: spot(t) = ln(1/P(t))/t, spot(0) = f(0)."""

    def __init__(self, inv: ParametricCurve) -> None:
        self._inv = inv
        self._log_inv = inv.log()

    def evaluate(self, x: float) -> float:
        if abs(x) < 1e-12:
            return self._log_inv.derivative(0.0)
        return self._log_inv.evaluate(x) / x


def forward_to_spot(fwd: ParametricCurve) -> ParametricCurve:
    """Convert forward rate curve to spot rate curve.

    spot(t) = (1/t) ∫₀ᵗ f(u)du, with spot(0) = f(0) by L'Hôpital.

    Args:
        fwd: Forward rate curve.

    Returns:
        Spot rate curve.
    """
    return _GuardedForwardToSpot(fwd)


def forward_to_zcbp(fwd: ParametricCurve) -> ParametricCurve:
    """Convert forward rate curve to ZCB price curve.

    P(t) = exp(-∫₀ᵗ f(u)du)

    Args:
        fwd: Forward rate curve.

    Returns:
        Zero-coupon bond price curve.
    """
    return (-fwd.integrate_curve()).exp()


def forward_to_inverse_zcbp(fwd: ParametricCurve) -> ParametricCurve:
    """Convert forward rate curve to accumulation factor curve.

    1/P(t) = exp(∫₀ᵗ f(u)du)

    Args:
        fwd: Forward rate curve.

    Returns:
        Inverse ZCB price (accumulation factor) curve.
    """
    return fwd.integrate_curve().exp()


def spot_to_forward(spot: ParametricCurve) -> ParametricCurve:
    """Convert spot rate curve to forward rate curve.

    f(t) = spot(t) + t × spot'(t) = d/dt[t × spot(t)]

    Args:
        spot: Spot rate curve.

    Returns:
        Forward rate curve.
    """
    return (LinearCurve() * spot).differentiate()


def spot_to_zcbp(spot: ParametricCurve) -> ParametricCurve:
    """Convert spot rate curve to ZCB price curve.

    P(t) = exp(-t × spot(t))

    Args:
        spot: Spot rate curve.

    Returns:
        Zero-coupon bond price curve.
    """
    return (-(LinearCurve() * spot)).exp()


def zcbp_to_forward(zcbp: ParametricCurve) -> ParametricCurve:
    """Convert ZCB price curve to forward rate curve.

    f(t) = -d/dt[ln P(t)]

    Args:
        zcbp: Zero-coupon bond price curve.

    Returns:
        Forward rate curve.
    """
    return -(zcbp.log()).differentiate()


def zcbp_to_spot(zcbp: ParametricCurve) -> ParametricCurve:
    """Convert ZCB price curve to spot rate curve.

    spot(t) = -ln P(t) / t, with spot(0) = f(0) by L'Hôpital.

    Args:
        zcbp: Zero-coupon bond price curve.

    Returns:
        Spot rate curve.
    """
    return _GuardedZcbpToSpot(zcbp)


def inverse_zcbp_to_forward(inv: ParametricCurve) -> ParametricCurve:
    """Convert accumulation factor to forward rate curve.

    f(t) = d/dt[ln(1/P(t))]

    Args:
        inv: Inverse ZCB price (accumulation factor) curve.

    Returns:
        Forward rate curve.
    """
    return inv.log().differentiate()


def inverse_zcbp_to_spot(inv: ParametricCurve) -> ParametricCurve:
    """Convert accumulation factor to spot rate curve.

    spot(t) = ln(1/P(t)) / t, with spot(0) = f(0) by L'Hôpital.

    Args:
        inv: Inverse ZCB price (accumulation factor) curve.

    Returns:
        Spot rate curve.
    """
    return _GuardedInvZcbpToSpot(inv)


def change_compounding(
    rate: float,
    input_period: float,
    output_period: float,
) -> float:
    """Convert between compounding frequencies.

    Convention:
        period = 0   → continuous compounding
        period = 1   → annual compounding
        period = 0.5 → semi-annual compounding
        period = 1/12 → monthly compounding

    The key relationship: the annualised growth factor must be
    equal regardless of compounding convention:
        continuous:  exp(r)
        discrete:    (1 + r_d * period)^(1/period)

    Args:
        rate: The interest rate.
        input_period: Input compounding period (0 = continuous).
        output_period: Output compounding period (0 = continuous).

    Returns:
        Rate in the output compounding convention.
    """
    if input_period == 0.0 and output_period == 0.0:
        return rate

    if input_period == 0.0:
        # Continuous to discrete:
        # exp(r) = (1 + r_d * p)^(1/p)
        # r_d = (exp(r * p) - 1) / p
        return (math.exp(rate * output_period) - 1.0) / output_period

    if output_period == 0.0:
        # Discrete to continuous:
        # (1 + r * p)^(1/p) = exp(r_c)
        # r_c = ln(1 + r * p) / p
        return math.log(1.0 + rate * input_period) / input_period

    # Discrete to discrete:
    # (1 + r_in * p_in)^(1/p_in) = (1 + r_out * p_out)^(1/p_out)
    # Compute annualised growth factor from input
    growth_annual = (1.0 + rate * input_period) ** (1.0 / input_period)
    # Convert to output convention
    return (growth_annual**output_period - 1.0) / output_period


# ─── Scalar helpers ───


def continuously_compounded_to_zcbp(rate: float, t: float) -> float:
    """Convert continuously compounded rate to zero-coupon bond price.

    P(t) = exp(-rate × t)

    Args:
        rate: Continuously compounded interest rate.
        t: Time to maturity.

    Returns:
        Zero-coupon bond price.
    """
    return math.exp(-rate * t)


def annually_compounded_to_inv_zcbp(rate: float, t: float) -> float:
    """Convert annually compounded rate to accumulation factor.

    1/P(t) = (1 + rate)^t

    Args:
        rate: Annually compounded interest rate.
        t: Time to maturity.

    Returns:
        Inverse ZCB price (accumulation factor).
    """
    return (1.0 + rate) ** t


def spot_to_inverse_zcbp(spot: ParametricCurve) -> ParametricCurve:
    """Convert spot rate curve to accumulation factor curve.

    1/P(t) = exp(t × spot(t))

    This is the inverse of inverse_zcbp_to_spot.

    Args:
        spot: Spot rate curve.

    Returns:
        Inverse ZCB price (accumulation factor) curve.
    """
    return (LinearCurve() * spot).exp()
