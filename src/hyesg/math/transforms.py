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


def forward_to_spot(fwd: ParametricCurve) -> ParametricCurve:
    """Convert forward rate curve to spot rate curve.

    spot(t) = (1/t) ∫₀ᵗ f(u)du

    Args:
        fwd: Forward rate curve.

    Returns:
        Spot rate curve.
    """
    return fwd.integrate_curve() / LinearCurve()


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

    spot(t) = -ln P(t) / t

    Args:
        zcbp: Zero-coupon bond price curve.

    Returns:
        Spot rate curve.
    """
    return -(zcbp.log()) / LinearCurve()


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

    spot(t) = ln(1/P(t)) / t

    Args:
        inv: Inverse ZCB price (accumulation factor) curve.

    Returns:
        Spot rate curve.
    """
    return inv.log() / LinearCurve()


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
