"""Akima yield curve calibration pipeline.

Builds Akima spline yield curves from spot rates at standard knot
points, derives forward and ZCB price curves, and supports real
yield curve construction via the Fisher equation.

Matches the C# ``AkimaCalibration`` pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.curves.splines import AkimaCubicSpline
from hyesg.math.transforms import spot_to_forward, spot_to_zcbp

# Standard 15 knot points for UK gilt/swap curves (years).
YIELD_CURVE_KNOTS: tuple[float, ...] = (
    0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
)


def build_akima_yield_curve(
    spot_rates: Sequence[float],
    knots: Sequence[float] = YIELD_CURVE_KNOTS,
    extrapolation: str = "flat",
) -> ParametricCurve:
    """Build an Akima spline yield curve from spot rates at knot points.

    Appends two flat-tail extrapolation points at 95y and 100y using
    the value at the last knot to ensure stable long-end behaviour.

    Args:
        spot_rates: Spot rates at each knot point.
        knots: Maturity knot points (default: standard 15-point grid).
        extrapolation: Extrapolation method (``'flat'`` supported).

    Returns:
        Akima spline spot-rate curve.

    Raises:
        ValueError: If ``spot_rates`` and ``knots`` have different lengths,
            or an unsupported extrapolation method is requested.
    """
    knots_list = list(knots)
    rates_list = list(spot_rates)

    if len(knots_list) != len(rates_list):
        raise ValueError(
            f"knots ({len(knots_list)}) and spot_rates ({len(rates_list)}) "
            "must have the same length"
        )

    if extrapolation != "flat":
        raise ValueError(f"Unsupported extrapolation method: {extrapolation!r}")

    # Append two flat-tail extrapolation points beyond the last knot.
    last_rate = rates_list[-1]
    last_knot = knots_list[-1]
    knots_list.extend([last_knot + 5.0, last_knot + 10.0])
    rates_list.extend([last_rate, last_rate])

    return AkimaCubicSpline(knots_list, rates_list)


def fisher_real_rate(nominal_spot: float, inflation_spot: float) -> float:
    """Derive the real spot rate from nominal and inflation via Fisher.

    ``s_real = (1 + s_nominal) / (1 + s_inflation) - 1``

    Args:
        nominal_spot: Nominal spot rate.
        inflation_spot: Inflation spot rate.

    Returns:
        Real spot rate.
    """
    return (1.0 + nominal_spot) / (1.0 + inflation_spot) - 1.0


def build_real_yield_curve(
    nominal_curve: ParametricCurve,
    inflation_curve: ParametricCurve,
    knots: Sequence[float] = YIELD_CURVE_KNOTS,
) -> ParametricCurve:
    """Derive a real yield curve from nominal and inflation curves.

    Evaluates both curves at the knot points, applies the Fisher
    equation at each knot, then fits an Akima spline through the
    resulting real rates.

    Args:
        nominal_curve: Nominal spot-rate curve.
        inflation_curve: Inflation spot-rate curve.
        knots: Maturity knot points.

    Returns:
        Akima spline real spot-rate curve.
    """
    knots_list = list(knots)
    real_rates = [
        fisher_real_rate(
            nominal_curve.evaluate(t),
            inflation_curve.evaluate(t),
        )
        for t in knots_list
    ]
    return build_akima_yield_curve(real_rates, knots_list)


class YieldCurveCalibrationResult(NamedTuple):
    """Result of yield curve calibration.

    Attributes:
        spot_curve: Akima spline spot-rate curve.
        forward_curve: Instantaneous forward-rate curve derived from spot.
        zcbp_curve: Zero-coupon bond price curve derived from spot.
        residuals: Calibration residuals at integer maturities 1–100.
    """

    spot_curve: ParametricCurve
    forward_curve: ParametricCurve
    zcbp_curve: ParametricCurve
    residuals: Array


def calibrate_yield_curve(
    spot_rates: Sequence[float],
    knots: Sequence[float] = YIELD_CURVE_KNOTS,
    residual_tolerance: float = 1e-10,
) -> YieldCurveCalibrationResult:
    """Full calibration: build Akima spot curve, derive forward and ZCB.

    1. Builds an Akima spline spot curve from the given spot rates.
    2. Derives forward and ZCB price curves via the transforms in
       ``hyesg.math.transforms``.
    3. Validates residuals at each knot point are below *residual_tolerance*.

    Args:
        spot_rates: Spot rates at each knot point.
        knots: Maturity knot points (default: standard 15-point grid).
        residual_tolerance: Maximum acceptable absolute residual at
            knot points.

    Returns:
        A :class:`YieldCurveCalibrationResult` containing the three
        curve representations and the residual vector.

    Raises:
        ValueError: If any knot-point residual exceeds *residual_tolerance*.
    """
    knots_list = list(knots)
    rates_list = list(spot_rates)

    # Build the spot curve.
    spot_curve = build_akima_yield_curve(rates_list, knots_list)

    # Derive forward and ZCB price curves.
    forward_curve = spot_to_forward(spot_curve)
    zcbp_curve = spot_to_zcbp(spot_curve)

    # Compute residuals at integer maturities 1–100.
    maturities = list(range(1, 101))
    evaluated = jnp.array(
        [spot_curve.evaluate(t) for t in maturities],
        dtype=jnp.float64,
    )

    # Build a reference array: at knot maturities use the input rate,
    # at non-knot maturities use the spline value (residual = 0).
    knot_map = {int(k): r for k, r in zip(knots_list, rates_list) if k >= 1}
    reference = jnp.array(
        [
            knot_map[t] if t in knot_map else spot_curve.evaluate(t)
            for t in maturities
        ],
        dtype=jnp.float64,
    )

    residuals = evaluated - reference

    # Validate at knot maturities only.
    max_residual = max(
        abs(spot_curve.evaluate(k) - r)
        for k, r in zip(knots_list, rates_list)
        if k >= 0
    )
    if max_residual > residual_tolerance:
        raise ValueError(
            f"Calibration residual {max_residual:.2e} exceeds tolerance "
            f"{residual_tolerance:.2e}"
        )

    return YieldCurveCalibrationResult(
        spot_curve=spot_curve,
        forward_curve=forward_curve,
        zcbp_curve=zcbp_curve,
        residuals=residuals,
    )


def _zcbp_from_spot(spot_rate: float, maturity: float) -> float:
    """ZCB price from continuous spot rate: P(t) = exp(-s(t) * t).

    Args:
        spot_rate: Continuous spot rate.
        maturity: Time to maturity.

    Returns:
        Zero-coupon bond price.
    """
    return math.exp(-spot_rate * maturity)
