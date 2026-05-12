"""Akima yield curve calibration pipeline.

Builds Akima spline yield curves from **forward rates** at standard knot
points, derives spot and ZCB price curves, and supports long-end 61→90y
extension, RPI reform blending, and CPI breakeven derivation.

Matches the C# ``AkimaYieldCurves`` / ``CalibrationRegime`` pipeline:

1. Fit Akima to **instantaneous forward rates** (not spot rates).
2. Derive spot via ``s(T) = (1/T) ∫₀ᵀ f(t) dt``.
3. Extend the long end by replacing knots ≥ ``transition_start`` with a
   single knot at ``transition_end`` set to the long-term target rate.
4. Blend RPI curves around the Feb-2030 reform date.
5. Derive CPI breakeven as RPI minus a blended RPI−CPI wedge.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from hyesg.math.curves.blending import ConstantExtrapolation, PolynomialBlend
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.curves.splines import AkimaCubicSpline, CubicSpline
from hyesg.math.transforms import (
    forward_to_spot,
    forward_to_zcbp,
    spot_to_forward,
    spot_to_zcbp,
)

# Standard 15 knot points for UK gilt/swap curves (years).
YIELD_CURVE_KNOTS: tuple[float, ...] = (
    0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
)

# Default long-end extension boundaries (C# targetTransitionStart/End).
_LONG_END_START: float = 61.0
_LONG_END_END: float = 90.0


# ── Forward-rate Akima pipeline ────────────────────────────────────


def build_forward_rate_curve(
    spot_rates: Sequence[float],
    knots: Sequence[float] = YIELD_CURVE_KNOTS,
    *,
    long_term_forward_rate: float | None = None,
    transition_start: float = _LONG_END_START,
    transition_end: float = _LONG_END_END,
) -> ParametricCurve:
    """Build an Akima spline **forward-rate** curve from spot-rate data.

    This is the pipeline matching C#
    ``AkimaYieldCurves.GetContinuouslyCompoundedInstantaneousForwardRateCurve
    FromContinuousSpotsData``:

    1. Convert input continuous spot rates to forward rates at each knot
       using ``f(t) = s(t) + t · s'(t)`` (exact derivative from a
       temporary Akima spot spline).
    2. If ``long_term_forward_rate`` is given, apply SetCustomLongTermLevel:
       discard knots ≥ ``transition_start``, append a single knot at
       ``transition_end`` with the target rate. The Akima spline
       naturally creates a smooth transition.
    3. Fit a final Akima spline through the forward-rate knots.

    Args:
        spot_rates: Continuously compounded spot rates at each knot.
        knots: Maturity knot points in years.
        long_term_forward_rate: Target long-term continuously compounded
            forward rate. If ``None``, no long-end extension is applied.
        transition_start: Maturity where long-end extension begins (default 61y).
        transition_end: Maturity for the long-term target knot (default 90y).

    Returns:
        Akima spline instantaneous forward-rate curve.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    knots_list = list(knots)
    rates_list = list(spot_rates)

    if len(knots_list) != len(rates_list):
        raise ValueError(
            f"knots ({len(knots_list)}) and spot_rates ({len(rates_list)}) "
            "must have the same length"
        )

    # Step 1: Convert spots to forwards.
    # Build a temporary Akima spline on spot rates to get s(t) and s'(t).
    # f(t) = s(t) + t * s'(t)
    temp_spot_spline = AkimaCubicSpline(knots_list, rates_list)
    fwd_rates = []
    for t, s in zip(knots_list, rates_list):
        if t == 0.0:
            # At t=0, f(0) = s(0) (limiting value).
            fwd_rates.append(s)
        else:
            s_prime = temp_spot_spline.derivative(t)
            fwd_rates.append(s + t * s_prime)

    # Step 2: Apply long-end extension if target is given.
    if long_term_forward_rate is not None:
        fwd_knots = [
            (k, f) for k, f in zip(knots_list, fwd_rates)
            if k < transition_start
        ]
        # Append a single knot at transition_end with the target rate.
        fwd_knots.append((transition_end, long_term_forward_rate))
        knots_list = [k for k, _ in fwd_knots]
        fwd_rates = [f for _, f in fwd_knots]

    # Step 3: Fit the final Akima spline to forward rates.
    return AkimaCubicSpline(knots_list, fwd_rates)


def build_akima_yield_curve(
    spot_rates: Sequence[float],
    knots: Sequence[float] = YIELD_CURVE_KNOTS,
    extrapolation: str = "flat",
) -> ParametricCurve:
    """Build an Akima spline yield curve from spot rates at knot points.

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


# ── Fisher real-rate helpers ───────────────────────────────────────


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


# ── Calibration result types ──────────────────────────────────────


class YieldCurveCalibrationResult(NamedTuple):
    """Result of yield curve calibration.

    Attributes:
        spot_curve: Spot-rate curve (derived from forward via integration).
        forward_curve: Akima spline instantaneous forward-rate curve.
        zcbp_curve: Zero-coupon bond price curve derived from forward.
        residuals: Calibration residuals at knot-point maturities.
    """

    spot_curve: ParametricCurve
    forward_curve: ParametricCurve
    zcbp_curve: ParametricCurve
    residuals: Array


# ── Main calibration entry point ──────────────────────────────────


def calibrate_yield_curve(
    spot_rates: Sequence[float],
    knots: Sequence[float] = YIELD_CURVE_KNOTS,
    *,
    long_term_forward_rate: float | None = None,
    transition_start: float = _LONG_END_START,
    transition_end: float = _LONG_END_END,
    residual_tolerance: float = 5e-3,
) -> YieldCurveCalibrationResult:
    """Full calibration: forward-rate Akima with long-end extension.

    Pipeline matching C# ``CalibrationRegime.GetMarketYieldCurvesViaAkima``:

    1. Builds a forward-rate Akima curve from continuous spot rates.
    2. Optionally applies long-end 61→90y extension to a target rate.
    3. Derives spot and ZCB price curves via analytical integration.
    4. Reports spot-rate recovery residuals as diagnostics.

    The forward curve is the canonical object. Spot rates recovered
    via ``(1/t)∫₀ᵗ f(u)du`` will not exactly match the inputs because
    the Akima forward spline's inter-knot behaviour differs from the
    derivative of the original spot Akima. Typical residuals are O(1e-3).

    Args:
        spot_rates: Continuously compounded spot rates at each knot.
        knots: Maturity knot points (default: standard 15-point grid).
        long_term_forward_rate: Target long-term forward rate (continuous).
            If ``None``, no long-end extension is applied.
        transition_start: Maturity where extension begins (default 61y).
        transition_end: Maturity for long-term target knot (default 90y).
        residual_tolerance: Maximum acceptable spot-rate recovery error
            at knot points. Defaults to 5e-3 which is appropriate for the
            forward-rate pipeline; tighter tolerances may not be achievable.

    Returns:
        A :class:`YieldCurveCalibrationResult` with forward, spot,
        and ZCB price curves plus residual diagnostics.

    Raises:
        ValueError: If knot-point residuals exceed tolerance for
            maturities before the long-end extension region.
    """
    knots_list = list(knots)
    rates_list = list(spot_rates)

    # Build the forward-rate curve.
    forward_curve = build_forward_rate_curve(
        spot_rates=rates_list,
        knots=knots_list,
        long_term_forward_rate=long_term_forward_rate,
        transition_start=transition_start,
        transition_end=transition_end,
    )

    # Derive spot and ZCB price curves from the forward curve.
    spot_curve = forward_to_spot(forward_curve)
    zcbp_curve = forward_to_zcbp(forward_curve)

    # Compute residuals at input knot maturities (before extension region).
    # After extension, knot points ≥ transition_start are no longer
    # expected to match the original spot rates.
    check_knots = [
        (k, r) for k, r in zip(knots_list, rates_list)
        if k > 0 and (
            long_term_forward_rate is None or k < transition_start
        )
    ]

    residual_values = []
    for k, r_input in check_knots:
        r_recovered = spot_curve.evaluate(k)
        residual_values.append(r_recovered - r_input)

    residuals = jnp.array(residual_values, dtype=jnp.float64)

    # Validate recovery at pre-extension knots.
    if len(residual_values) > 0:
        max_residual = max(abs(v) for v in residual_values)
        if max_residual > residual_tolerance:
            raise ValueError(
                f"Spot-rate recovery residual {max_residual:.2e} exceeds "
                f"tolerance {residual_tolerance:.2e}"
            )

    return YieldCurveCalibrationResult(
        spot_curve=spot_curve,
        forward_curve=forward_curve,
        zcbp_curve=zcbp_curve,
        residuals=residuals,
    )


# ── RPI reform blending ──────────────────────────────────────────


class PowerBlend(ParametricCurve):
    """Power-law blending from curve f to curve g over [t_start, t_end].

    Weight = ((x - t_start) / (t_end - t_start)) ^ strength.
    Matches C# ``PolynomialBlendingCurve(start, end, strength)``
    semantics where strength=1 → linear, strength=2 → quadratic.

    Args:
        f: The starting curve (returned for x ≤ t_start).
        g: The ending curve (returned for x ≥ t_end).
        t_start: Start of the blend region.
        t_end: End of the blend region.
        strength: Power exponent for the blend weight (default 1.0).
    """

    def __init__(
        self,
        f: ParametricCurve,
        g: ParametricCurve,
        t_start: float,
        t_end: float,
        strength: float = 1.0,
    ) -> None:
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start")
        self._f = f
        self._g = g
        self._t_start = t_start
        self._t_end = t_end
        self._strength = strength

    def evaluate(self, x: float) -> float:
        """Evaluate the power-law blended curve at x.

        Args:
            x: The input value.

        Returns:
            Blended value between f and g.
        """
        if x <= self._t_start:
            return self._f.evaluate(x)
        if x >= self._t_end:
            return self._g.evaluate(x)
        t = (x - self._t_start) / (self._t_end - self._t_start)
        w = t ** self._strength
        return (1.0 - w) * self._f.evaluate(x) + w * self._g.evaluate(x)


def _build_reform_segment_curve(
    source_curve: ParametricCurve,
    sample_maturities: Sequence[float],
    reform_maturity: float,
    rate_at_reform: float,
) -> ParametricCurve:
    """Build a CubicSpline segment for RPI reform blending.

    Matches C# ``BuildRpiReformAdjustedCurve``: samples the source curve
    at filtered maturities, adds the reform point, fits a natural cubic
    spline, and wraps with constant extrapolation.

    Args:
        source_curve: The breakeven RPI forward-rate curve.
        sample_maturities: Maturities to sample from the source curve.
        reform_maturity: Time to reform effective date (years).
        rate_at_reform: Target forward rate at the reform date.

    Returns:
        CubicSpline with constant extrapolation.
    """
    points: list[tuple[float, float]] = []
    for m in sample_maturities:
        points.append((m, source_curve.evaluate(m)))
    points.append((reform_maturity, rate_at_reform))
    points.sort(key=lambda p: p[0])

    maturities = [p[0] for p in points]
    rates = [p[1] for p in points]

    if len(maturities) < 2:
        return ConstantCurve(rate_at_reform)

    spline = CubicSpline(maturities, rates)
    return ConstantExtrapolation(spline, maturities[0], maturities[-1])


def reform_adjusted_forward_curve(
    breakeven_rpi_fwd: ParametricCurve,
    expected_rate_at_reform: float,
    reform_maturity: float,
    rpi_cpih_wedge: float,
    inflation_maturities: Sequence[float],
    *,
    adjustment_period_pre: float = 2.0,
    adjustment_period_post: float = 5.0,
    transition_period_post: float = 1.0 / 12.0,
    transition_strength: float = 2.0,
) -> ParametricCurve:
    """Compute the RPI reform-adjusted forward-rate curve.

    Matches C# ``CalculateReformAdjustedCtsFwdMarketCurve``:

    1. Pre-reform segment: market data before (reform − pre_period),
       plus reform point at ``expected_rate_at_reform``.
    2. Post-reform segment: market data after (reform + post_period),
       plus reform point at ``expected_rate_at_reform − rpi_cpih_wedge``.
    3. Blend from pre to post over ``[reform, reform + transition_period]``
       using power-law with ``transition_strength``.

    Args:
        breakeven_rpi_fwd: Breakeven RPI instantaneous forward rate curve.
        expected_rate_at_reform: Expected instantaneous forward rate at
            the reform date.
        reform_maturity: Time to reform effective date (years).
        rpi_cpih_wedge: Market-implied RPI−CPIH forward wedge.
        inflation_maturities: Full inflation maturity grid.
        adjustment_period_pre: Years before reform for sampling gap.
        adjustment_period_post: Years after reform for sampling gap.
        transition_period_post: Transition period after reform for blending.
        transition_strength: Power exponent for blend (2.0 = quadratic).

    Returns:
        Reform-adjusted RPI forward-rate curve.
    """
    adj_start = reform_maturity - adjustment_period_pre
    adj_end = reform_maturity + adjustment_period_post

    post_reform_rate = expected_rate_at_reform - rpi_cpih_wedge

    # Pre-reform: sample before adjustment start.
    pre_mats = [m for m in inflation_maturities if m < adj_start]
    pre_curve = _build_reform_segment_curve(
        breakeven_rpi_fwd, pre_mats, reform_maturity, expected_rate_at_reform,
    )

    # Post-reform: sample after adjustment end.
    post_mats = [m for m in inflation_maturities if m > adj_end]
    post_curve = _build_reform_segment_curve(
        breakeven_rpi_fwd, post_mats, reform_maturity, post_reform_rate,
    )

    # Blend: sharp transition at reform date.
    return PowerBlend(
        pre_curve,
        post_curve,
        reform_maturity,
        reform_maturity + transition_period_post,
        strength=transition_strength,
    )


def breakeven_cpi_forward_curve(
    breakeven_rpi_fwd: ParametricCurve,
    reform_maturity: float,
    pre_reform_rpi_minus_cpi: float,
    post_reform_rpi_minus_cpi: float,
    *,
    transition_pre: float = 2.0,
    transition_post: float = 2.0,
    transition_strength: float = 1.0,
) -> ParametricCurve:
    """Derive CPI breakeven forward curve from RPI breakeven.

    Matches C# ``SmoothEstimatedBreakevenCpiInitialInstantaneousForwardRateCurve``:
    ``CPI_breakeven = RPI_breakeven − smooth_rpi_minus_cpi_wedge``

    The wedge transitions from ``pre_reform_rpi_minus_cpi`` to
    ``post_reform_rpi_minus_cpi`` over ``[reform − pre, reform + post]``
    using power-law blending.

    Args:
        breakeven_rpi_fwd: Breakeven RPI instantaneous forward rate curve.
        reform_maturity: Time to reform effective date (years).
        pre_reform_rpi_minus_cpi: Pre-reform RPI−CPI wedge (continuous rate).
        post_reform_rpi_minus_cpi: Post-reform RPI−CPI wedge (continuous rate).
        transition_pre: Years before reform for wedge transition.
        transition_post: Years after reform for wedge transition.
        transition_strength: Power exponent for wedge blending.

    Returns:
        CPI breakeven instantaneous forward-rate curve.
    """
    wedge_pre = ConstantCurve(pre_reform_rpi_minus_cpi)
    wedge_post = ConstantCurve(post_reform_rpi_minus_cpi)

    smooth_wedge = PowerBlend(
        wedge_pre,
        wedge_post,
        reform_maturity - transition_pre,
        reform_maturity + transition_post,
        strength=transition_strength,
    )

    return breakeven_rpi_fwd - smooth_wedge


# ── Legacy API (kept for backward compatibility) ──────────────────


def _zcbp_from_spot(spot_rate: float, maturity: float) -> float:
    """ZCB price from continuous spot rate: P(t) = exp(-s(t) * t).

    Args:
        spot_rate: Continuous spot rate.
        maturity: Time to maturity.

    Returns:
        Zero-coupon bond price.
    """
    return math.exp(-spot_rate * maturity)
