"""Tests for the Akima yield curve calibration pipeline."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from hyesg.calibration.yield_curves import (
    YIELD_CURVE_KNOTS,
    PowerBlend,
    YieldCurveCalibrationResult,
    breakeven_cpi_forward_curve,
    build_akima_yield_curve,
    build_forward_rate_curve,
    build_real_yield_curve,
    calibrate_yield_curve,
    fisher_real_rate,
    reform_adjusted_forward_curve,
)
from hyesg.calibration.yield_curve_config import (
    LongEndExtensionConfig,
    RpiReformConfig,
    YieldCurvePipelineConfig,
)
from hyesg.calibration.yield_curve_model import InitialYieldCurveModel
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.curves.splines import AkimaCubicSpline

jax.config.update("jax_enable_x64", True)

# ── Realistic UK gilt spot-rate data ────────────────────────────────
# 15 knot points matching YIELD_CURVE_KNOTS
UK_GILT_RATES: list[float] = [
    0.0425,  # 0y
    0.0400,  # 1y
    0.0410,  # 2y
    0.0415,  # 3y
    0.0420,  # 5y
    0.0430,  # 10y
    0.0435,  # 15y
    0.0440,  # 20y
    0.0445,  # 30y
    0.0440,  # 40y
    0.0435,  # 50y
    0.0430,  # 60y
    0.0425,  # 70y
    0.0420,  # 80y
    0.0415,  # 90y
]

UK_INFLATION_RATES: list[float] = [
    0.0300,  # 0y
    0.0280,  # 1y
    0.0290,  # 2y
    0.0295,  # 3y
    0.0300,  # 5y
    0.0310,  # 10y
    0.0315,  # 15y
    0.0320,  # 20y
    0.0325,  # 30y
    0.0320,  # 40y
    0.0315,  # 50y
    0.0310,  # 60y
    0.0305,  # 70y
    0.0300,  # 80y
    0.0295,  # 90y
]


# ── YIELD_CURVE_KNOTS ───────────────────────────────────────────────


class TestYieldCurveKnots:
    """Tests for the standard knot points constant."""

    def test_knot_count(self) -> None:
        """Standard grid has exactly 15 knot points."""
        assert len(YIELD_CURVE_KNOTS) == 15

    def test_knots_sorted(self) -> None:
        """Knots are strictly increasing."""
        for i in range(len(YIELD_CURVE_KNOTS) - 1):
            assert YIELD_CURVE_KNOTS[i] < YIELD_CURVE_KNOTS[i + 1]

    def test_knots_start_at_zero(self) -> None:
        """First knot is at maturity 0."""
        assert YIELD_CURVE_KNOTS[0] == 0

    def test_knots_end_at_90(self) -> None:
        """Last knot is at maturity 90."""
        assert YIELD_CURVE_KNOTS[-1] == 90

    def test_expected_maturities(self) -> None:
        """Knots match the expected standard maturities."""
        expected = (0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90)
        assert YIELD_CURVE_KNOTS == expected


# ── build_akima_yield_curve ─────────────────────────────────────────


class TestBuildAkimaYieldCurve:
    """Tests for building Akima spline yield curves."""

    def test_flat_rates_recovery(self) -> None:
        """Flat rates should be recovered exactly at every knot."""
        flat_rate = 0.05
        rates = [flat_rate] * len(YIELD_CURVE_KNOTS)
        curve = build_akima_yield_curve(rates)

        for knot in YIELD_CURVE_KNOTS:
            assert abs(curve.evaluate(knot) - flat_rate) < 1e-14

    def test_knot_interpolation(self) -> None:
        """Curve passes through all input knot points."""
        curve = build_akima_yield_curve(UK_GILT_RATES)

        for knot, rate in zip(YIELD_CURVE_KNOTS, UK_GILT_RATES):
            assert abs(curve.evaluate(knot) - rate) < 1e-14

    def test_flat_tail_extrapolation_at_95y(self) -> None:
        """Value at 95y matches the 90y rate (flat tail)."""
        curve = build_akima_yield_curve(UK_GILT_RATES)
        assert abs(curve.evaluate(95.0) - UK_GILT_RATES[-1]) < 1e-14

    def test_flat_tail_extrapolation_at_100y(self) -> None:
        """Value at 100y matches the 90y rate (flat tail)."""
        curve = build_akima_yield_curve(UK_GILT_RATES)
        assert abs(curve.evaluate(100.0) - UK_GILT_RATES[-1]) < 1e-14

    def test_flat_tail_extrapolation_beyond_100y(self) -> None:
        """Value at 110y uses flat extrapolation beyond the last knot."""
        curve = build_akima_yield_curve(UK_GILT_RATES)
        # Beyond the appended 100y point, AkimaCubicSpline uses flat extrap.
        assert abs(curve.evaluate(110.0) - UK_GILT_RATES[-1]) < 1e-14

    def test_returns_parametric_curve(self) -> None:
        """Return type is a ParametricCurve."""
        curve = build_akima_yield_curve(UK_GILT_RATES)
        assert isinstance(curve, ParametricCurve)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched knots/rates lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            build_akima_yield_curve([0.04, 0.05], knots=(0, 1, 2))

    def test_unsupported_extrapolation_raises(self) -> None:
        """Unsupported extrapolation method raises ValueError."""
        rates = [0.04] * len(YIELD_CURVE_KNOTS)
        with pytest.raises(ValueError, match="Unsupported"):
            build_akima_yield_curve(rates, extrapolation="linear")

    def test_interpolation_between_knots(self) -> None:
        """Values between knots are smooth and bounded by neighbours."""
        curve = build_akima_yield_curve(UK_GILT_RATES)
        # At maturity 7.5 (between 5y and 10y knots)
        val = curve.evaluate(7.5)
        rate_5y = UK_GILT_RATES[4]   # 0.042
        rate_10y = UK_GILT_RATES[5]  # 0.043
        assert min(rate_5y, rate_10y) - 0.001 <= val <= max(rate_5y, rate_10y) + 0.001

    def test_custom_knots(self) -> None:
        """Works with non-standard knot configuration."""
        knots = [0, 5, 10, 20, 30]
        rates = [0.04, 0.042, 0.044, 0.043, 0.041]
        curve = build_akima_yield_curve(rates, knots=knots)

        for k, r in zip(knots, rates):
            assert abs(curve.evaluate(k) - r) < 1e-14


# ── Fisher equation ─────────────────────────────────────────────────


class TestFisherRealRate:
    """Tests for the Fisher equation."""

    def test_zero_inflation(self) -> None:
        """Real rate equals nominal when inflation is zero."""
        assert abs(fisher_real_rate(0.05, 0.0) - 0.05) < 1e-14

    def test_equal_nominal_inflation(self) -> None:
        """Real rate is zero when nominal equals inflation."""
        assert abs(fisher_real_rate(0.05, 0.05)) < 1e-14

    def test_typical_values(self) -> None:
        """Standard Fisher calculation with typical market values."""
        nominal = 0.05
        inflation = 0.03
        expected = (1.05 / 1.03) - 1.0
        assert abs(fisher_real_rate(nominal, inflation) - expected) < 1e-14

    def test_negative_real_rate(self) -> None:
        """Negative real rate when inflation exceeds nominal."""
        result = fisher_real_rate(0.02, 0.04)
        assert result < 0.0
        expected = (1.02 / 1.04) - 1.0
        assert abs(result - expected) < 1e-14

    def test_zero_nominal(self) -> None:
        """Zero nominal rate with positive inflation gives negative real."""
        result = fisher_real_rate(0.0, 0.03)
        expected = (1.0 / 1.03) - 1.0
        assert abs(result - expected) < 1e-14


# ── build_real_yield_curve ──────────────────────────────────────────


class TestBuildRealYieldCurve:
    """Tests for deriving real yield curves via Fisher."""

    def test_real_curve_at_knots(self) -> None:
        """Real curve values at knots match Fisher(nominal, inflation)."""
        nominal_curve = build_akima_yield_curve(UK_GILT_RATES)
        inflation_curve = build_akima_yield_curve(UK_INFLATION_RATES)
        real_curve = build_real_yield_curve(nominal_curve, inflation_curve)

        for knot in YIELD_CURVE_KNOTS:
            nom = nominal_curve.evaluate(knot)
            inf = inflation_curve.evaluate(knot)
            expected_real = fisher_real_rate(nom, inf)
            assert abs(real_curve.evaluate(knot) - expected_real) < 1e-12

    def test_real_curve_is_parametric(self) -> None:
        """Real curve is a ParametricCurve."""
        nominal_curve = build_akima_yield_curve(UK_GILT_RATES)
        inflation_curve = build_akima_yield_curve(UK_INFLATION_RATES)
        real_curve = build_real_yield_curve(nominal_curve, inflation_curve)
        assert isinstance(real_curve, ParametricCurve)


# ── calibrate_yield_curve ───────────────────────────────────────────


class TestCalibrateYieldCurve:
    """Tests for the full forward-rate calibration pipeline."""

    def test_result_type(self) -> None:
        """Result is a YieldCurveCalibrationResult NamedTuple."""
        result = calibrate_yield_curve(UK_GILT_RATES)
        assert isinstance(result, YieldCurveCalibrationResult)
        assert isinstance(result.spot_curve, ParametricCurve)
        assert isinstance(result.forward_curve, ParametricCurve)
        assert isinstance(result.zcbp_curve, ParametricCurve)

    def test_residuals_below_tolerance(self) -> None:
        """Residuals at pre-extension knots are below default tolerance.

        The forward-rate pipeline (spot → forward → Akima → integrate)
        has inherent O(1e-3) round-trip error at short maturities because
        the Akima forward spline's inter-knot behaviour differs from the
        derivative of the original spot Akima. This is expected.
        """
        result = calibrate_yield_curve(UK_GILT_RATES)
        for v in result.residuals:
            assert abs(float(v)) < 5e-3

    def test_zcbp_at_zero_is_one(self) -> None:
        """ZCB price at maturity 0 should be 1.0."""
        result = calibrate_yield_curve(UK_GILT_RATES)
        p0 = result.zcbp_curve.evaluate(0.0)
        assert abs(p0 - 1.0) < 1e-10

    def test_zcbp_decreasing(self) -> None:
        """ZCB prices decrease with maturity (positive rates)."""
        result = calibrate_yield_curve(UK_GILT_RATES)
        prev = result.zcbp_curve.evaluate(0.0)
        for t in [1, 5, 10, 20, 30, 50]:
            current = result.zcbp_curve.evaluate(t)
            assert current < prev, f"ZCBP not decreasing at t={t}"
            prev = current

    def test_flat_curve_calibration(self) -> None:
        """Flat spot curve calibrates — forward ≈ spot everywhere."""
        flat_rate = 0.04
        rates = [flat_rate] * len(YIELD_CURVE_KNOTS)
        result = calibrate_yield_curve(rates)

        for t in [1.0, 5.0, 10.0, 30.0, 50.0]:
            fwd = result.forward_curve.evaluate(t)
            assert abs(fwd - flat_rate) < 1e-3, (
                f"Forward != spot for flat curve at t={t}: {fwd}"
            )

    def test_with_steep_curve(self) -> None:
        """Calibrate a steeply upward-sloping curve."""
        rates = [0.01 + 0.001 * i for i in range(len(YIELD_CURVE_KNOTS))]
        result = calibrate_yield_curve(rates)
        # Forward rates should be positive and reasonable.
        for t in [1, 5, 10, 20, 30]:
            fwd = result.forward_curve.evaluate(t)
            assert -0.1 < fwd < 0.5, f"Unreasonable forward at t={t}: {fwd}"


# ── Forward-rate Akima pipeline ────────────────────────────────────


class TestBuildForwardRateCurve:
    """Tests for the forward-rate Akima pipeline."""

    def test_flat_spot_gives_flat_forward(self) -> None:
        """Flat spot rates should produce flat forward rates."""
        flat_rate = 0.04
        rates = [flat_rate] * len(YIELD_CURVE_KNOTS)
        fwd_curve = build_forward_rate_curve(rates)

        for t in [1.0, 5.0, 10.0, 30.0, 60.0]:
            fwd = fwd_curve.evaluate(t)
            assert abs(fwd - flat_rate) < 1e-6, (
                f"Forward != {flat_rate} at t={t}: {fwd}"
            )

    def test_returns_parametric_curve(self) -> None:
        """Return type is a ParametricCurve."""
        fwd = build_forward_rate_curve(UK_GILT_RATES)
        assert isinstance(fwd, ParametricCurve)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched knots/rates lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            build_forward_rate_curve([0.04, 0.05], knots=(0, 1, 2))

    def test_long_end_extension(self) -> None:
        """With long_term_forward_rate, curve converges to target at 90y."""
        target = math.log(1 + 0.05)  # ~4.88% continuous
        fwd = build_forward_rate_curve(
            UK_GILT_RATES,
            long_term_forward_rate=target,
            transition_start=61.0,
            transition_end=90.0,
        )
        # At 90y the Akima should hit the target knot.
        assert abs(fwd.evaluate(90.0) - target) < 1e-6

    def test_long_end_market_region_preserved(self) -> None:
        """Knots before transition_start should be unaffected."""
        target = math.log(1 + 0.05)
        fwd_no_ext = build_forward_rate_curve(UK_GILT_RATES)
        fwd_ext = build_forward_rate_curve(
            UK_GILT_RATES,
            long_term_forward_rate=target,
        )
        # At maturities well inside the market region, values should match.
        for t in [1.0, 5.0, 10.0, 20.0, 30.0]:
            val_no_ext = fwd_no_ext.evaluate(t)
            val_ext = fwd_ext.evaluate(t)
            assert abs(val_no_ext - val_ext) < 1e-6, (
                f"Long-end extension affected market region at t={t}"
            )


# ── Long-end extension config ──────────────────────────────────────


class TestLongEndExtensionConfig:
    """Tests for LongEndExtensionConfig validation."""

    def test_defaults(self) -> None:
        cfg = LongEndExtensionConfig()
        assert cfg.transition_start == 61.0
        assert cfg.transition_end == 90.0

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError):
            LongEndExtensionConfig(transition_start=90, transition_end=61)


# ── RPI Reform Config ──────────────────────────────────────────────


class TestRpiReformConfig:
    """Tests for RpiReformConfig."""

    def test_defaults(self) -> None:
        cfg = RpiReformConfig()
        assert cfg.breakeven_transition_pre == 2.0
        assert cfg.breakeven_transition_post == 2.0

    def test_time_to_effective_date(self) -> None:
        from datetime import date
        cfg = RpiReformConfig()
        sim_date = date(2025, 1, 1)
        ttd = cfg.time_to_effective_date(sim_date)
        assert ttd > 4.0  # ~5.1 years to Feb 2030


# ── Pipeline config ────────────────────────────────────────────────


class TestYieldCurvePipelineConfig:
    """Tests for YieldCurvePipelineConfig."""

    def test_inflation_maturities_length(self) -> None:
        cfg = YieldCurvePipelineConfig()
        mats = cfg.inflation_maturities()
        # 0 to 10 in 0.25 steps = 41, plus 11 to 100 in 1.0 steps = 90
        assert len(mats) == 131

    def test_inflation_maturities_sorted(self) -> None:
        cfg = YieldCurvePipelineConfig()
        mats = cfg.inflation_maturities()
        for i in range(len(mats) - 1):
            assert mats[i] < mats[i + 1]


# ── InitialYieldCurveModel ─────────────────────────────────────────


class TestInitialYieldCurveModel:
    """Tests for the unified yield curve model."""

    def test_from_forward_curve(self) -> None:
        """Model can be constructed from a forward curve."""
        fwd = build_forward_rate_curve(UK_GILT_RATES)
        model = InitialYieldCurveModel.from_forward_curve(fwd)

        assert isinstance(model.forward_curve, ParametricCurve)
        assert isinstance(model.spot_curve, ParametricCurve)
        assert isinstance(model.zcbp_curve, ParametricCurve)
        assert isinstance(model.inv_zcbp_curve, ParametricCurve)

    def test_zcb_times_inv_zcb_is_one(self) -> None:
        """P(t) * (1/P(t)) ≈ 1.0."""
        fwd = build_forward_rate_curve(UK_GILT_RATES)
        model = InitialYieldCurveModel.from_forward_curve(fwd)

        for t in [1.0, 5.0, 10.0, 20.0, 30.0]:
            product = model.zcb_price(t) * model.accumulation_factor(t)
            assert abs(product - 1.0) < 1e-4, (
                f"P*invP != 1 at t={t}: {product}"
            )

    def test_spot_compounding_continuous(self) -> None:
        """Continuous spot matches direct evaluation."""
        fwd = build_forward_rate_curve(UK_GILT_RATES)
        model = InitialYieldCurveModel.from_forward_curve(fwd)
        assert model.spot_rate(10.0) == model.spot_curve.evaluate(10.0)

    def test_spot_compounding_annual(self) -> None:
        """Annual spot rate = exp(cts_spot) - 1."""
        fwd = build_forward_rate_curve(UK_GILT_RATES)
        model = InitialYieldCurveModel.from_forward_curve(fwd)
        cts = model.spot_rate(10.0, "continuous")
        annual = model.spot_rate(10.0, "annual")
        assert abs(annual - (math.exp(cts) - 1.0)) < 1e-12


# ── PowerBlend ─────────────────────────────────────────────────────


class TestPowerBlend:
    """Tests for the power-law blending curve."""

    def test_linear_blend(self) -> None:
        """strength=1 gives linear interpolation."""
        f = ConstantCurve(1.0)
        g = ConstantCurve(2.0)
        blend = PowerBlend(f, g, 0.0, 1.0, strength=1.0)

        assert abs(blend.evaluate(-1.0) - 1.0) < 1e-14  # before start
        assert abs(blend.evaluate(0.0) - 1.0) < 1e-14   # at start
        assert abs(blend.evaluate(0.5) - 1.5) < 1e-14   # midpoint
        assert abs(blend.evaluate(1.0) - 2.0) < 1e-14   # at end
        assert abs(blend.evaluate(2.0) - 2.0) < 1e-14   # after end

    def test_quadratic_blend(self) -> None:
        """strength=2 gives quadratic weight."""
        f = ConstantCurve(0.0)
        g = ConstantCurve(1.0)
        blend = PowerBlend(f, g, 0.0, 1.0, strength=2.0)

        # At t=0.5: w = 0.25, so blend = 0.25
        assert abs(blend.evaluate(0.5) - 0.25) < 1e-14

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError):
            PowerBlend(ConstantCurve(0.0), ConstantCurve(1.0), 1.0, 0.0)


# ── RPI reform blending ───────────────────────────────────────────


class TestReformAdjustedForwardCurve:
    """Tests for RPI reform-adjusted curve construction."""

    def test_pre_reform_unchanged(self) -> None:
        """Well before reform, curve matches the pre-reform segment."""
        flat_rpi = ConstantCurve(0.03)
        reform_maturity = 5.0
        mats = [float(i) * 0.25 for i in range(41)] + list(range(11, 101))

        adjusted = reform_adjusted_forward_curve(
            flat_rpi,
            expected_rate_at_reform=0.03,
            reform_maturity=reform_maturity,
            rpi_cpih_wedge=0.01,
            inflation_maturities=mats,
            adjustment_period_pre=2.0,
            adjustment_period_post=5.0,
        )
        # At t=0, should be close to original flat RPI.
        val = adjusted.evaluate(0.0)
        assert abs(val - 0.03) < 0.01

    def test_post_reform_stepped_down(self) -> None:
        """After reform + transition, rate should reflect the step-down."""
        flat_rpi = ConstantCurve(0.03)
        reform_maturity = 5.0
        mats = [float(i) * 0.25 for i in range(41)] + list(range(11, 101))

        adjusted = reform_adjusted_forward_curve(
            flat_rpi,
            expected_rate_at_reform=0.03,
            reform_maturity=reform_maturity,
            rpi_cpih_wedge=0.01,
            inflation_maturities=mats,
            adjustment_period_pre=2.0,
            adjustment_period_post=5.0,
            transition_period_post=1.0 / 12.0,
        )
        # Well after reform, rate should be stepped down by wedge.
        val = adjusted.evaluate(50.0)
        assert abs(val - 0.02) < 0.01


# ── CPI breakeven ─────────────────────────────────────────────────


class TestBreakevenCpiForwardCurve:
    """Tests for CPI breakeven derivation."""

    def test_pre_reform_wedge(self) -> None:
        """Before reform, CPI = RPI - pre_reform_wedge."""
        rpi = ConstantCurve(0.04)
        reform_maturity = 5.0
        pre_wedge = math.log(1.01)
        post_wedge = 0.0

        cpi = breakeven_cpi_forward_curve(
            rpi,
            reform_maturity=reform_maturity,
            pre_reform_rpi_minus_cpi=pre_wedge,
            post_reform_rpi_minus_cpi=post_wedge,
            transition_pre=2.0,
            transition_post=2.0,
        )
        # Well before reform: CPI ≈ 0.04 - log(1.01) ≈ 0.03005
        val = cpi.evaluate(0.0)
        expected = 0.04 - pre_wedge
        assert abs(val - expected) < 1e-10

    def test_post_reform_wedge(self) -> None:
        """After reform, CPI = RPI - post_reform_wedge."""
        rpi = ConstantCurve(0.04)
        reform_maturity = 5.0
        pre_wedge = math.log(1.01)
        post_wedge = 0.0

        cpi = breakeven_cpi_forward_curve(
            rpi,
            reform_maturity=reform_maturity,
            pre_reform_rpi_minus_cpi=pre_wedge,
            post_reform_rpi_minus_cpi=post_wedge,
            transition_pre=2.0,
            transition_post=2.0,
        )
        # Well after reform: CPI ≈ 0.04 - 0.0 = 0.04
        val = cpi.evaluate(50.0)
        assert abs(val - 0.04) < 1e-10


# ── Integration: full pipeline with realistic data ──────────────────


class TestRealisticYieldCurve:
    """Integration tests with realistic UK gilt data."""

    def test_full_pipeline_realistic_data(self) -> None:
        """End-to-end: calibrate realistic UK gilt curve.

        Forward-rate pipeline has O(1e-3) spot-rate recovery residuals
        at short maturities. Beyond ~3y the error is much smaller.
        """
        result = calibrate_yield_curve(UK_GILT_RATES)

        # Spot curve round-trip — forward-rate pipeline has inherent
        # residuals; worst case ~1e-3 at short maturities.
        for knot, rate in zip(YIELD_CURVE_KNOTS, UK_GILT_RATES):
            assert abs(result.spot_curve.evaluate(knot) - rate) < 5e-3

        # Forward curve is positive everywhere (positive spot rates).
        for t in range(1, 91):
            fwd = result.forward_curve.evaluate(t)
            # Forward can be negative in principle for non-monotone curves,
            # but should be reasonable.
            assert -0.1 < fwd < 0.2, f"Unreasonable forward at t={t}: {fwd}"

    def test_nominal_and_real_pipeline(self) -> None:
        """Build nominal, inflation, and real curves together."""
        nominal_result = calibrate_yield_curve(UK_GILT_RATES)
        inflation_result = calibrate_yield_curve(UK_INFLATION_RATES)

        real_curve = build_real_yield_curve(
            nominal_result.spot_curve,
            inflation_result.spot_curve,
        )

        # Check real rates at a few knots.
        for knot in [1, 10, 30, 50]:
            nom = nominal_result.spot_curve.evaluate(knot)
            inf = inflation_result.spot_curve.evaluate(knot)
            expected_real = fisher_real_rate(nom, inf)
            actual_real = real_curve.evaluate(knot)
            assert abs(actual_real - expected_real) < 1e-12

    def test_swap_curve_distinct_from_gilt(self) -> None:
        """Different rates produce different curves."""
        swap_rates = [r + 0.002 for r in UK_GILT_RATES]
        gilt_result = calibrate_yield_curve(UK_GILT_RATES)
        swap_result = calibrate_yield_curve(swap_rates)

        # Swap rates should be higher at every knot.
        for knot in YIELD_CURVE_KNOTS:
            gilt_val = gilt_result.spot_curve.evaluate(knot)
            swap_val = swap_result.spot_curve.evaluate(knot)
            assert swap_val > gilt_val
