"""Tests for RPI reform transition curves."""

from __future__ import annotations

import pytest

from hyesg.math.curves.blending import LinearBlend, PolynomialBlend
from hyesg.math.curves.primitives import ConstantCurve, LinearCurve
from hyesg.models.inflation.rpi_reform import (
    RPI_REFORM_DATE_YEARS,
    RpiReformBreakevenCurve,
    RpiReformConfig,
    RpiReformRealisedCurve,
    _InstantSwitch,
)


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def rpi_curve() -> ConstantCurve:
    """Flat RPI breakeven at 3%."""
    return ConstantCurve(0.03)


@pytest.fixture
def cpih_curve() -> ConstantCurve:
    """Flat CPIH breakeven at 2%."""
    return ConstantCurve(0.02)


@pytest.fixture
def rpi_linear() -> LinearCurve:
    """RPI curve with slope: 0.01*t + 0.03."""
    return LinearCurve(slope=0.01, intercept=0.03)


@pytest.fixture
def cpih_linear() -> LinearCurve:
    """CPIH curve with slope: 0.005*t + 0.02."""
    return LinearCurve(slope=0.005, intercept=0.02)


# ─── RPI_REFORM_DATE_YEARS ─────────────────────────────────────────


class TestReformDateConstant:
    def test_reform_date_value(self) -> None:
        """Feb 2030 as year fraction: 2030 + 1.5/12."""
        expected = 2030 + 1.5 / 12
        assert RPI_REFORM_DATE_YEARS == pytest.approx(expected)

    def test_reform_date_is_february_2030(self) -> None:
        """Reform date lies between Jan and Mar 2030."""
        assert 2030.0 < RPI_REFORM_DATE_YEARS < 2030.25


# ─── RpiReformBreakevenCurve ───────────────────────────────────────


class TestBreakevenPreReform:
    def test_pure_rpi_before_reform(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Before reform_start, returns pure RPI value."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        assert curve.evaluate(0.0) == pytest.approx(0.03)
        assert curve.evaluate(4.99) == pytest.approx(0.03)

    def test_pure_rpi_at_boundary(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """At t == reform_start exactly, blend weight is 0 → pure RPI."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=2.0
        )
        assert curve.evaluate(5.0) == pytest.approx(0.03)


class TestBreakevenPostReform:
    def test_pure_cpih_after_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """After reform_start + blend_period, returns pure CPIH."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        assert curve.evaluate(6.0) == pytest.approx(0.02)
        assert curve.evaluate(10.0) == pytest.approx(0.02)

    def test_pure_cpih_at_end_boundary(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """At t == reform_start + blend_period, pure CPIH."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=2.0
        )
        assert curve.evaluate(7.0) == pytest.approx(0.02)


class TestBreakevenDuringTransition:
    def test_midpoint_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """At midpoint of blend, value is average of RPI and CPIH."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=2.0
        )
        mid = curve.evaluate(6.0)
        assert mid == pytest.approx(0.025)

    def test_quarter_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """At 25% through blend, 75% RPI + 25% CPIH."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=4.0, blend_period=4.0
        )
        val = curve.evaluate(5.0)  # 25% through
        expected = 0.75 * 0.03 + 0.25 * 0.02
        assert val == pytest.approx(expected)

    def test_uses_linear_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Breakeven curve uses LinearBlend internally."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        assert isinstance(curve._blend, LinearBlend)


class TestBreakevenInstantaneous:
    def test_instant_switch(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """blend_period=0: hard switch at reform_start."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=0.0
        )
        assert curve.evaluate(4.999) == pytest.approx(0.03)
        assert curve.evaluate(5.0) == pytest.approx(0.02)

    def test_instant_uses_switch(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Instantaneous blend uses _InstantSwitch, not LinearBlend."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=0.0
        )
        assert isinstance(curve._blend, _InstantSwitch)


# ─── RpiReformRealisedCurve ────────────────────────────────────────


class TestRealisedPreReform:
    def test_pure_rpi_before_reform(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Before reform_start, returns pure RPI value."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        assert curve.evaluate(0.0) == pytest.approx(0.03)
        assert curve.evaluate(4.99) == pytest.approx(0.03)


class TestRealisedPostReform:
    def test_pure_cpih_after_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """After reform_start + blend_period, returns pure CPIH."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        assert curve.evaluate(6.0) == pytest.approx(0.02)
        assert curve.evaluate(10.0) == pytest.approx(0.02)


class TestRealisedDuringTransition:
    def test_midpoint_smooth_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Polynomial midpoint: smoothstep(0.5) = 0.5 for cubic."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=2.0
        )
        mid = curve.evaluate(6.0)
        # Cubic smoothstep at t=0.5: 3*(0.25) - 2*(0.125) = 0.5
        assert mid == pytest.approx(0.025)

    def test_uses_polynomial_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Realised curve uses PolynomialBlend internally."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        assert isinstance(curve._blend, PolynomialBlend)


class TestRealisedInstantaneous:
    def test_instant_switch(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """blend_period=0: hard switch at reform_start."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=0.0
        )
        assert curve.evaluate(4.999) == pytest.approx(0.03)
        assert curve.evaluate(5.0) == pytest.approx(0.02)


class TestPolynomialBoundaryDerivatives:
    def test_zero_derivative_at_start(
        self, rpi_linear: LinearCurve, cpih_linear: LinearCurve
    ) -> None:
        """Polynomial blend has zero derivative at transition start."""
        curve = RpiReformRealisedCurve(
            rpi_linear, cpih_linear, reform_start=5.0, blend_period=2.0
        )
        # Numerical derivative of the blend weight at the boundary
        h = 1e-6
        deriv_at_start = (curve.evaluate(5.0 + h) - curve.evaluate(5.0 - h)) / (
            2.0 * h
        )
        # The derivative of the pure RPI curve at t=5.0
        rpi_deriv = rpi_linear.derivative(5.0)
        # At the boundary the blended derivative should match the RPI
        # derivative (smoothstep derivative is zero at boundaries)
        assert deriv_at_start == pytest.approx(rpi_deriv, abs=1e-4)

    def test_zero_derivative_at_end(
        self, rpi_linear: LinearCurve, cpih_linear: LinearCurve
    ) -> None:
        """Polynomial blend has zero derivative at transition end."""
        reform_start = 5.0
        blend_period = 2.0
        curve = RpiReformRealisedCurve(
            rpi_linear, cpih_linear,
            reform_start=reform_start,
            blend_period=blend_period,
        )
        t_end = reform_start + blend_period
        h = 1e-6
        deriv_at_end = (curve.evaluate(t_end + h) - curve.evaluate(t_end - h)) / (
            2.0 * h
        )
        cpih_deriv = cpih_linear.derivative(t_end)
        assert deriv_at_end == pytest.approx(cpih_deriv, abs=1e-4)


# ─── RpiReformConfig ──────────────────────────────────────────────


class TestRpiReformConfig:
    def test_config_properties(self) -> None:
        """Config stores reform parameters correctly."""
        config = RpiReformConfig(
            reform_date=10.0,
            blend_period_breakeven=0.5,
            blend_period_realised=2.0,
        )
        assert config.reform_date == pytest.approx(10.0)
        assert config.blend_period_breakeven == pytest.approx(0.5)
        assert config.blend_period_realised == pytest.approx(2.0)

    def test_config_defaults(self) -> None:
        """Default blend periods: breakeven=0, realised=1."""
        config = RpiReformConfig(reform_date=5.0)
        assert config.blend_period_breakeven == pytest.approx(0.0)
        assert config.blend_period_realised == pytest.approx(1.0)

    def test_create_breakeven_curve(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Factory creates RpiReformBreakevenCurve with correct params."""
        config = RpiReformConfig(
            reform_date=5.0, blend_period_breakeven=1.0
        )
        curve = config.create_breakeven_curve(rpi_curve, cpih_curve)
        assert isinstance(curve, RpiReformBreakevenCurve)
        assert curve.evaluate(4.0) == pytest.approx(0.03)
        assert curve.evaluate(6.0) == pytest.approx(0.02)

    def test_create_realised_curve(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Factory creates RpiReformRealisedCurve with correct params."""
        config = RpiReformConfig(
            reform_date=5.0, blend_period_realised=2.0
        )
        curve = config.create_realised_curve(rpi_curve, cpih_curve)
        assert isinstance(curve, RpiReformRealisedCurve)
        assert curve.evaluate(4.0) == pytest.approx(0.03)
        assert curve.evaluate(7.0) == pytest.approx(0.02)

    def test_factory_breakeven_uses_linear(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Config factory breakeven curve uses LinearBlend."""
        config = RpiReformConfig(
            reform_date=5.0, blend_period_breakeven=1.0
        )
        curve = config.create_breakeven_curve(rpi_curve, cpih_curve)
        assert isinstance(curve._blend, LinearBlend)

    def test_factory_realised_uses_polynomial(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Config factory realised curve uses PolynomialBlend."""
        config = RpiReformConfig(
            reform_date=5.0, blend_period_realised=1.0
        )
        curve = config.create_realised_curve(rpi_curve, cpih_curve)
        assert isinstance(curve._blend, PolynomialBlend)


# ─── Edge Cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_reform_date_zero_breakeven(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Immediate reform (reform_date=0): always CPIH."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=0.0, blend_period=0.0
        )
        assert curve.evaluate(0.0) == pytest.approx(0.02)
        assert curve.evaluate(5.0) == pytest.approx(0.02)

    def test_reform_date_zero_realised(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Immediate reform (reform_date=0): always CPIH for realised."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=0.0, blend_period=0.0
        )
        assert curve.evaluate(0.0) == pytest.approx(0.02)
        assert curve.evaluate(5.0) == pytest.approx(0.02)

    def test_flat_curves_transition(self) -> None:
        """Flat RPI=4%, flat CPIH=2%, blend should go from 4 to 2."""
        rpi = ConstantCurve(0.04)
        cpih = ConstantCurve(0.02)
        curve = RpiReformBreakevenCurve(
            rpi, cpih, reform_start=10.0, blend_period=5.0
        )
        assert curve.evaluate(9.0) == pytest.approx(0.04)
        assert curve.evaluate(12.5) == pytest.approx(0.03)  # midpoint
        assert curve.evaluate(15.0) == pytest.approx(0.02)

    def test_negative_time_pre_reform(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Negative time values still return pure RPI."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        assert curve.evaluate(-1.0) == pytest.approx(0.03)

    def test_very_long_blend_period(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Long blend period (50 years) still transitions correctly."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=0.0, blend_period=50.0
        )
        assert curve.evaluate(-1.0) == pytest.approx(0.03)
        assert curve.evaluate(25.0) == pytest.approx(0.025)
        assert curve.evaluate(50.0) == pytest.approx(0.02)


# ─── Evaluate Interface ───────────────────────────────────────────


class TestEvaluateInterface:
    def test_breakeven_evaluate_callable(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Breakeven curve evaluate() is callable."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        result = curve.evaluate(3.0)
        assert isinstance(result, float)

    def test_realised_evaluate_callable(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Realised curve evaluate() is callable."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=1.0
        )
        result = curve.evaluate(3.0)
        assert isinstance(result, float)

    def test_breakeven_monotonic_during_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Breakeven transitions monotonically from RPI to CPIH."""
        curve = RpiReformBreakevenCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=2.0
        )
        prev = curve.evaluate(5.0)
        for i in range(1, 21):
            t = 5.0 + i * 0.1
            val = curve.evaluate(t)
            assert val <= prev + 1e-15  # non-increasing (RPI > CPIH)
            prev = val

    def test_realised_monotonic_during_blend(
        self, rpi_curve: ConstantCurve, cpih_curve: ConstantCurve
    ) -> None:
        """Realised transitions monotonically from RPI to CPIH."""
        curve = RpiReformRealisedCurve(
            rpi_curve, cpih_curve, reform_start=5.0, blend_period=2.0
        )
        prev = curve.evaluate(5.0)
        for i in range(1, 21):
            t = 5.0 + i * 0.1
            val = curve.evaluate(t)
            assert val <= prev + 1e-15
            prev = val
