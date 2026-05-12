"""Tests for inverse cancellation optimisations in curve algebra."""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves import ConstantCurve, LinearCurve
from hyesg.math.curves.operators import (
    Differentiated,
    Exp,
    Integrated,
    Log,
    Power,
    ScalarMultiplied,
)


class TestExpLogCancellation:
    """Tests that exp(log(f)) and log(exp(f)) cancel to f."""

    def test_exp_of_log_returns_inner(self) -> None:
        f = ConstantCurve(3.0)
        result = f.log().exp()
        # Should return f directly, not Exp(Log(f))
        assert result is f

    def test_log_of_exp_returns_inner(self) -> None:
        f = ConstantCurve(2.0)
        result = f.exp().log()
        assert result is f

    def test_exp_of_log_evaluates_correctly(self) -> None:
        f = LinearCurve(1.0, 1.0)
        result = f.log().exp()
        assert result.evaluate(2.0) == pytest.approx(3.0)

    def test_log_of_exp_evaluates_correctly(self) -> None:
        f = ConstantCurve(0.5)
        result = f.exp().log()
        assert result.evaluate(0.0) == pytest.approx(0.5)

    def test_exp_without_log_still_works(self) -> None:
        f = ConstantCurve(1.0)
        result = f.exp()
        assert isinstance(result, Exp)
        assert result.evaluate(0.0) == pytest.approx(math.e)

    def test_log_without_exp_still_works(self) -> None:
        f = ConstantCurve(math.e)
        result = f.log()
        assert isinstance(result, Log)
        assert result.evaluate(0.0) == pytest.approx(1.0)


class TestDifferentiateIntegrateCancellation:
    """Tests that differentiate(integrate(f)) cancels."""

    def test_differentiate_of_integrated_returns_inner(self) -> None:
        f = ConstantCurve(5.0)
        result = f.integrate_curve().differentiate()
        assert result is f

    def test_integrate_of_differentiated_returns_inner(self) -> None:
        f = LinearCurve(2.0, 3.0)
        result = f.differentiate().integrate_curve()
        assert result is f

    def test_differentiate_without_integrated_still_works(self) -> None:
        f = LinearCurve(2.0, 3.0)
        result = f.differentiate()
        assert isinstance(result, Differentiated)

    def test_integrate_without_differentiated_still_works(self) -> None:
        f = ConstantCurve(5.0)
        result = f.integrate_curve()
        assert isinstance(result, Integrated)


class TestPowerCancellation:
    """Tests that Power(Power(f, a), b) = Power(f, a*b)."""

    def test_power_of_power_collapses(self) -> None:
        f = LinearCurve(1.0, 1.0)  # x + 1
        # (f^2)^3 should become f^6
        result = (f**2) ** 3
        assert isinstance(result, Power)
        assert result._exponent == pytest.approx(6.0)

    def test_power_of_power_evaluates_correctly(self) -> None:
        f = ConstantCurve(2.0)
        result = (f**2) ** 3  # 2^6 = 64
        assert result.evaluate(0.0) == pytest.approx(64.0)

    def test_single_power_unchanged(self) -> None:
        f = ConstantCurve(3.0)
        result = f**2
        assert isinstance(result, Power)
        assert result._exponent == pytest.approx(2.0)


class TestScalarMultiplyCancellation:
    """Tests that ScalarMultiplied(ScalarMultiplied(f, a), b) collapses."""

    def test_double_scalar_multiply_collapses(self) -> None:
        f = LinearCurve(1.0, 0.0)
        result = (f * 3.0) * 4.0
        assert isinstance(result, ScalarMultiplied)
        assert result._scalar == pytest.approx(12.0)

    def test_double_scalar_multiply_evaluates_correctly(self) -> None:
        f = ConstantCurve(2.0)
        result = (f * 3.0) * 4.0
        assert result.evaluate(0.0) == pytest.approx(24.0)

    def test_rmul_also_collapses(self) -> None:
        f = LinearCurve(1.0, 0.0)
        result = 4.0 * (3.0 * f)
        assert isinstance(result, ScalarMultiplied)
        assert result._scalar == pytest.approx(12.0)


class TestNegatedNegatedCancellation:
    """Tests that -(-f) returns f."""

    def test_double_negation_returns_inner(self) -> None:
        f = ConstantCurve(5.0)
        neg_f = -f
        result = -neg_f
        # -(-f) = ScalarMultiplied(ScalarMultiplied(f, -1), -1)
        # The inner is ScalarMultiplied(f, -1) with scalar=-1
        # So neg detects scalar==-1 and returns inner
        assert result is f

    def test_double_negation_evaluates_correctly(self) -> None:
        f = LinearCurve(2.0, 3.0)
        result = -(-f)
        assert result.evaluate(1.0) == pytest.approx(5.0)

    def test_single_negation_still_works(self) -> None:
        f = ConstantCurve(5.0)
        result = -f
        assert isinstance(result, ScalarMultiplied)
        assert result.evaluate(0.0) == pytest.approx(-5.0)
