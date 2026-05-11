"""Tests for operator/decorator curve classes."""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves import (
    ConstantCurve,
    LinearCurve,
)


class TestAddition:
    """Tests for curve addition."""

    def test_constant_plus_constant(self) -> None:
        result = ConstantCurve(2.0) + ConstantCurve(3.0)
        assert result.evaluate(0.0) == pytest.approx(5.0)
        assert result.evaluate(10.0) == pytest.approx(5.0)

    def test_curve_plus_scalar(self) -> None:
        result = ConstantCurve(2.0) + 3.0
        assert result.evaluate(0.0) == pytest.approx(5.0)

    def test_scalar_plus_curve(self) -> None:
        result = 3.0 + ConstantCurve(2.0)
        assert result.evaluate(0.0) == pytest.approx(5.0)

    def test_linear_plus_constant(self) -> None:
        result = LinearCurve(2.0, 0.0) + ConstantCurve(1.0)
        assert result.evaluate(3.0) == pytest.approx(7.0)

    def test_integral_of_sum(self) -> None:
        f = ConstantCurve(2.0) + ConstantCurve(3.0)
        assert f.integral(0.0, 1.0) == pytest.approx(5.0)


class TestSubtraction:
    """Tests for curve subtraction."""

    def test_constant_minus_constant(self) -> None:
        result = ConstantCurve(5.0) - ConstantCurve(3.0)
        assert result.evaluate(0.0) == pytest.approx(2.0)

    def test_curve_minus_scalar(self) -> None:
        result = ConstantCurve(5.0) - 3.0
        assert result.evaluate(0.0) == pytest.approx(2.0)

    def test_scalar_minus_curve(self) -> None:
        result = 10.0 - ConstantCurve(3.0)
        assert result.evaluate(0.0) == pytest.approx(7.0)


class TestMultiplication:
    """Tests for curve multiplication."""

    def test_scalar_multiply(self) -> None:
        result = ConstantCurve(3.0) * 2.0
        assert result.evaluate(0.0) == pytest.approx(6.0)

    def test_rscalar_multiply(self) -> None:
        result = 2.0 * ConstantCurve(3.0)
        assert result.evaluate(0.0) == pytest.approx(6.0)

    def test_curve_multiply(self) -> None:
        result = LinearCurve(1.0) * LinearCurve(1.0)
        # x * x = x²
        assert result.evaluate(3.0) == pytest.approx(9.0)

    def test_integral_of_scalar_multiply(self) -> None:
        result = ConstantCurve(3.0) * 2.0
        assert result.integral(0.0, 1.0) == pytest.approx(6.0)


class TestDivision:
    """Tests for curve division."""

    def test_scalar_divide(self) -> None:
        result = ConstantCurve(6.0) / 2.0
        assert result.evaluate(0.0) == pytest.approx(3.0)

    def test_curve_divide(self) -> None:
        result = ConstantCurve(6.0) / ConstantCurve(3.0)
        assert result.evaluate(0.0) == pytest.approx(2.0)

    def test_integrate_curve_over_linear(self) -> None:
        """Test spot rate from flat forward: ∫₀ᵗ f(u)du / t."""
        fwd = ConstantCurve(0.05)
        spot = fwd.integrate_curve() / LinearCurve()
        # For flat forward, spot should equal forward
        assert spot.evaluate(5.0) == pytest.approx(0.05, rel=1e-4)
        assert spot.evaluate(10.0) == pytest.approx(0.05, rel=1e-4)


class TestNegation:
    """Tests for curve negation."""

    def test_negate(self) -> None:
        result = -ConstantCurve(5.0)
        assert result.evaluate(0.0) == pytest.approx(-5.0)


class TestPower:
    """Tests for curve power."""

    def test_square(self) -> None:
        result = LinearCurve(1.0, 0.0) ** 2.0
        assert result.evaluate(3.0) == pytest.approx(9.0)
        assert result.evaluate(4.0) == pytest.approx(16.0)


class TestFunctionalTransforms:
    """Tests for exp, log, sin, cos, tanh."""

    def test_exp(self) -> None:
        result = ConstantCurve(1.0).exp()
        assert result.evaluate(0.0) == pytest.approx(math.e)

    def test_log(self) -> None:
        result = ConstantCurve(math.e).log()
        assert result.evaluate(0.0) == pytest.approx(1.0)

    def test_exp_log_roundtrip(self) -> None:
        c = ConstantCurve(3.0)
        result = c.exp().log()
        assert result.evaluate(0.0) == pytest.approx(3.0)

    def test_sin(self) -> None:
        result = ConstantCurve(math.pi / 2).sin()
        assert result.evaluate(0.0) == pytest.approx(1.0)

    def test_cos(self) -> None:
        result = ConstantCurve(0.0).cos()
        assert result.evaluate(0.0) == pytest.approx(1.0)

    def test_tanh(self) -> None:
        result = ConstantCurve(0.0).tanh()
        assert result.evaluate(0.0) == pytest.approx(0.0)


class TestCapFloor:
    """Tests for cap and floor."""

    def test_cap(self) -> None:
        result = LinearCurve(1.0).cap(5.0)
        assert result.evaluate(3.0) == pytest.approx(3.0)
        assert result.evaluate(7.0) == pytest.approx(5.0)
        assert result.evaluate(5.0) == pytest.approx(5.0)

    def test_floor(self) -> None:
        result = LinearCurve(1.0).floor(2.0)
        assert result.evaluate(3.0) == pytest.approx(3.0)
        assert result.evaluate(1.0) == pytest.approx(2.0)
        assert result.evaluate(2.0) == pytest.approx(2.0)


class TestShift:
    """Tests for curve shifting."""

    def test_shift(self) -> None:
        result = LinearCurve(1.0, 0.0).shift(3.0)
        # f(x+3) = x+3
        assert result.evaluate(0.0) == pytest.approx(3.0)
        assert result.evaluate(2.0) == pytest.approx(5.0)


class TestDifferentiated:
    """Tests for differentiated curve."""

    def test_differentiate_linear(self) -> None:
        result = LinearCurve(2.0, 3.0).differentiate()
        assert result.evaluate(0.0) == pytest.approx(2.0, abs=1e-6)
        assert result.evaluate(5.0) == pytest.approx(2.0, abs=1e-6)

    def test_differentiate_quadratic(self) -> None:
        # x² → 2x
        quad = LinearCurve(1.0) ** 2
        deriv = quad.differentiate()
        assert deriv.evaluate(3.0) == pytest.approx(6.0, abs=1e-3)


class TestIntegrated:
    """Tests for integrated curve."""

    def test_integrate_constant(self) -> None:
        result = ConstantCurve(5.0).integrate_curve()
        # ∫₀ˣ 5 du = 5x
        assert result.evaluate(0.0) == pytest.approx(0.0)
        assert result.evaluate(2.0) == pytest.approx(10.0)
        assert result.evaluate(5.0) == pytest.approx(25.0)

    def test_integrate_linear(self) -> None:
        result = LinearCurve(1.0, 0.0).integrate_curve()
        # ∫₀ˣ u du = x²/2
        assert result.evaluate(2.0) == pytest.approx(2.0, rel=1e-4)
        assert result.evaluate(4.0) == pytest.approx(8.0, rel=1e-4)


class TestComposed:
    """Tests for curve composition."""

    def test_compose(self) -> None:
        outer = LinearCurve(2.0, 1.0)  # 2x + 1
        inner = LinearCurve(3.0, 0.0)  # 3x
        result = outer.compose(inner)
        # f(g(x)) = 2(3x) + 1 = 6x + 1
        assert result.evaluate(1.0) == pytest.approx(7.0)
        assert result.evaluate(2.0) == pytest.approx(13.0)


class TestRounded:
    """Tests for rounded curve."""

    def test_round_to_places(self) -> None:
        result = ConstantCurve(3.14159).round_to(2)
        assert result.evaluate(0.0) == pytest.approx(3.14)

    def test_round_to_integer(self) -> None:
        result = ConstantCurve(3.7).round_to(0)
        assert result.evaluate(0.0) == pytest.approx(4.0)


class TestComplexComposition:
    """Tests for complex algebraic compositions."""

    def test_spot_from_forward(self) -> None:
        """Flat forward → flat spot via ∫₀ᵗ f/t."""
        fwd = ConstantCurve(0.05)
        spot = fwd.integrate_curve() / LinearCurve()
        for t in [1.0, 5.0, 10.0, 30.0]:
            assert spot.evaluate(t) == pytest.approx(0.05, rel=1e-4)

    def test_zcb_from_forward(self) -> None:
        """Flat forward → ZCB price = exp(-r*t)."""
        fwd = ConstantCurve(0.05)
        zcb = (-fwd.integrate_curve()).exp()
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(-0.05 * t)
            assert zcb.evaluate(t) == pytest.approx(expected, rel=1e-4)
