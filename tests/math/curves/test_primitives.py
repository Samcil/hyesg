"""Tests for primitive curve implementations."""

from __future__ import annotations

import pytest

from hyesg.math.curves import ConstantCurve, IdentityCurve, LinearCurve


class TestConstantCurve:
    """Tests for ConstantCurve."""

    def test_evaluate_returns_constant(self) -> None:
        c = ConstantCurve(5.0)
        assert c.evaluate(0.0) == 5.0
        assert c.evaluate(1.0) == 5.0
        assert c.evaluate(-100.0) == 5.0
        assert c.evaluate(1e6) == 5.0

    def test_callable_syntax(self) -> None:
        c = ConstantCurve(3.0)
        assert c(0.0) == 3.0
        assert c(42.0) == 3.0

    def test_derivative_is_zero(self) -> None:
        c = ConstantCurve(5.0)
        assert c.derivative(0.0) == 0.0
        assert c.derivative(10.0) == 0.0

    def test_integral(self) -> None:
        c = ConstantCurve(5.0)
        assert c.integral(0.0, 1.0) == pytest.approx(5.0)
        assert c.integral(2.0, 5.0) == pytest.approx(15.0)
        assert c.integral(-1.0, 1.0) == pytest.approx(10.0)

    def test_default_value(self) -> None:
        c = ConstantCurve()
        assert c.evaluate(0.0) == 0.0

    def test_parameters(self) -> None:
        c = ConstantCurve(7.0)
        assert c.parameters == (7.0,)

    def test_with_parameters(self) -> None:
        c = ConstantCurve(5.0)
        c2 = c.with_parameters((10.0,))
        assert c2.evaluate(0.0) == 10.0
        assert c.evaluate(0.0) == 5.0  # original unchanged


class TestLinearCurve:
    """Tests for LinearCurve."""

    def test_evaluate(self) -> None:
        line = LinearCurve(2.0, 3.0)
        assert line.evaluate(0.0) == pytest.approx(3.0)
        assert line.evaluate(1.0) == pytest.approx(5.0)
        assert line.evaluate(-1.0) == pytest.approx(1.0)
        assert line.evaluate(10.0) == pytest.approx(23.0)

    def test_derivative_is_slope(self) -> None:
        line = LinearCurve(2.0, 3.0)
        assert line.derivative(0.0) == 2.0
        assert line.derivative(100.0) == 2.0

    def test_integral(self) -> None:
        line = LinearCurve(2.0, 3.0)
        # ∫₀¹ (2x + 3)dx = [x² + 3x]₀¹ = 1 + 3 = 4
        assert line.integral(0.0, 1.0) == pytest.approx(4.0)
        # ∫₀² (2x + 3)dx = [x² + 3x]₀² = 4 + 6 = 10
        assert line.integral(0.0, 2.0) == pytest.approx(10.0)

    def test_default_values(self) -> None:
        line = LinearCurve()
        assert line.evaluate(5.0) == pytest.approx(5.0)

    def test_parameters(self) -> None:
        line = LinearCurve(2.0, 3.0)
        assert line.parameters == (2.0, 3.0)

    def test_with_parameters(self) -> None:
        line = LinearCurve(2.0, 3.0)
        line2 = line.with_parameters((5.0, 1.0))
        assert line2.evaluate(1.0) == pytest.approx(6.0)


class TestIdentityCurve:
    """Tests for IdentityCurve."""

    def test_evaluate_returns_x(self) -> None:
        ident = IdentityCurve()
        assert ident.evaluate(0.0) == 0.0
        assert ident.evaluate(5.0) == 5.0
        assert ident.evaluate(-3.0) == -3.0

    def test_derivative_is_one(self) -> None:
        ident = IdentityCurve()
        assert ident.derivative(0.0) == 1.0
        assert ident.derivative(100.0) == 1.0

    def test_integral(self) -> None:
        ident = IdentityCurve()
        # ∫₀¹ x dx = 0.5
        assert ident.integral(0.0, 1.0) == pytest.approx(0.5)
        # ∫₁³ x dx = (9 - 1)/2 = 4
        assert ident.integral(1.0, 3.0) == pytest.approx(4.0)
