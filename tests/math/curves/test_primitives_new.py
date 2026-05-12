"""Tests for new primitive curve types."""

from __future__ import annotations

import pytest

from hyesg.math.curves import (
    BlendedCurve,
    ConstantCurve,
    HorizontallyShiftedCurve,
    IntegratedOverFixedIntervalCurve,
    InverseParametricCurve,
    LinearCurve,
    PiecewiseConstantCurve,
    VerticallyShiftedCurve,
)


class TestPiecewiseConstantCurve:
    """Tests for PiecewiseConstantCurve."""

    def test_evaluate_within_intervals(self) -> None:
        c = PiecewiseConstantCurve((0.0, 1.0, 3.0), (10.0, 20.0, 30.0))
        assert c.evaluate(0.5) == 10.0
        assert c.evaluate(1.0) == 20.0
        assert c.evaluate(2.0) == 20.0
        assert c.evaluate(3.0) == 30.0
        assert c.evaluate(5.0) == 30.0

    def test_evaluate_before_first_breakpoint(self) -> None:
        c = PiecewiseConstantCurve((1.0, 2.0), (5.0, 10.0))
        assert c.evaluate(0.0) == 5.0
        assert c.evaluate(-10.0) == 5.0

    def test_evaluate_after_last_breakpoint(self) -> None:
        c = PiecewiseConstantCurve((1.0, 2.0), (5.0, 10.0))
        assert c.evaluate(3.0) == 10.0
        assert c.evaluate(100.0) == 10.0

    def test_derivative_is_zero(self) -> None:
        c = PiecewiseConstantCurve((0.0, 1.0), (5.0, 10.0))
        assert c.derivative(0.5) == 0.0
        assert c.derivative(1.5) == 0.0

    def test_integral_single_interval(self) -> None:
        c = PiecewiseConstantCurve((0.0,), (5.0,))
        assert c.integral(0.0, 2.0) == pytest.approx(10.0)

    def test_integral_across_intervals(self) -> None:
        c = PiecewiseConstantCurve((0.0, 1.0), (2.0, 4.0))
        # [0, 1): value=2, width=1 → 2
        # [1, 3): value=4, width=2 → 8
        assert c.integral(0.0, 3.0) == pytest.approx(10.0)

    def test_integral_before_first_breakpoint(self) -> None:
        c = PiecewiseConstantCurve((1.0,), (5.0,))
        # Before bp: value=5, width=1 → 5
        assert c.integral(0.0, 1.0) == pytest.approx(5.0)

    def test_callable_syntax(self) -> None:
        c = PiecewiseConstantCurve((0.0,), (7.0,))
        assert c(1.0) == 7.0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            PiecewiseConstantCurve((0.0, 1.0), (5.0,))


class TestHorizontallyShiftedCurve:
    """Tests for HorizontallyShiftedCurve."""

    def test_evaluate(self) -> None:
        inner = LinearCurve(2.0, 1.0)  # 2x + 1
        shifted = HorizontallyShiftedCurve(inner, 3.0)
        # g(x) = f(x+3) = 2(x+3)+1 = 2x+7
        assert shifted.evaluate(0.0) == pytest.approx(7.0)
        assert shifted.evaluate(1.0) == pytest.approx(9.0)

    def test_derivative(self) -> None:
        inner = LinearCurve(2.0, 1.0)
        shifted = HorizontallyShiftedCurve(inner, 3.0)
        assert shifted.derivative(0.0) == pytest.approx(2.0, abs=1e-6)

    def test_integral(self) -> None:
        inner = ConstantCurve(5.0)
        shifted = HorizontallyShiftedCurve(inner, 10.0)
        assert shifted.integral(0.0, 2.0) == pytest.approx(10.0)

    def test_convenience_method(self) -> None:
        inner = LinearCurve(1.0, 0.0)
        shifted = inner.shift_horizontal(5.0)
        assert shifted.evaluate(0.0) == pytest.approx(5.0)


class TestVerticallyShiftedCurve:
    """Tests for VerticallyShiftedCurve."""

    def test_evaluate(self) -> None:
        inner = LinearCurve(2.0, 0.0)  # 2x
        shifted = VerticallyShiftedCurve(inner, 10.0)
        assert shifted.evaluate(0.0) == pytest.approx(10.0)
        assert shifted.evaluate(3.0) == pytest.approx(16.0)

    def test_derivative(self) -> None:
        inner = LinearCurve(2.0, 0.0)
        shifted = VerticallyShiftedCurve(inner, 10.0)
        assert shifted.derivative(0.0) == pytest.approx(2.0, abs=1e-6)

    def test_integral(self) -> None:
        inner = ConstantCurve(3.0)
        shifted = VerticallyShiftedCurve(inner, 2.0)
        # ∫₀¹ (3+2)dx = 5
        assert shifted.integral(0.0, 1.0) == pytest.approx(5.0)

    def test_convenience_method(self) -> None:
        inner = ConstantCurve(5.0)
        shifted = inner.shift_vertical(3.0)
        assert shifted.evaluate(0.0) == pytest.approx(8.0)


class TestInverseParametricCurve:
    """Tests for InverseParametricCurve."""

    def test_evaluate(self) -> None:
        inner = ConstantCurve(4.0)
        inv = InverseParametricCurve(inner)
        assert inv.evaluate(0.0) == pytest.approx(0.25)

    def test_inverse_of_linear(self) -> None:
        inner = LinearCurve(1.0, 1.0)  # x + 1
        inv = InverseParametricCurve(inner)
        assert inv.evaluate(1.0) == pytest.approx(0.5)  # 1/(1+1)
        assert inv.evaluate(3.0) == pytest.approx(0.25)  # 1/(3+1)


class TestBlendedCurve:
    """Tests for BlendedCurve."""

    def test_full_weight_a(self) -> None:
        a = ConstantCurve(10.0)
        b = ConstantCurve(20.0)
        w = ConstantCurve(1.0)  # 100% a
        blended = BlendedCurve(a, b, w)
        assert blended.evaluate(0.0) == pytest.approx(10.0)

    def test_full_weight_b(self) -> None:
        a = ConstantCurve(10.0)
        b = ConstantCurve(20.0)
        w = ConstantCurve(0.0)  # 100% b
        blended = BlendedCurve(a, b, w)
        assert blended.evaluate(0.0) == pytest.approx(20.0)

    def test_half_blend(self) -> None:
        a = ConstantCurve(10.0)
        b = ConstantCurve(20.0)
        w = ConstantCurve(0.5)
        blended = BlendedCurve(a, b, w)
        assert blended.evaluate(0.0) == pytest.approx(15.0)


class TestIntegratedOverFixedIntervalCurve:
    """Tests for IntegratedOverFixedIntervalCurve."""

    def test_constant_integral(self) -> None:
        inner = ConstantCurve(3.0)
        c = IntegratedOverFixedIntervalCurve(inner, 2.0)
        # ∫ₓ^{x+2} 3 ds = 6 for any x
        assert c.evaluate(0.0) == pytest.approx(6.0)
        assert c.evaluate(5.0) == pytest.approx(6.0)

    def test_linear_integral(self) -> None:
        inner = LinearCurve(1.0, 0.0)  # f(s) = s
        c = IntegratedOverFixedIntervalCurve(inner, 1.0)
        # ∫₀¹ s ds = 0.5
        assert c.evaluate(0.0) == pytest.approx(0.5)
        # ∫₂³ s ds = (9-4)/2 = 2.5
        assert c.evaluate(2.0) == pytest.approx(2.5)
