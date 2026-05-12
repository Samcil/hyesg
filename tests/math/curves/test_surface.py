"""Tests for ParametricSurface."""

from __future__ import annotations

import pytest

from hyesg.math.curves import ConstantCurve, LinearCurve, ParametricSurface


class TestParametricSurface:
    """Tests for ParametricSurface."""

    def test_evaluate_at_exact_y(self) -> None:
        def factory(y: float) -> ConstantCurve:
            return ConstantCurve(y * 10.0)

        surf = ParametricSurface(factory, (1.0, 2.0, 3.0))
        assert surf.evaluate(0.0, 1.0) == pytest.approx(10.0)
        assert surf.evaluate(0.0, 2.0) == pytest.approx(20.0)
        assert surf.evaluate(0.0, 3.0) == pytest.approx(30.0)

    def test_interpolates_between_y_values(self) -> None:
        def factory(y: float) -> ConstantCurve:
            return ConstantCurve(y * 10.0)

        surf = ParametricSurface(factory, (1.0, 3.0))
        # At y=2.0, interpolate between 10 and 30 → 20
        assert surf.evaluate(0.0, 2.0) == pytest.approx(20.0)

    def test_extrapolates_below_min_y(self) -> None:
        def factory(y: float) -> ConstantCurve:
            return ConstantCurve(y * 10.0)

        surf = ParametricSurface(factory, (1.0, 2.0))
        assert surf.evaluate(0.0, 0.0) == pytest.approx(10.0)
        assert surf.evaluate(0.0, -5.0) == pytest.approx(10.0)

    def test_extrapolates_above_max_y(self) -> None:
        def factory(y: float) -> ConstantCurve:
            return ConstantCurve(y * 10.0)

        surf = ParametricSurface(factory, (1.0, 2.0))
        assert surf.evaluate(0.0, 5.0) == pytest.approx(20.0)

    def test_x_dependency(self) -> None:
        def factory(y: float) -> LinearCurve:
            return LinearCurve(slope=y, intercept=0.0)

        surf = ParametricSurface(factory, (1.0, 2.0))
        # At y=1.0, curve is x → 1*x
        assert surf.evaluate(5.0, 1.0) == pytest.approx(5.0)
        # At y=2.0, curve is x → 2*x
        assert surf.evaluate(5.0, 2.0) == pytest.approx(10.0)
        # At y=1.5, interpolate: 0.5 * 5.0 + 0.5 * 10.0 = 7.5
        assert surf.evaluate(5.0, 1.5) == pytest.approx(7.5)

    def test_slice_at_y_returns_curve(self) -> None:
        def factory(y: float) -> ConstantCurve:
            return ConstantCurve(y)

        surf = ParametricSurface(factory, (1.0, 2.0, 3.0))
        curve = surf.slice_at_y(2.0)
        assert curve.evaluate(0.0) == pytest.approx(2.0)

    def test_slice_at_missing_y_raises(self) -> None:
        def factory(y: float) -> ConstantCurve:
            return ConstantCurve(y)

        surf = ParametricSurface(factory, (1.0, 2.0))
        with pytest.raises(KeyError, match="No curve at y=1.5"):
            surf.slice_at_y(1.5)

    def test_unordered_y_values_sorted(self) -> None:
        def factory(y: float) -> ConstantCurve:
            return ConstantCurve(y * 10.0)

        surf = ParametricSurface(factory, (3.0, 1.0, 2.0))
        # Should still work — y_values are sorted internally
        assert surf.evaluate(0.0, 1.5) == pytest.approx(15.0)

    def test_imports_from_package(self) -> None:
        from hyesg.math.curves import ParametricSurface as PS

        assert PS is not None
