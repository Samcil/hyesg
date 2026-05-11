"""Tests for spline curve implementations."""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves import AkimaCubicSpline, CubicSpline


class TestCubicSpline:
    """Tests for natural cubic spline."""

    def test_passes_through_knots(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 1.0, 0.5, 2.0, 1.5]
        spline = CubicSpline(xs, ys)
        for x, y in zip(xs, ys, strict=True):
            assert spline.evaluate(x) == pytest.approx(y, abs=1e-12)

    def test_flat_extrapolation_left(self) -> None:
        xs = [0.0, 1.0, 2.0]
        ys = [1.0, 2.0, 3.0]
        spline = CubicSpline(xs, ys)
        assert spline.evaluate(-1.0) == pytest.approx(1.0)
        assert spline.evaluate(-100.0) == pytest.approx(1.0)

    def test_flat_extrapolation_right(self) -> None:
        xs = [0.0, 1.0, 2.0]
        ys = [1.0, 2.0, 3.0]
        spline = CubicSpline(xs, ys)
        assert spline.evaluate(3.0) == pytest.approx(3.0)
        assert spline.evaluate(100.0) == pytest.approx(3.0)

    def test_linear_data_gives_linear(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 2.0, 4.0, 6.0]
        spline = CubicSpline(xs, ys)
        # Midpoints should be linear
        assert spline.evaluate(0.5) == pytest.approx(1.0, abs=1e-10)
        assert spline.evaluate(1.5) == pytest.approx(3.0, abs=1e-10)
        assert spline.evaluate(2.5) == pytest.approx(5.0, abs=1e-10)

    def test_smooth_interpolation(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 0.0, 1.0]
        spline = CubicSpline(xs, ys)
        # Just check it returns finite values
        for x in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5]:
            val = spline.evaluate(x)
            assert math.isfinite(val)

    def test_two_points(self) -> None:
        xs = [0.0, 1.0]
        ys = [0.0, 1.0]
        spline = CubicSpline(xs, ys)
        assert spline.evaluate(0.0) == pytest.approx(0.0)
        assert spline.evaluate(1.0) == pytest.approx(1.0)

    def test_callable_syntax(self) -> None:
        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 1.0, 0.0]
        spline = CubicSpline(xs, ys)
        assert spline(0.0) == pytest.approx(0.0)

    def test_validation_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            CubicSpline([0.0, 1.0], [0.0])

    def test_validation_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            CubicSpline([0.0], [0.0])


class TestAkimaCubicSpline:
    """Tests for Akima cubic spline."""

    def test_passes_through_knots(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 1.0, 0.5, 2.0, 1.5]
        spline = AkimaCubicSpline(xs, ys)
        for x, y in zip(xs, ys, strict=True):
            assert spline.evaluate(x) == pytest.approx(y, abs=1e-12)

    def test_flat_extrapolation(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 2.0, 3.0, 4.0]
        spline = AkimaCubicSpline(xs, ys)
        assert spline.evaluate(-1.0) == pytest.approx(1.0)
        assert spline.evaluate(5.0) == pytest.approx(4.0)

    def test_monotone_data_stays_monotone(self) -> None:
        """Akima should reduce oscillation near monotone data."""
        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        spline = AkimaCubicSpline(xs, ys)
        # Check monotonicity at intermediate points
        prev = spline.evaluate(0.0)
        for x_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            curr = spline.evaluate(x_val)
            assert curr >= prev - 1e-10
            prev = curr

    def test_reduces_oscillation(self) -> None:
        """Akima should handle outliers better than natural spline."""
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 0.0, 10.0, 0.0, 0.0]  # spike at x=2
        spline = AkimaCubicSpline(xs, ys)
        # Values between 0 and 1 should stay close to 0
        val = spline.evaluate(0.5)
        assert abs(val) < 5.0  # shouldn't overshoot much

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            AkimaCubicSpline([0.0, 1.0], [0.0])
        with pytest.raises(ValueError, match="at least 2"):
            AkimaCubicSpline([0.0], [0.0])
