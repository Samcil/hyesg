"""Primitive curve implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hyesg.math.curves.protocol import ParametricCurve

if TYPE_CHECKING:
    from collections.abc import Callable


class ConstantCurve(ParametricCurve):
    """Constant curve f(x) = c.

    Args:
        value: The constant value.
    """

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def evaluate(self, x: float) -> float:
        """Evaluate the curve at point x.

        Args:
            x: The input value (unused).

        Returns:
            The constant value c.
        """
        return self._value

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of a constant is zero.

        Args:
            x: The input value (unused).
            h: Step size (unused).

        Returns:
            0.0
        """
        return 0.0

    def integral(self, a: float, b: float) -> float:
        """Integral of constant c over [a,b] is c*(b-a).

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            c * (b - a).
        """
        return self._value * (b - a)

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return curve parameters."""
        return (self._value,)

    def with_parameters(self, params: tuple[float, ...]) -> ConstantCurve:
        """Return new ConstantCurve with updated value.

        Args:
            params: Tuple with the new constant value.

        Returns:
            New ConstantCurve instance.
        """
        return ConstantCurve(params[0])


class LinearCurve(ParametricCurve):
    """Linear curve f(x) = slope * x + intercept.

    Args:
        slope: The slope coefficient.
        intercept: The y-intercept.
    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        self._slope = slope
        self._intercept = intercept

    def evaluate(self, x: float) -> float:
        """Evaluate slope * x + intercept.

        Args:
            x: The input value.

        Returns:
            slope * x + intercept.
        """
        return self._slope * x + self._intercept

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of a linear function is the slope.

        Args:
            x: The input value (unused).
            h: Step size (unused).

        Returns:
            The slope.
        """
        return self._slope

    def integral(self, a: float, b: float) -> float:
        """Analytic integral of slope*x + intercept.

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            slope*(b²-a²)/2 + intercept*(b-a).
        """
        return self._slope * (b**2 - a**2) / 2 + self._intercept * (b - a)

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (slope, intercept)."""
        return (self._slope, self._intercept)

    def with_parameters(self, params: tuple[float, ...]) -> LinearCurve:
        """Return new LinearCurve with updated parameters.

        Args:
            params: Tuple of (slope,) or (slope, intercept).

        Returns:
            New LinearCurve instance.
        """
        return LinearCurve(params[0], params[1] if len(params) > 1 else 0.0)


class IdentityCurve(ParametricCurve):
    """Identity curve f(x) = x.

    Equivalent to LinearCurve(1, 0).
    """

    def evaluate(self, x: float) -> float:
        """Return x.

        Args:
            x: The input value.

        Returns:
            x unchanged.
        """
        return x

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of identity is 1.

        Args:
            x: The input value (unused).
            h: Step size (unused).

        Returns:
            1.0
        """
        return 1.0

    def integral(self, a: float, b: float) -> float:
        """Integral of x over [a,b] is (b²-a²)/2.

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            (b² - a²) / 2.
        """
        return (b**2 - a**2) / 2


class PiecewiseConstantCurve(ParametricCurve):
    """Step function defined by breakpoints and values.

    f(x) = values[i] where breakpoints[i] <= x < breakpoints[i+1].
    For x < breakpoints[0], returns values[0].
    For x >= breakpoints[-1], returns values[-1].

    Args:
        breakpoints: Strictly increasing x-coordinates.
        values: Function value on each interval.
    """

    def __init__(
        self,
        breakpoints: tuple[float, ...],
        values: tuple[float, ...],
    ) -> None:
        if len(breakpoints) != len(values):
            raise ValueError("breakpoints and values must have same length")
        self._breakpoints = breakpoints
        self._values = values

    def evaluate(self, x: float) -> float:
        """Evaluate the step function at x.

        Args:
            x: The input value.

        Returns:
            The step function value at x.
        """
        for i in range(len(self._breakpoints) - 1, -1, -1):
            if x >= self._breakpoints[i]:
                return self._values[i]
        return self._values[0]

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of a piecewise constant is zero (away from jumps).

        Args:
            x: The input value (unused).
            h: Step size (unused).

        Returns:
            0.0
        """
        return 0.0

    def integral(self, a: float, b: float) -> float:
        """Analytic integral of the step function over [a, b].

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            The definite integral.
        """
        if a >= b:
            return 0.0
        total = 0.0
        bps = self._breakpoints
        vals = self._values

        # Region before first breakpoint
        if a < bps[0]:
            total += vals[0] * (min(b, bps[0]) - a)

        # Each interval [bps[i], bps[i+1])
        for i in range(len(bps)):
            lower = bps[i]
            upper = bps[i + 1] if i + 1 < len(bps) else b
            lo = max(a, lower)
            hi = min(b, upper)
            if lo < hi:
                total += vals[i] * (hi - lo)

        # Region after last breakpoint (if not already covered)
        if b > bps[-1]:
            lo = max(a, bps[-1])
            # The last interval above already covers [bps[-1], b) via
            # upper=b, but only if i == len(bps)-1. Since we set
            # upper=b for the last i, this is already handled.
            pass

        return total


class HorizontallyShiftedCurve(ParametricCurve):
    """Horizontally shifted curve: g(x) = f(x + shift).

    Args:
        inner: The curve to shift.
        shift: Horizontal shift amount.
    """

    def __init__(self, inner: ParametricCurve, shift: float) -> None:
        self._inner = inner
        self._shift = shift

    def evaluate(self, x: float) -> float:
        """Evaluate f(x + shift).

        Args:
            x: The input value.

        Returns:
            f(x + shift).
        """
        return self._inner.evaluate(x + self._shift)

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of f(x + shift) is f'(x + shift).

        Args:
            x: The input value.
            h: Step size for finite differences.

        Returns:
            f'(x + shift).
        """
        return self._inner.derivative(x + self._shift, h)

    def integral(self, a: float, b: float) -> float:
        """Integral via change of variables.

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            ∫ₐᵇ f(x + shift) dx = ∫_{a+shift}^{b+shift} f(u) du.
        """
        return self._inner.integral(a + self._shift, b + self._shift)


class VerticallyShiftedCurve(ParametricCurve):
    """Vertically shifted curve: g(x) = f(x) + shift.

    Semantically equivalent to ScalarAdded but named for clarity.

    Args:
        inner: The curve to shift.
        shift: Vertical shift amount.
    """

    def __init__(self, inner: ParametricCurve, shift: float) -> None:
        self._inner = inner
        self._shift = shift

    def evaluate(self, x: float) -> float:
        """Evaluate f(x) + shift.

        Args:
            x: The input value.

        Returns:
            f(x) + shift.
        """
        return self._inner.evaluate(x) + self._shift

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of f(x) + c is f'(x).

        Args:
            x: The input value.
            h: Step size for finite differences.

        Returns:
            f'(x).
        """
        return self._inner.derivative(x, h)

    def integral(self, a: float, b: float) -> float:
        """Integral of f(x) + c is ∫f + c*(b-a).

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            ∫ₐᵇ f(x) dx + shift * (b - a).
        """
        return self._inner.integral(a, b) + self._shift * (b - a)


class InverseParametricCurve(ParametricCurve):
    """Inverse curve: g(x) = 1 / f(x).

    Named to match the C# ESG engine convention.

    Args:
        inner: The curve to invert.
    """

    def __init__(self, inner: ParametricCurve) -> None:
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate 1 / f(x).

        Args:
            x: The input value.

        Returns:
            1 / f(x).
        """
        return 1.0 / self._inner.evaluate(x)


class BlendedCurve(ParametricCurve):
    """Blended curve: g(x) = w(x)*f(x) + (1-w(x))*h(x).

    Uses a blending weight curve w to interpolate between two curves.

    Args:
        curve_a: First curve (weight = w).
        curve_b: Second curve (weight = 1-w).
        weight: Weight curve producing values in [0, 1].
    """

    def __init__(
        self,
        curve_a: ParametricCurve,
        curve_b: ParametricCurve,
        weight: ParametricCurve,
    ) -> None:
        self._a = curve_a
        self._b = curve_b
        self._weight = weight

    def evaluate(self, x: float) -> float:
        """Evaluate the blended curve at x.

        Args:
            x: The input value.

        Returns:
            w(x)*a(x) + (1-w(x))*b(x).
        """
        w = self._weight.evaluate(x)
        return w * self._a.evaluate(x) + (1.0 - w) * self._b.evaluate(x)


class IntegratedOverFixedIntervalCurve(ParametricCurve):
    """Curve that evaluates the integral of inner over [x, x+interval].

    g(x) = ∫ₓ^{x+interval} f(s) ds

    Args:
        inner: The curve to integrate.
        interval: Fixed integration interval width.
    """

    def __init__(self, inner: ParametricCurve, interval: float) -> None:
        self._inner = inner
        self._interval = interval

    def evaluate(self, x: float) -> float:
        """Evaluate ∫ₓ^{x+interval} f(s) ds.

        Args:
            x: Lower bound of the integration window.

        Returns:
            The definite integral over [x, x+interval].
        """
        return self._inner.integral(x, x + self._interval)
