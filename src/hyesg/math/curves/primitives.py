"""Primitive curve implementations."""

from __future__ import annotations

from hyesg.math.curves.protocol import ParametricCurve


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
