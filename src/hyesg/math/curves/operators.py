"""Derived/decorator curve classes for algebraic composition."""

from __future__ import annotations

import math

from hyesg.math.curves.protocol import ParametricCurve


class Added(ParametricCurve):
    """Sum of two curves: (f + g)(x) = f(x) + g(x)."""

    def __init__(self, left: ParametricCurve, right: ParametricCurve) -> None:
        self._left = left
        self._right = right

    def evaluate(self, x: float) -> float:
        """Evaluate f(x) + g(x).

        Args:
            x: The input value.

        Returns:
            Sum of the two curves at x.
        """
        return self._left.evaluate(x) + self._right.evaluate(x)

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of sum is sum of derivatives.

        Args:
            x: The input value.
            h: Step size for finite differences.

        Returns:
            f'(x) + g'(x).
        """
        return self._left.derivative(x, h) + self._right.derivative(x, h)

    def integral(self, a: float, b: float) -> float:
        """Integral of sum is sum of integrals.

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            ∫f + ∫g.
        """
        return self._left.integral(a, b) + self._right.integral(a, b)


class Multiplied(ParametricCurve):
    """Product of two curves: (f * g)(x) = f(x) * g(x)."""

    def __init__(self, left: ParametricCurve, right: ParametricCurve) -> None:
        self._left = left
        self._right = right

    def evaluate(self, x: float) -> float:
        """Evaluate f(x) * g(x).

        Args:
            x: The input value.

        Returns:
            Product of the two curves at x.
        """
        return self._left.evaluate(x) * self._right.evaluate(x)


class Divided(ParametricCurve):
    """Quotient of two curves: (f / g)(x) = f(x) / g(x)."""

    def __init__(
        self, numerator: ParametricCurve, denominator: ParametricCurve
    ) -> None:
        self._numerator = numerator
        self._denominator = denominator

    def evaluate(self, x: float) -> float:
        """Evaluate f(x) / g(x).

        Args:
            x: The input value.

        Returns:
            Quotient of the two curves at x.
        """
        return self._numerator.evaluate(x) / self._denominator.evaluate(x)


class ScalarMultiplied(ParametricCurve):
    """Scalar multiple: (c * f)(x) = c * f(x)."""

    def __init__(self, inner: ParametricCurve, scalar: float) -> None:
        self._inner = inner
        self._scalar = scalar

    def evaluate(self, x: float) -> float:
        """Evaluate c * f(x).

        Args:
            x: The input value.

        Returns:
            Scalar times the inner curve at x.
        """
        return self._scalar * self._inner.evaluate(x)

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of c*f is c*f'.

        Args:
            x: The input value.
            h: Step size for finite differences.

        Returns:
            c * f'(x).
        """
        return self._scalar * self._inner.derivative(x, h)

    def integral(self, a: float, b: float) -> float:
        """Integral of c*f is c * ∫f.

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            c * ∫f.
        """
        return self._scalar * self._inner.integral(a, b)


class Composed(ParametricCurve):
    """Composition: f(g(x))."""

    def __init__(self, outer: ParametricCurve, inner: ParametricCurve) -> None:
        self._outer = outer
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate f(g(x)).

        Args:
            x: The input value.

        Returns:
            f(g(x)).
        """
        return self._outer.evaluate(self._inner.evaluate(x))


class Capped(ParametricCurve):
    """Capped curve: min(f(x), cap)."""

    def __init__(self, inner: ParametricCurve, cap: float) -> None:
        self._inner = inner
        self._cap = cap

    def evaluate(self, x: float) -> float:
        """Evaluate min(f(x), cap).

        Args:
            x: The input value.

        Returns:
            min(f(x), cap).
        """
        return min(self._inner.evaluate(x), self._cap)


class Floored(ParametricCurve):
    """Floored curve: max(f(x), floor)."""

    def __init__(self, inner: ParametricCurve, floor_val: float) -> None:
        self._inner = inner
        self._floor = floor_val

    def evaluate(self, x: float) -> float:
        """Evaluate max(f(x), floor).

        Args:
            x: The input value.

        Returns:
            max(f(x), floor).
        """
        return max(self._inner.evaluate(x), self._floor)


class Power(ParametricCurve):
    """Power curve: f(x)^n."""

    def __init__(self, inner: ParametricCurve, exponent: float) -> None:
        self._inner = inner
        self._exponent = exponent

    def evaluate(self, x: float) -> float:
        """Evaluate f(x)^n.

        Args:
            x: The input value.

        Returns:
            f(x) raised to the power n.
        """
        return self._inner.evaluate(x) ** self._exponent


class Exp(ParametricCurve):
    """Exponential: exp(f(x))."""

    def __init__(self, inner: ParametricCurve) -> None:
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate exp(f(x)).

        Args:
            x: The input value.

        Returns:
            exp(f(x)).
        """
        return math.exp(self._inner.evaluate(x))


class Log(ParametricCurve):
    """Natural logarithm: ln(f(x))."""

    def __init__(self, inner: ParametricCurve) -> None:
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate ln(f(x)).

        Args:
            x: The input value.

        Returns:
            ln(f(x)).
        """
        return math.log(self._inner.evaluate(x))


class Sin(ParametricCurve):
    """Sine: sin(f(x))."""

    def __init__(self, inner: ParametricCurve) -> None:
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate sin(f(x)).

        Args:
            x: The input value.

        Returns:
            sin(f(x)).
        """
        return math.sin(self._inner.evaluate(x))


class Cos(ParametricCurve):
    """Cosine: cos(f(x))."""

    def __init__(self, inner: ParametricCurve) -> None:
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate cos(f(x)).

        Args:
            x: The input value.

        Returns:
            cos(f(x)).
        """
        return math.cos(self._inner.evaluate(x))


class Tanh(ParametricCurve):
    """Hyperbolic tangent: tanh(f(x))."""

    def __init__(self, inner: ParametricCurve) -> None:
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate tanh(f(x)).

        Args:
            x: The input value.

        Returns:
            tanh(f(x)).
        """
        return math.tanh(self._inner.evaluate(x))


class Differentiated(ParametricCurve):
    """Numerical derivative curve: f'(x).

    Evaluates the inner curve's derivative at each point.
    """

    def __init__(self, inner: ParametricCurve, h: float = 1e-4) -> None:
        self._inner = inner
        self._h = h

    def evaluate(self, x: float) -> float:
        """Evaluate f'(x) using the inner curve's derivative.

        Args:
            x: The input value.

        Returns:
            f'(x).
        """
        return self._inner.derivative(x, self._h)


class Integrated(ParametricCurve):
    """Integral curve: ∫₀ˣ f(u)du.

    Evaluates the definite integral from 0 to x of the inner curve.
    """

    def __init__(self, inner: ParametricCurve) -> None:
        self._inner = inner

    def evaluate(self, x: float) -> float:
        """Evaluate ∫₀ˣ f(u)du.

        Args:
            x: The upper limit of integration.

        Returns:
            The definite integral from 0 to x.
        """
        if x == 0.0:
            return 0.0
        return self._inner.integral(0.0, x)


class Shifted(ParametricCurve):
    """Shifted curve: f(x + dx)."""

    def __init__(self, inner: ParametricCurve, dx: float) -> None:
        self._inner = inner
        self._dx = dx

    def evaluate(self, x: float) -> float:
        """Evaluate f(x + dx).

        Args:
            x: The input value.

        Returns:
            f(x + dx).
        """
        return self._inner.evaluate(x + self._dx)

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Derivative of f(x + dx) is f'(x + dx).

        Args:
            x: The input value.
            h: Step size for finite differences.

        Returns:
            f'(x + dx).
        """
        return self._inner.derivative(x + self._dx, h)


class Rounded(ParametricCurve):
    """Rounded curve: round(f(x), places)."""

    def __init__(self, inner: ParametricCurve, places: int) -> None:
        self._inner = inner
        self._places = places

    def evaluate(self, x: float) -> float:
        """Evaluate round(f(x), places).

        Args:
            x: The input value.

        Returns:
            f(x) rounded to the specified decimal places.
        """
        return round(self._inner.evaluate(x), self._places)
