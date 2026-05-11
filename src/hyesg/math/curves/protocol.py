"""ParametricCurve protocol and base implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self


class ParametricCurve(ABC):
    """Evaluable mathematical curve f: ℝ → ℝ with algebraic composition.

    Provides operator overloading for curve algebra:
        (f + g)(x) = f(x) + g(x)
        (f * g)(x) = f(x) * g(x)
        (f / g)(x) = f(x) / g(x)
        (-f)(x) = -f(x)

    Subclasses MUST implement evaluate(). derivative() and integral()
    have default implementations using numerical methods.
    """

    @abstractmethod
    def evaluate(self, x: float) -> float:
        """Evaluate the curve at point x.

        Args:
            x: The input value.

        Returns:
            The curve value at x.
        """
        ...

    def derivative(self, x: float, h: float = 1e-4) -> float:
        """Numerical derivative using central differences.

        Args:
            x: The point at which to evaluate the derivative.
            h: Step size for finite differences.

        Returns:
            Approximate derivative f'(x).
        """
        return (self.evaluate(x + h) - self.evaluate(x - h)) / (2 * h)

    def integral(self, a: float, b: float) -> float:
        """Numerical integral using composite Simpson's rule.

        For subclasses with analytic integrals, override this method.

        Args:
            a: Lower bound of integration.
            b: Upper bound of integration.

        Returns:
            Approximate value of ∫ₐᵇ f(x)dx.
        """
        n = 100  # number of intervals (must be even)
        h_step = (b - a) / n
        result = self.evaluate(a) + self.evaluate(b)
        for i in range(1, n):
            x = a + i * h_step
            coeff = 4.0 if i % 2 == 1 else 2.0
            result += coeff * self.evaluate(x)
        return result * h_step / 3.0

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return curve parameters. Override in parametric subclasses."""
        return ()

    def with_parameters(self, params: tuple[float, ...]) -> Self:
        """Return new instance with updated parameters.

        Args:
            params: New parameter values.

        Returns:
            New curve instance with updated parameters.

        Raises:
            NotImplementedError: If the subclass does not support this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support with_parameters()"
        )

    # ─── Operator Overloading ───

    def __add__(self, other: ParametricCurve | float) -> ParametricCurve:
        from hyesg.math.curves.operators import Added
        from hyesg.math.curves.primitives import ConstantCurve

        if isinstance(other, (int, float)):
            return Added(self, ConstantCurve(float(other)))
        return Added(self, other)

    def __radd__(self, other: float) -> ParametricCurve:
        from hyesg.math.curves.operators import Added
        from hyesg.math.curves.primitives import ConstantCurve

        return Added(ConstantCurve(float(other)), self)

    def __sub__(self, other: ParametricCurve | float) -> ParametricCurve:
        from hyesg.math.curves.operators import Added, ScalarMultiplied
        from hyesg.math.curves.primitives import ConstantCurve

        if isinstance(other, (int, float)):
            return Added(self, ScalarMultiplied(ConstantCurve(float(other)), -1.0))
        return Added(self, ScalarMultiplied(other, -1.0))

    def __rsub__(self, other: float) -> ParametricCurve:
        from hyesg.math.curves.operators import Added, ScalarMultiplied
        from hyesg.math.curves.primitives import ConstantCurve

        return Added(ConstantCurve(float(other)), ScalarMultiplied(self, -1.0))

    def __mul__(self, other: ParametricCurve | float) -> ParametricCurve:
        from hyesg.math.curves.operators import Multiplied, ScalarMultiplied

        if isinstance(other, (int, float)):
            return ScalarMultiplied(self, float(other))
        return Multiplied(self, other)

    def __rmul__(self, other: float) -> ParametricCurve:
        from hyesg.math.curves.operators import ScalarMultiplied

        return ScalarMultiplied(self, float(other))

    def __truediv__(self, other: ParametricCurve | float) -> ParametricCurve:
        from hyesg.math.curves.operators import Divided, ScalarMultiplied

        if isinstance(other, (int, float)):
            return ScalarMultiplied(self, 1.0 / float(other))
        return Divided(self, other)

    def __neg__(self) -> ParametricCurve:
        from hyesg.math.curves.operators import ScalarMultiplied

        return ScalarMultiplied(self, -1.0)

    def __pow__(self, n: float) -> ParametricCurve:
        from hyesg.math.curves.operators import Power

        return Power(self, n)

    # ─── Functional Transforms ───

    def exp(self) -> ParametricCurve:
        """Return exp(f(x))."""
        from hyesg.math.curves.operators import Exp

        return Exp(self)

    def log(self) -> ParametricCurve:
        """Return ln(f(x))."""
        from hyesg.math.curves.operators import Log

        return Log(self)

    def sin(self) -> ParametricCurve:
        """Return sin(f(x))."""
        from hyesg.math.curves.operators import Sin

        return Sin(self)

    def cos(self) -> ParametricCurve:
        """Return cos(f(x))."""
        from hyesg.math.curves.operators import Cos

        return Cos(self)

    def tanh(self) -> ParametricCurve:
        """Return tanh(f(x))."""
        from hyesg.math.curves.operators import Tanh

        return Tanh(self)

    def cap(self, upper: float) -> ParametricCurve:
        """Return min(f(x), upper).

        Args:
            upper: Upper cap value.

        Returns:
            Capped curve.
        """
        from hyesg.math.curves.operators import Capped

        return Capped(self, upper)

    def floor(self, lower: float) -> ParametricCurve:
        """Return max(f(x), lower).

        Args:
            lower: Lower floor value.

        Returns:
            Floored curve.
        """
        from hyesg.math.curves.operators import Floored

        return Floored(self, lower)

    def shift(self, dx: float) -> ParametricCurve:
        """Return f(x + dx).

        Args:
            dx: Horizontal shift amount.

        Returns:
            Shifted curve.
        """
        from hyesg.math.curves.operators import Shifted

        return Shifted(self, dx)

    def differentiate(self) -> ParametricCurve:
        """Return a curve representing f'(x)."""
        from hyesg.math.curves.operators import Differentiated

        return Differentiated(self)

    def integrate_curve(self) -> ParametricCurve:
        """Return a curve representing ∫₀ˣ f(u)du."""
        from hyesg.math.curves.operators import Integrated

        return Integrated(self)

    def compose(self, inner: ParametricCurve) -> ParametricCurve:
        """Return f(g(x)).

        Args:
            inner: The inner curve g.

        Returns:
            Composed curve f(g(x)).
        """
        from hyesg.math.curves.operators import Composed

        return Composed(self, inner)

    def round_to(self, places: int) -> ParametricCurve:
        """Return round(f(x), places).

        Args:
            places: Number of decimal places.

        Returns:
            Rounded curve.
        """
        from hyesg.math.curves.operators import Rounded

        return Rounded(self, places)

    def __call__(self, x: float) -> float:
        """Allow curve(x) syntax."""
        return self.evaluate(x)
