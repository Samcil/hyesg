"""Blending and extrapolation curve wrappers."""

from __future__ import annotations

from hyesg.math.curves.protocol import ParametricCurve


class LinearBlend(ParametricCurve):
    """Linearly blend from curve f to curve g over [t_start, t_end].

    At t_start, returns f(x). At t_end, returns g(x).
    Between them, linearly interpolates: w*g(x) + (1-w)*f(x)
    where w = (x - t_start) / (t_end - t_start).
    Outside the blend region, returns f for x < t_start, g for x > t_end.

    Args:
        f: The starting curve.
        g: The ending curve.
        t_start: Start of the blend region.
        t_end: End of the blend region.
    """

    def __init__(
        self,
        f: ParametricCurve,
        g: ParametricCurve,
        t_start: float,
        t_end: float,
    ) -> None:
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start")
        self._f = f
        self._g = g
        self._t_start = t_start
        self._t_end = t_end

    def evaluate(self, x: float) -> float:
        """Evaluate the blended curve at x.

        Args:
            x: The input value.

        Returns:
            Blended value between f and g.
        """
        if x <= self._t_start:
            return self._f.evaluate(x)
        if x >= self._t_end:
            return self._g.evaluate(x)
        w = (x - self._t_start) / (self._t_end - self._t_start)
        return (1.0 - w) * self._f.evaluate(x) + w * self._g.evaluate(x)


class PolynomialBlend(ParametricCurve):
    """Smooth polynomial blending from curve f to curve g.

    Uses Hermite-style polynomial for C¹-continuous blending.
    The blend weight uses a smoothstep polynomial of the specified degree.

    Args:
        f: The starting curve.
        g: The ending curve.
        t_start: Start of the blend region.
        t_end: End of the blend region.
        degree: Polynomial degree for the smoothstep (default 3).
    """

    def __init__(
        self,
        f: ParametricCurve,
        g: ParametricCurve,
        t_start: float,
        t_end: float,
        degree: int = 3,
    ) -> None:
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start")
        self._f = f
        self._g = g
        self._t_start = t_start
        self._t_end = t_end
        self._degree = degree

    def _smoothstep(self, t: float) -> float:
        """Compute smoothstep polynomial weight.

        Args:
            t: Normalised blend parameter in [0, 1].

        Returns:
            Smooth weight in [0, 1].
        """
        t = max(0.0, min(1.0, t))
        if self._degree == 3:
            # Cubic Hermite: 3t² - 2t³
            return t * t * (3.0 - 2.0 * t)
        if self._degree == 5:
            # Quintic: 6t⁵ - 15t⁴ + 10t³
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        # Fallback to cubic
        return t * t * (3.0 - 2.0 * t)

    def evaluate(self, x: float) -> float:
        """Evaluate the smoothly blended curve at x.

        Args:
            x: The input value.

        Returns:
            Smoothly blended value between f and g.
        """
        if x <= self._t_start:
            return self._f.evaluate(x)
        if x >= self._t_end:
            return self._g.evaluate(x)
        t = (x - self._t_start) / (self._t_end - self._t_start)
        w = self._smoothstep(t)
        return (1.0 - w) * self._f.evaluate(x) + w * self._g.evaluate(x)


class ConstantExtrapolation(ParametricCurve):
    """Use inner curve within [x_min, x_max], constant outside.

    Returns the endpoint values of the inner curve for x values
    outside the specified range.

    Args:
        inner: The inner curve.
        x_min: Lower bound of the active range.
        x_max: Upper bound of the active range.
    """

    def __init__(
        self,
        inner: ParametricCurve,
        x_min: float,
        x_max: float,
    ) -> None:
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")
        self._inner = inner
        self._x_min = x_min
        self._x_max = x_max

    def evaluate(self, x: float) -> float:
        """Evaluate the curve with constant extrapolation.

        Args:
            x: The input value.

        Returns:
            Inner curve value if in range, endpoint value otherwise.
        """
        if x <= self._x_min:
            return self._inner.evaluate(self._x_min)
        if x >= self._x_max:
            return self._inner.evaluate(self._x_max)
        return self._inner.evaluate(x)
