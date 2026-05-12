"""Blending and extrapolation curve wrappers."""

from __future__ import annotations

from hyesg.math.curves.protocol import ParametricCurve


class PolynomialBlendingCurve(ParametricCurve):
    """Polynomial blending weight curve matching C# PolynomialBlendingCurve.

    Returns weight values decaying from 1.0 (at start_point) to 0.0
    (at end_point). The shape is a clamped cubic spline raised to a
    power, producing smooth C1-continuous transitions.

    The base spline is a cubic Hermite interpolant through (start, 1)
    and (end, 0) with zero first derivatives at both endpoints::

        S(t) = 1 - 3t² + 2t³   where t = (x - start) / (end - start)

    The final weight is S(t)^strength, which sharpens the decay for
    strength > 1.

    Args:
        start_point: x value where the weight begins to decay from 1.0.
        end_point: x value where the weight reaches 0.0.
        strength: Power exponent applied to the base spline (default 1.0).

    Raises:
        ValueError: If end_point <= start_point or strength <= 0.
    """

    def __init__(
        self,
        start_point: float,
        end_point: float,
        strength: float = 1.0,
    ) -> None:
        if end_point <= start_point:
            raise ValueError("end_point must be greater than start_point")
        if strength <= 0:
            raise ValueError("strength must be positive")
        self._start = start_point
        self._end = end_point
        self._strength = strength
        self._h = end_point - start_point

    @property
    def start_point(self) -> float:
        """Start of the blending region."""
        return self._start

    @property
    def end_point(self) -> float:
        """End of the blending region."""
        return self._end

    @property
    def strength(self) -> float:
        """Power exponent controlling decay sharpness."""
        return self._strength

    def evaluate(self, x: float) -> float:
        """Evaluate the blending weight at x.

        Args:
            x: The input value (typically time in years).

        Returns:
            Weight in [0, 1]. Returns 1.0 for x <= start_point,
            0.0 for x >= end_point, and S(t)^strength in between.
        """
        if x <= self._start:
            return 1.0
        if x >= self._end:
            return 0.0
        t = (x - self._start) / self._h
        # Clamped cubic Hermite: H00(t) = 2t³ - 3t² + 1
        spline = 1.0 - 3.0 * t * t + 2.0 * t * t * t
        if self._strength == 1.0:
            return spline
        return spline**self._strength


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
        if degree not in (3, 5):
            raise ValueError(
                f"Unsupported smoothstep degree {degree}; must be 3 or 5"
            )
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

        Raises:
            ValueError: If degree is not 3 or 5.
        """
        t = max(0.0, min(1.0, t))
        if self._degree == 3:
            # Cubic Hermite: 3t² - 2t³
            return t * t * (3.0 - 2.0 * t)
        if self._degree == 5:
            # Quintic: 6t⁵ - 15t⁴ + 10t³
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        raise ValueError(
            f"Unsupported smoothstep degree {self._degree}; must be 3 or 5"
        )

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


def _hermite_blend_weight(t: float, strength: float) -> float:
    """Cubic Hermite blend weight raised to a power.

    Computes h(t)^strength where h is the cubic Hermite basis
    function from (0, 1) to (1, 0) with zero derivatives at
    both endpoints:

        h(t) = 2t³ - 3t² + 1

    This matches the C# ``PolynomialBlendingCurve`` which uses
    ``CubicSpline.InterpolateBoundariesSorted`` with two points
    (start, 1.0) and (end, 0.0) and zero-derivative boundary
    conditions, then raises the result to ``Strength``.

    Args:
        t: Normalised position in [0, 1].
        strength: Exponent applied to the blend weight.

    Returns:
        Blend weight in [0, 1].
    """
    t = max(0.0, min(1.0, t))
    h = 2.0 * t * t * t - 3.0 * t * t + 1.0
    return h**strength


class SmoothConstantExtrapolation(ParametricCurve):
    """Smoothly blend a curve into a constant at extrapolation point.

    Matches C# ``CurveWithConstantExtrapolation``:

    - For x < extrapolation_point - blend_width: return curve(x)
    - For x > extrapolation_point: return constant
    - In between: smoothly blend using cubic Hermite^strength

    The constant defaults to the curve value at the extrapolation
    point. The blend weight uses a cubic Hermite basis function
    raised to ``strength`` (default 4), giving a smooth C¹ transition.

    Args:
        inner: The inner curve.
        extrapolation_point: Point beyond which we extrapolate.
        blend_width: Width of the blending region before the
            extrapolation point.
        strength: Exponent for the blend weight (default 4).
        constant: Extrapolation constant. If None, uses the curve
            value at the extrapolation point.
    """

    STRENGTH_DEFAULT: float = 4.0

    def __init__(
        self,
        inner: ParametricCurve,
        extrapolation_point: float,
        blend_width: float,
        strength: float = STRENGTH_DEFAULT,
        constant: float | None = None,
    ) -> None:
        if blend_width <= 0:
            raise ValueError("blend_width must be positive")
        self._inner = inner
        self._extrap_point = extrapolation_point
        self._blend_width = blend_width
        self._strength = strength
        self._blend_start = extrapolation_point - blend_width
        self._constant = (
            inner.evaluate(extrapolation_point)
            if constant is None
            else constant
        )

    def evaluate(self, x: float) -> float:
        """Evaluate the curve with smooth constant extrapolation.

        Args:
            x: The input value.

        Returns:
            Blended value transitioning from curve to constant.
        """
        if x <= self._blend_start:
            return self._inner.evaluate(x)
        if x >= self._extrap_point:
            return self._constant
        # t goes from 0 (at blend_start) to 1 (at extrap_point)
        t = (x - self._blend_start) / self._blend_width
        w = _hermite_blend_weight(t, self._strength)
        return w * self._inner.evaluate(x) + (1.0 - w) * self._constant
