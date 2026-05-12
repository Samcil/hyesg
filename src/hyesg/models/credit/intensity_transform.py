"""Risk-neutral to real-world intensity transform.

Maps CIR++ risk-neutral default intensities to real-world intensities
using a monotonic sub-linear compression via natural cubic spline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

from hyesg.math.curves.splines import CubicSpline

if TYPE_CHECKING:
    from jax import Array


@runtime_checkable
class IntensityTransform(Protocol):
    """Protocol for RN→RW intensity transforms."""

    def transform(self, rn_intensity: Array) -> Array:
        """Map risk-neutral intensity to real-world intensity.

        Args:
            rn_intensity: Risk-neutral default intensity (scalar or array).

        Returns:
            Real-world intensity of the same shape.
        """
        ...


class SplineIntensityTransform:
    """RN→RW intensity transform using cubic spline sub-linear compression.

    Uses a natural cubic spline through calibrated knot points.  The
    transform is monotonically increasing and sub-linear for large
    intensities (output < input), reflecting the empirical observation
    that real-world default rates are lower than risk-neutral rates.

    Args:
        knot_xs: X coordinates (risk-neutral intensities), sorted ascending.
        knot_ys: Y coordinates (real-world intensities) at each knot.
    """

    def __init__(self, knot_xs: list[float], knot_ys: list[float]) -> None:
        self._spline = CubicSpline(knot_xs, knot_ys)
        self._knot_xs = knot_xs
        self._knot_ys = knot_ys

    def transform(self, rn_intensity: Array) -> Array:
        """Map risk-neutral intensity to real-world intensity.

        Evaluates the cubic spline at each element of ``rn_intensity``.
        Uses flat extrapolation beyond the knot range (inherent in
        ``CubicSpline``).

        Args:
            rn_intensity: Risk-neutral default intensity (scalar or array).

        Returns:
            Real-world intensity of the same shape.
        """
        rn_float = float(jnp.asarray(rn_intensity))
        result = self._spline.evaluate(rn_float)
        return jnp.asarray(result, dtype=jnp.float64)

    @property
    def knot_xs(self) -> list[float]:
        """X coordinates of the transform spline."""
        return list(self._knot_xs)

    @property
    def knot_ys(self) -> list[float]:
        """Y coordinates of the transform spline."""
        return list(self._knot_ys)


class ScaledIntensityTransform:
    """Scaled version of an intensity transform.

    Multiplies the base transform output by a constant scale factor.
    Used for liquidity RN→RW transforms where the base credit spline
    is scaled by 0.1.

    Args:
        base_transform: Underlying intensity transform.
        scale_factor: Multiplicative scale factor.
    """

    def __init__(
        self,
        base_transform: IntensityTransform,
        scale_factor: float,
    ) -> None:
        self._base = base_transform
        self._scale = scale_factor

    def transform(self, rn_intensity: Array) -> Array:
        """Apply base transform then scale.

        Args:
            rn_intensity: Risk-neutral intensity.

        Returns:
            Scaled real-world intensity.
        """
        base_result = self._base.transform(rn_intensity)
        return base_result * self._scale

    @property
    def scale_factor(self) -> float:
        """Scale factor applied to base transform."""
        return self._scale
