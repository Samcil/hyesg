"""Fourier-based seasonal adjustment for parametric curves.

Used for inflation seasonality in the ESG engine.
"""

from __future__ import annotations

import math

from hyesg.math.curves import ParametricCurve


class _SeasonalAdjustmentCurve(ParametricCurve):
    """Curve representing the Fourier seasonal adjustment.

    adjustment(t) = Σ_k (a_k cos(2πkt) + b_k sin(2πkt))
    """

    def __init__(
        self,
        coeffs_cos: tuple[float, ...],
        coeffs_sin: tuple[float, ...],
    ) -> None:
        self._coeffs_cos = coeffs_cos
        self._coeffs_sin = coeffs_sin

    def evaluate(self, x: float) -> float:
        """Evaluate the seasonal adjustment at time x.

        Args:
            x: Time point.

        Returns:
            Seasonal adjustment value.
        """
        total = 0.0
        for k, (a_k, b_k) in enumerate(
            zip(self._coeffs_cos, self._coeffs_sin), start=1
        ):
            angle = 2.0 * math.pi * k * x
            total += a_k * math.cos(angle) + b_k * math.sin(angle)
        return total


class FourierSeasonalityAdjuster:
    """Adjusts a curve for seasonal patterns using Fourier series.

    Implements:
        adjustment(t) = Σ_k (a_k cos(2πkt) + b_k sin(2πkt))

    Used for inflation seasonality in the C# ESG engine.

    Args:
        coeffs_cos: Cosine coefficients (a_1, a_2, ...).
        coeffs_sin: Sine coefficients (b_1, b_2, ...).

    Raises:
        ValueError: If coefficient tuples have different lengths.
    """

    def __init__(
        self,
        coeffs_cos: tuple[float, ...],
        coeffs_sin: tuple[float, ...],
    ) -> None:
        if len(coeffs_cos) != len(coeffs_sin):
            raise ValueError(
                "coeffs_cos and coeffs_sin must have the same length, "
                f"got {len(coeffs_cos)} and {len(coeffs_sin)}"
            )
        self._coeffs_cos = coeffs_cos
        self._coeffs_sin = coeffs_sin
        self._adjustment_curve = _SeasonalAdjustmentCurve(coeffs_cos, coeffs_sin)

    def adjustment(self, t: float) -> float:
        """Evaluate the seasonal adjustment at time t.

        Args:
            t: Time point.

        Returns:
            The Fourier series adjustment value at t.
        """
        return self._adjustment_curve.evaluate(t)

    def adjust_curve(self, curve: ParametricCurve) -> ParametricCurve:
        """Return a new curve equal to original + seasonal adjustment.

        Args:
            curve: The base curve to adjust.

        Returns:
            Adjusted curve: curve(t) + adjustment(t).
        """
        return curve + self._adjustment_curve
