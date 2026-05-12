"""RPI reform transition curves for UK inflation modelling.

The UK RPI index is scheduled to align with CPIH from February 2030.
This module provides blended transition curves for both breakeven
and realised inflation, using linear and polynomial (S-curve) blending.

Integration points:
    - Breakeven curves: use ``RpiReformBreakevenCurve`` to blend RPI and
      CPIH term-structure curves for bond pricing / ZCB construction.
    - Realised inflation: use ``RpiReformRealisedCurve`` to blend simulated
      RPI and CPIH realised paths for LPI / index-linked cashflows.
    - ``RpiReformConfig`` is the recommended entry point — it stores reform
      parameters and creates correctly-configured curve pairs.
"""

from __future__ import annotations

from hyesg.math.curves.blending import LinearBlend, PolynomialBlend
from hyesg.math.curves.protocol import ParametricCurve

# Reform effective date: February 2030 as year fraction from epoch
RPI_REFORM_DATE_YEARS: float = 2030 + 1.5 / 12


class _InstantSwitch(ParametricCurve):
    """Hard switch from curve f to curve g at a threshold time.

    Returns f(x) for x < threshold, g(x) for x >= threshold.
    Used when blend_period is zero (instantaneous transition).

    Args:
        f: Pre-switch curve.
        g: Post-switch curve.
        threshold: Switch point.
    """

    def __init__(
        self,
        f: ParametricCurve,
        g: ParametricCurve,
        threshold: float,
    ) -> None:
        self._f = f
        self._g = g
        self._threshold = threshold

    def evaluate(self, x: float) -> float:
        """Evaluate the switched curve at x.

        Args:
            x: The input value.

        Returns:
            f(x) if x < threshold, else g(x).
        """
        if x < self._threshold:
            return self._f.evaluate(x)
        return self._g.evaluate(x)


class RpiReformBreakevenCurve:
    """Blended breakeven inflation curve for RPI reform transition.

    Pre-reform: pure RPI breakeven.
    Post-reform: pure CPIH breakeven.
    During transition: linear blend.

    Implements ParametricCurve protocol via composition. The internal
    ``_blend`` attribute is a ``ParametricCurve`` that can be evaluated
    directly.

    Args:
        rpi_curve: Pre-reform RPI breakeven curve.
        cpih_curve: Post-reform CPIH breakeven curve.
        reform_start: Reform date in years from simulation start.
        blend_period: Transition period in years (0 = instantaneous).
    """

    def __init__(
        self,
        rpi_curve: ParametricCurve,
        cpih_curve: ParametricCurve,
        reform_start: float,
        blend_period: float = 0.0,
    ) -> None:
        self._rpi_curve = rpi_curve
        self._cpih_curve = cpih_curve
        self._reform_start = reform_start
        self._blend_period = blend_period

        if blend_period > 0.0:
            self._blend: ParametricCurve = LinearBlend(
                f=rpi_curve,
                g=cpih_curve,
                t_start=reform_start,
                t_end=reform_start + blend_period,
            )
        else:
            self._blend = _InstantSwitch(
                f=rpi_curve,
                g=cpih_curve,
                threshold=reform_start,
            )

    def evaluate(self, t: float) -> float:
        """Evaluate blended breakeven at time t.

        Args:
            t: Time in years from simulation start.

        Returns:
            Blended breakeven inflation value.
        """
        return self._blend.evaluate(t)


class RpiReformRealisedCurve:
    """Blended realised inflation curve for RPI reform transition.

    Same structure as breakeven but uses polynomial (S-curve) blending
    to ensure smooth transition with zero-derivative boundary conditions.

    Args:
        rpi_curve: Pre-reform RPI realised curve.
        cpih_curve: Post-reform CPIH realised curve.
        reform_start: Reform date in years from simulation start.
        blend_period: Transition period in years (0 = instantaneous).
    """

    def __init__(
        self,
        rpi_curve: ParametricCurve,
        cpih_curve: ParametricCurve,
        reform_start: float,
        blend_period: float = 1.0,
    ) -> None:
        self._rpi_curve = rpi_curve
        self._cpih_curve = cpih_curve
        self._reform_start = reform_start
        self._blend_period = blend_period

        if blend_period > 0.0:
            self._blend: ParametricCurve = PolynomialBlend(
                f=rpi_curve,
                g=cpih_curve,
                t_start=reform_start,
                t_end=reform_start + blend_period,
                degree=3,
            )
        else:
            self._blend = _InstantSwitch(
                f=rpi_curve,
                g=cpih_curve,
                threshold=reform_start,
            )

    def evaluate(self, t: float) -> float:
        """Evaluate blended realised inflation at time t.

        Args:
            t: Time in years from simulation start.

        Returns:
            Smoothly blended realised inflation value.
        """
        return self._blend.evaluate(t)


class RpiReformConfig:
    """Configuration for RPI reform transition.

    Holds all parameters needed for the reform transition and provides
    factory methods for creating breakeven and realised transition curves.

    Args:
        reform_date: Reform date in years from simulation start.
        blend_period_breakeven: Linear blend period for breakeven curves
            (0 = instantaneous switch).
        blend_period_realised: Polynomial blend period for realised curves
            (1 year smooth transition by default).
    """

    def __init__(
        self,
        reform_date: float,
        blend_period_breakeven: float = 0.0,
        blend_period_realised: float = 1.0,
    ) -> None:
        self._reform_date = reform_date
        self._blend_period_breakeven = blend_period_breakeven
        self._blend_period_realised = blend_period_realised

    @property
    def reform_date(self) -> float:
        """Reform date in years from simulation start."""
        return self._reform_date

    @property
    def blend_period_breakeven(self) -> float:
        """Blend period for breakeven curves."""
        return self._blend_period_breakeven

    @property
    def blend_period_realised(self) -> float:
        """Blend period for realised curves."""
        return self._blend_period_realised

    def create_breakeven_curve(
        self,
        rpi_curve: ParametricCurve,
        cpih_curve: ParametricCurve,
    ) -> RpiReformBreakevenCurve:
        """Create a breakeven transition curve with linear blending.

        Args:
            rpi_curve: Pre-reform RPI breakeven curve.
            cpih_curve: Post-reform CPIH breakeven curve.

        Returns:
            Configured RpiReformBreakevenCurve.
        """
        return RpiReformBreakevenCurve(
            rpi_curve=rpi_curve,
            cpih_curve=cpih_curve,
            reform_start=self._reform_date,
            blend_period=self._blend_period_breakeven,
        )

    def create_realised_curve(
        self,
        rpi_curve: ParametricCurve,
        cpih_curve: ParametricCurve,
    ) -> RpiReformRealisedCurve:
        """Create a realised transition curve with polynomial blending.

        Args:
            rpi_curve: Pre-reform RPI realised curve.
            cpih_curve: Post-reform CPIH realised curve.

        Returns:
            Configured RpiReformRealisedCurve.
        """
        return RpiReformRealisedCurve(
            rpi_curve=rpi_curve,
            cpih_curve=cpih_curve,
            reform_start=self._reform_date,
            blend_period=self._blend_period_realised,
        )
