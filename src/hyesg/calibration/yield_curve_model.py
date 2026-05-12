"""Yield curve model providing forward/spot/ZCB/invZCB representations.

Matches the C# ``InitialYieldCurveModel`` hierarchy, providing a
unified interface to access all curve representations with consistent
compounding conventions.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.transforms import (
    forward_to_inverse_zcbp,
    forward_to_spot,
    forward_to_zcbp,
)


class InitialYieldCurveModel(NamedTuple):
    """Unified yield curve model with all representations.

    The forward curve is the primary representation. Spot, ZCB price,
    and accumulation factor curves are derived from it.

    Attributes:
        forward_curve: Continuously compounded instantaneous forward rate f(t).
        spot_curve: Continuously compounded spot rate s(t) = (1/t)∫₀ᵗ f(u)du.
        zcbp_curve: Zero-coupon bond price P(t) = exp(-∫₀ᵗ f(u)du).
        inv_zcbp_curve: Accumulation factor 1/P(t) = exp(∫₀ᵗ f(u)du).
    """

    forward_curve: ParametricCurve
    spot_curve: ParametricCurve
    zcbp_curve: ParametricCurve
    inv_zcbp_curve: ParametricCurve

    @classmethod
    def from_forward_curve(cls, fwd: ParametricCurve) -> InitialYieldCurveModel:
        """Build all representations from the forward rate curve.

        Args:
            fwd: Continuously compounded instantaneous forward rate curve.

        Returns:
            Complete yield curve model.
        """
        return cls(
            forward_curve=fwd,
            spot_curve=forward_to_spot(fwd),
            zcbp_curve=forward_to_zcbp(fwd),
            inv_zcbp_curve=forward_to_inverse_zcbp(fwd),
        )

    def forward_rate(self, t: float) -> float:
        """Evaluate the instantaneous forward rate at maturity t.

        Args:
            t: Time to maturity in years.

        Returns:
            Continuously compounded instantaneous forward rate.
        """
        return self.forward_curve.evaluate(t)

    def spot_rate(self, t: float, compounding: str = "continuous") -> float:
        """Evaluate the spot rate at maturity t.

        Args:
            t: Time to maturity in years.
            compounding: ``'continuous'``, ``'annual'``, or ``'semi_annual'``.

        Returns:
            Spot rate in the requested compounding convention.

        Raises:
            ValueError: If an unknown compounding convention is given.
        """
        cts_spot = self.spot_curve.evaluate(t)
        if compounding == "continuous":
            return cts_spot
        if compounding == "annual":
            return math.exp(cts_spot) - 1.0
        if compounding == "semi_annual":
            return 2.0 * (math.exp(cts_spot / 2.0) - 1.0)
        raise ValueError(f"Unknown compounding convention: {compounding!r}")

    def zcb_price(self, t: float) -> float:
        """Evaluate the zero-coupon bond price at maturity t.

        Args:
            t: Time to maturity in years.

        Returns:
            ZCB price P(t).
        """
        return self.zcbp_curve.evaluate(t)

    def accumulation_factor(self, t: float) -> float:
        """Evaluate the accumulation factor at maturity t.

        Args:
            t: Time to maturity in years.

        Returns:
            Accumulation factor 1/P(t).
        """
        return self.inv_zcbp_curve.evaluate(t)
