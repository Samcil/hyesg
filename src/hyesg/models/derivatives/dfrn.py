"""Discounted Floating Rate Note (DFRN) pricing.

A DFRN pays the floating rate plus a spread, discounted at the
risk-free rate. At inception its value equals the notional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array


class DFRN:
    """Discounted Floating Rate Note.

    A floating-rate instrument that pays par at maturity plus
    periodic floating coupons with a fixed spread.

    Attributes:
        notional: Face value of the note.
        spread: Fixed spread over the floating index.
        maturity: Time to maturity in years.
    """

    def __init__(self, notional: float, spread: float, maturity: float) -> None:
        """Initialise DFRN.

        Args:
            notional: Face value of the note.
            spread: Fixed spread over the floating index.
            maturity: Time to maturity in years.
        """
        self.notional = notional
        self.spread = spread
        self.maturity = maturity

    def value(self, discount_factor: Array, t: float) -> Array:
        """Compute the present value of the DFRN.

        At par (no credit risk, spread = 0), PV = notional * df.
        The spread contribution adds notional * spread * (T-t) * df
        as a first-order approximation.

        Args:
            discount_factor: Discount factor from t to maturity.
            t: Current valuation time.

        Returns:
            Present value of the DFRN.
        """
        tau = self.maturity - t
        # Par value plus spread accrual, discounted
        return self.notional * discount_factor * (1.0 + self.spread * tau)
