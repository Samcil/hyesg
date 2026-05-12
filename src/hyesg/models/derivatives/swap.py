"""Interest rate swap pricing.

Provides FixedLeg, FloatingLeg, and Swap types for pricing
vanilla interest rate swaps using discount and forward curves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Callable


class FixedLeg(NamedTuple):
    """Fixed leg of an interest rate swap.

    Attributes:
        notional: Notional principal amount.
        rate: Fixed coupon rate (annualised).
        payment_dates: Tuple of payment times in years.
    """

    notional: float
    rate: float
    payment_dates: tuple[float, ...]


class FloatingLeg(NamedTuple):
    """Floating leg of an interest rate swap.

    Attributes:
        notional: Notional principal amount.
        index_ref: Label of the floating rate source (e.g. 'SONIA').
        spread: Spread over the floating index.
        payment_dates: Tuple of payment times in years.
    """

    notional: float
    index_ref: str
    spread: float
    payment_dates: tuple[float, ...]


class Swap:
    """Interest rate swap.

    A swap exchanges fixed-rate payments for floating-rate payments.
    The value is computed as the difference between the floating
    and fixed leg present values.
    """

    def __init__(self, fixed_leg: FixedLeg, floating_leg: FloatingLeg) -> None:
        """Initialise swap with fixed and floating legs.

        Args:
            fixed_leg: The fixed-rate payment leg.
            floating_leg: The floating-rate payment leg.
        """
        self.fixed_leg = fixed_leg
        self.floating_leg = floating_leg

    def fixed_leg_value(
        self,
        discount_curve: Callable[[float], Array],
        t: float,
    ) -> Array:
        """Present value of the fixed leg.

        PV_fixed = N * c * sum(df(t_i) * delta_i)

        Args:
            discount_curve: Function mapping time to discount factor.
            t: Valuation time.

        Returns:
            Present value of the fixed leg.
        """
        dates = jnp.array(self.fixed_leg.payment_dates)
        # Accrual periods (assume simple difference between dates)
        deltas = jnp.diff(jnp.concatenate([jnp.array([t]), dates]))
        dfs = jnp.array([discount_curve(float(d)) for d in dates])
        return self.fixed_leg.notional * self.fixed_leg.rate * jnp.sum(dfs * deltas)

    def floating_leg_value(
        self,
        discount_curve: Callable[[float], Array],
        forward_rates: Callable[[float, float], Array],
        t: float,
    ) -> Array:
        """Present value of the floating leg.

        PV_float = N * sum(df(t_i) * (fwd(t_{i-1}, t_i) + spread) * delta_i)

        Args:
            discount_curve: Function mapping time to discount factor.
            forward_rates: Function mapping (t_start, t_end) to forward rate.
            t: Valuation time.

        Returns:
            Present value of the floating leg.
        """
        dates = jnp.array(self.floating_leg.payment_dates)
        all_dates = jnp.concatenate([jnp.array([t]), dates])
        deltas = jnp.diff(all_dates)

        pv = jnp.float64(0.0)
        for i in range(len(dates)):
            t_start = float(all_dates[i])
            t_end = float(dates[i])
            df = discount_curve(t_end)
            fwd = forward_rates(t_start, t_end)
            pv = pv + df * (fwd + self.floating_leg.spread) * deltas[i]

        return self.floating_leg.notional * pv

    def value(
        self,
        discount_curve: Callable[[float], Array],
        forward_rates: Callable[[float, float], Array],
        t: float,
    ) -> Array:
        """Net present value of swap (receive float, pay fixed).

        NPV = PV_floating - PV_fixed

        Args:
            discount_curve: Function mapping time to discount factor.
            forward_rates: Function mapping (t_start, t_end) to forward rate.
            t: Valuation time.

        Returns:
            Net present value of the swap.
        """
        fixed_pv = self.fixed_leg_value(discount_curve, t)
        floating_pv = self.floating_leg_value(discount_curve, forward_rates, t)
        return floating_pv - fixed_pv
