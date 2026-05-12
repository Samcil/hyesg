"""FactorWedge — pseudo-currency for equity factor excess returns.

Used for size, value, low-vol, quality, and other equity factor tilts.
The exchange rate Q(t) is the factor index level relative to its
initial value.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.outputs import OutputName


class FactorWedge:
    """FCA factor wedge pseudo-currency.

    Represents the excess return of an equity factor (e.g. size, value,
    low-vol) as a pseudo-currency.  The exchange rate is the factor
    index level and the pseudo short rate is the factor spread.

    Args:
        factor_spread: Constant factor excess return spread.
        factor_key: Key for the factor model in the state dict.
        initial_factor_level: Initial factor index level.
    """

    def __init__(
        self,
        factor_spread: float = 0.0,
        factor_key: str = "factor",
        initial_factor_level: float = 1.0,
    ) -> None:
        self._factor_spread = factor_spread
        self._factor_key = factor_key
        self._initial_level = initial_factor_level

    @property
    def factor_key(self) -> str:
        """Key for the factor model in the state dict."""
        return self._factor_key

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Cash account accumulated at the factor spread rate.

        For a constant spread s, this is exp(s * t).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Factor cash account value.
        """
        s = jnp.asarray(self._factor_spread, dtype=jnp.float64)
        return jnp.exp(s * jnp.asarray(t, dtype=jnp.float64))

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Exchange rate Q(t) = factor_index(t) / factor_index(0).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Factor index level relative to initial level.
        """
        factor_state = state.get(self._factor_key, {})
        if isinstance(factor_state, dict):
            level = jnp.asarray(
                factor_state.get(OutputName.TOTAL_RETURN_INDEX, self._initial_level),
                dtype=jnp.float64,
            )
        else:
            level = jnp.asarray(
                getattr(factor_state, "level", self._initial_level),
                dtype=jnp.float64,
            )
        return level / jnp.asarray(self._initial_level, dtype=jnp.float64)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Zero-coupon bond price using constant factor spread.

        P_factor(t, T) = exp(-spread * (T - t))

        Args:
            state: Simulation state dict (unused for constant spread).
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Factor ZCB price.
        """
        s = jnp.asarray(self._factor_spread, dtype=jnp.float64)
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        return jnp.exp(-s * tau)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Spot rate: -ln(P_factor(t,T)) / (T-t) = spread (constant).

        Args:
            state: Simulation state dict (unused for constant spread).
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Factor spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)
