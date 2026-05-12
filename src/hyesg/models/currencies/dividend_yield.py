"""DividendYield — pseudo-currency for equity dividend yields.

In the FCA framework, the dividend yield is treated as a
pseudo-currency where:
    - Short rate = continuous dividend yield
    - Exchange rate Q(t) = equity_level / initial_equity_level
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.outputs import OutputName


class DividendYield:
    """FCA dividend yield pseudo-currency.

    The dividend yield pseudo-currency enables unified pricing of
    equity total returns using the FCA framework.  The exchange rate
    is the equity price level relative to its initial value.

    Args:
        dividend_yield: Continuous dividend yield (constant or callable).
        equity_key: Key for the equity model in the state dict.
        initial_equity_level: Initial equity price level S(0).
    """

    def __init__(
        self,
        dividend_yield: float = 0.0,
        equity_key: str = "equity",
        initial_equity_level: float = 1.0,
    ) -> None:
        self._dividend_yield = dividend_yield
        self._equity_key = equity_key
        self._initial_level = initial_equity_level

    @property
    def equity_key(self) -> str:
        """Key for the equity model in the state dict."""
        return self._equity_key

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Cash account accumulated at the dividend yield.

        For a constant dividend yield q, this is exp(q * t).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Dividend cash account value.
        """
        q = jnp.asarray(self._dividend_yield, dtype=jnp.float64)
        return jnp.exp(q * jnp.asarray(t, dtype=jnp.float64))

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Exchange rate Q(t) = S(t) / S(0).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Equity level relative to initial level.
        """
        eq_state = state.get(self._equity_key, {})
        if isinstance(eq_state, dict):
            level = jnp.asarray(
                eq_state.get(OutputName.TOTAL_RETURN_INDEX, self._initial_level), dtype=jnp.float64
            )
        else:
            level = jnp.asarray(
                getattr(eq_state, "level", self._initial_level),
                dtype=jnp.float64,
            )
        return level / jnp.asarray(self._initial_level, dtype=jnp.float64)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Zero-coupon bond price using constant dividend yield.

        P_div(t, T) = exp(-q * (T - t))

        Args:
            state: Simulation state dict (unused for constant yield).
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Dividend ZCB price.
        """
        q = jnp.asarray(self._dividend_yield, dtype=jnp.float64)
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        return jnp.exp(-q * tau)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Spot rate: -ln(P_div(t,T)) / (T-t) = q (constant).

        Args:
            state: Simulation state dict (unused for constant yield).
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Continuously compounded dividend spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)
