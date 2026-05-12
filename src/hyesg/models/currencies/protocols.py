"""Protocol definitions for Foreign Currency Analogy (FCA) types.

The FCA treats real rates, inflation, dividends, and equity factors as
'pseudo-currencies' with exchange rates, enabling unified pricing via
zero-coupon bond prices.

Each pseudo-currency exposes:
    - cash_account: accumulated value of holding the pseudo short rate
    - exchange_to_base: exchange rate Q(t) to the domestic (base) currency
    - zcb_price: zero-coupon bond price in this pseudo-currency
    - spot_rate: continuously compounded rate derived from ZCB prices
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class CurrencyAnalogy(Protocol):
    """Protocol for FCA pseudo-currencies.

    Every pseudo-currency in the FCA framework must expose these four
    methods.  Implementations wrap underlying stochastic models (short
    rate, equity, inflation, FX) and present a uniform pricing interface.
    """

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Accumulator: exp(∫₀ᵗ r(s) ds) where r is the pseudo short rate.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Cash account value (scalar JAX array).
        """
        ...

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Q(t) — exchange rate from this pseudo-currency to base (domestic nominal).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Exchange rate to base currency (scalar JAX array).
        """
        ...

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Zero-coupon bond price in this pseudo-currency.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            ZCB price P(t, T) in this currency (scalar JAX array).
        """
        ...

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Spot rate derived from ZCB: -ln(P(t,T))/(T-t).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Continuously compounded spot rate (scalar JAX array).
        """
        ...
