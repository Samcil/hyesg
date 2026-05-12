"""ExchangeRateAnalogyChain — chains FX rates for cross-currency.

Chains multiple pseudo-currencies to compute cross-currency exchange
rates as the product of individual Q values:
    Q_base = ∏ₖ Q_k

Example: GBP→USD→EUR = Q_usd_gbp * Q_eur_usd
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.models.currencies.protocols import CurrencyAnalogy


class ExchangeRateAnalogyChain:
    """Chain of pseudo-currencies for cross-currency FX computation.

    Multiplies exchange rates from multiple pseudo-currencies to
    derive a cross rate.  ZCB prices are similarly chained.

    Args:
        currencies: Ordered sequence of pseudo-currencies to chain.
            The first currency's exchange rate is multiplied by
            the second, then the third, etc.
    """

    def __init__(self, currencies: list[CurrencyAnalogy]) -> None:
        if not currencies:
            raise ValueError("Chain requires at least one currency")
        self._currencies = currencies

    @property
    def currencies(self) -> list[CurrencyAnalogy]:
        """The ordered sequence of pseudo-currencies in the chain."""
        return self._currencies

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Chained cash account: product across all currencies.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Product of all cash accounts in the chain.
        """
        result = jnp.array(1.0, dtype=jnp.float64)
        for currency in self._currencies:
            result = result * currency.cash_account(state, t)
        return result

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Chained exchange rate: Q_base = ∏ₖ Q_k.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Product of all exchange rates in the chain.
        """
        result = jnp.array(1.0, dtype=jnp.float64)
        for currency in self._currencies:
            result = result * currency.exchange_to_base(state, t)
        return result

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Chained ZCB price: product across all currencies.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Product of all ZCB prices in the chain.
        """
        result = jnp.array(1.0, dtype=jnp.float64)
        for currency in self._currencies:
            result = result * currency.zcb_price(state, t, T)
        return result

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Chained spot rate derived from chained ZCB price.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Chained spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)
