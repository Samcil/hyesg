"""Composition classes that combine FCA pseudo-currencies.

Each composition pairs two FCA types to produce a derived economy:
    - NominalAndExchangeRate: nominal + FX → foreign economy
    - RealRateAndInflation: real rate + inflation → real economy
    - EquityAndDividendYield: equity + dividend → total return
    - WedgeCurrencyAndInflationWedge: RPI reform wedge
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.models.currencies.protocols import CurrencyAnalogy


class NominalAndExchangeRate:
    """Composition: nominal rate + FX rate → foreign economy output.

    The composed exchange rate is the product of the nominal exchange
    rate and the FX exchange rate:
        Q_composed(t) = Q_nominal(t) * Q_fx(t)

    The composed ZCB price is the product of the individual ZCB prices,
    adjusted by the exchange rate ratio.

    Args:
        nominal: Nominal pseudo-currency (domestic or foreign).
        fx_currency: FX pseudo-currency providing the exchange rate.
    """

    def __init__(
        self,
        nominal: CurrencyAnalogy,
        fx_currency: CurrencyAnalogy,
    ) -> None:
        self._nominal = nominal
        self._fx = fx_currency

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Composed cash account: product of both cash accounts.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Composed cash account value.
        """
        return self._nominal.cash_account(state, t) * self._fx.cash_account(
            state, t
        )

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Composed exchange rate: Q_nominal * Q_fx.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Composed exchange rate.
        """
        return self._nominal.exchange_to_base(
            state, t
        ) * self._fx.exchange_to_base(state, t)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed ZCB price: product of nominal and FX ZCB prices.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Composed ZCB price.
        """
        return self._nominal.zcb_price(
            state, t, T
        ) * self._fx.zcb_price(state, t, T)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed spot rate derived from composed ZCB price.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Composed spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)


class RealRateAndInflation:
    """Composition: real rate + inflation index → real economy output.

    The composed exchange rate is the product of the real rate
    exchange rate (price index) and the inflation exchange rate.

    Args:
        real_rate: Real rate pseudo-currency.
        inflation: Inflation pseudo-currency providing the price index.
    """

    def __init__(
        self,
        real_rate: CurrencyAnalogy,
        inflation: CurrencyAnalogy,
    ) -> None:
        self._real = real_rate
        self._inflation = inflation

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Composed cash account: product of both cash accounts.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Composed cash account value.
        """
        return self._real.cash_account(
            state, t
        ) * self._inflation.cash_account(state, t)

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Composed exchange rate: Q_real * Q_inflation.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Composed exchange rate.
        """
        return self._real.exchange_to_base(
            state, t
        ) * self._inflation.exchange_to_base(state, t)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed ZCB price: product of real and inflation ZCB prices.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Composed ZCB price.
        """
        return self._real.zcb_price(
            state, t, T
        ) * self._inflation.zcb_price(state, t, T)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed spot rate derived from composed ZCB price.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Composed spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)


class EquityAndDividendYield:
    """Composition: equity level + dividend yield → total return.

    Combines the equity exchange rate (price level) with the
    dividend yield pseudo-currency to get total return metrics.

    Args:
        equity: Equity price pseudo-currency.
        dividend: Dividend yield pseudo-currency.
    """

    def __init__(
        self,
        equity: CurrencyAnalogy,
        dividend: CurrencyAnalogy,
    ) -> None:
        self._equity = equity
        self._dividend = dividend

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Composed cash account: product of both cash accounts.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Total return cash account value.
        """
        return self._equity.cash_account(
            state, t
        ) * self._dividend.cash_account(state, t)

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Composed exchange rate: Q_equity * Q_dividend.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Total return exchange rate.
        """
        return self._equity.exchange_to_base(
            state, t
        ) * self._dividend.exchange_to_base(state, t)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed ZCB price: product of equity and dividend ZCB prices.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Total return ZCB price.
        """
        return self._equity.zcb_price(
            state, t, T
        ) * self._dividend.zcb_price(state, t, T)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed spot rate derived from composed ZCB price.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Total return spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)


class WedgeCurrencyAndInflationWedge:
    """Composition: RPI reform wedge pseudo-currency.

    Combines two pseudo-currencies to represent the RPI reform
    wedge — the gap between unreformed and reformed RPI.

    Args:
        wedge: Wedge pseudo-currency (e.g. RPI reform adjustment).
        inflation_wedge: Inflation wedge pseudo-currency.
    """

    def __init__(
        self,
        wedge: CurrencyAnalogy,
        inflation_wedge: CurrencyAnalogy,
    ) -> None:
        self._wedge = wedge
        self._inflation_wedge = inflation_wedge

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Composed cash account: product of both cash accounts.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Wedge cash account value.
        """
        return self._wedge.cash_account(
            state, t
        ) * self._inflation_wedge.cash_account(state, t)

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Composed exchange rate: Q_wedge * Q_inflation_wedge.

        Args:
            state: Simulation state dict.
            t: Current time in years.

        Returns:
            Composed wedge exchange rate.
        """
        return self._wedge.exchange_to_base(
            state, t
        ) * self._inflation_wedge.exchange_to_base(state, t)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed ZCB price: product of wedge ZCB prices.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Composed wedge ZCB price.
        """
        return self._wedge.zcb_price(
            state, t, T
        ) * self._inflation_wedge.zcb_price(state, t, T)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Composed spot rate derived from composed ZCB price.

        Args:
            state: Simulation state dict.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Composed wedge spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)
