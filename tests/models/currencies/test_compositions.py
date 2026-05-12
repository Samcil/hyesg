"""Tests for FCA composition classes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hyesg.models.currencies.compositions import (
    EquityAndDividendYield,
    NominalAndExchangeRate,
    RealRateAndInflation,
    WedgeCurrencyAndInflationWedge,
)
from hyesg.models.currencies.dividend_yield import DividendYield
from hyesg.models.currencies.factor_wedge import FactorWedge


class _StubCurrency:
    """Stub currency with configurable return values."""

    def __init__(
        self,
        cash: float = 1.0,
        exchange: float = 1.0,
        zcb: float = 1.0,
    ) -> None:
        self._cash = cash
        self._exchange = exchange
        self._zcb = zcb

    def cash_account(self, state, t):
        return jnp.asarray(self._cash, dtype=jnp.float64)

    def exchange_to_base(self, state, t):
        return jnp.asarray(self._exchange, dtype=jnp.float64)

    def zcb_price(self, state, t, T):
        return jnp.asarray(self._zcb, dtype=jnp.float64)

    def spot_rate(self, state, t, T):
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        return -jnp.log(self.zcb_price(state, t, T)) / jnp.maximum(tau, 1e-12)


class TestNominalAndExchangeRate:
    """Tests for NominalAndExchangeRate composition."""

    def test_exchange_rate_is_product(self):
        """Composed exchange rate is product of both."""
        a = _StubCurrency(exchange=1.5)
        b = _StubCurrency(exchange=0.8)
        comp = NominalAndExchangeRate(a, b)
        q = comp.exchange_to_base({}, t=1.0)
        assert jnp.isclose(q, 1.5 * 0.8)

    def test_cash_account_is_product(self):
        """Composed cash account is product of both."""
        a = _StubCurrency(cash=1.2)
        b = _StubCurrency(cash=1.3)
        comp = NominalAndExchangeRate(a, b)
        result = comp.cash_account({}, t=1.0)
        assert jnp.isclose(result, 1.2 * 1.3)

    def test_zcb_price_is_product(self):
        """Composed ZCB price is product of both."""
        a = _StubCurrency(zcb=0.9)
        b = _StubCurrency(zcb=0.95)
        comp = NominalAndExchangeRate(a, b)
        p = comp.zcb_price({}, t=0.0, T=5.0)
        assert jnp.isclose(p, 0.9 * 0.95)

    def test_spot_rate_from_composed_zcb(self):
        """Spot rate is derived from composed ZCB price."""
        a = _StubCurrency(zcb=0.9)
        b = _StubCurrency(zcb=0.95)
        comp = NominalAndExchangeRate(a, b)
        r = comp.spot_rate({}, t=0.0, T=5.0)
        expected = -jnp.log(0.9 * 0.95) / 5.0
        assert jnp.isclose(r, expected, atol=1e-10)


class TestRealRateAndInflation:
    """Tests for RealRateAndInflation composition."""

    def test_exchange_rate_is_product(self):
        """Composed exchange rate is product of both."""
        a = _StubCurrency(exchange=1.1)
        b = _StubCurrency(exchange=1.05)
        comp = RealRateAndInflation(a, b)
        q = comp.exchange_to_base({}, t=1.0)
        assert jnp.isclose(q, 1.1 * 1.05)

    def test_zcb_price_is_product(self):
        """Composed ZCB price is product of both."""
        a = _StubCurrency(zcb=0.85)
        b = _StubCurrency(zcb=0.92)
        comp = RealRateAndInflation(a, b)
        p = comp.zcb_price({}, t=0.0, T=10.0)
        assert jnp.isclose(p, 0.85 * 0.92)

    def test_cash_account_is_product(self):
        """Composed cash account is product of both."""
        a = _StubCurrency(cash=1.02)
        b = _StubCurrency(cash=1.03)
        comp = RealRateAndInflation(a, b)
        result = comp.cash_account({}, t=1.0)
        assert jnp.isclose(result, 1.02 * 1.03)


class TestEquityAndDividendYield:
    """Tests for EquityAndDividendYield composition."""

    def test_exchange_rate_is_product(self):
        """Composed exchange rate is product of both."""
        a = _StubCurrency(exchange=1.2)
        b = _StubCurrency(exchange=1.05)
        comp = EquityAndDividendYield(a, b)
        q = comp.exchange_to_base({}, t=1.0)
        assert jnp.isclose(q, 1.2 * 1.05)

    def test_zcb_price_is_product(self):
        """Composed ZCB price is product of both."""
        a = _StubCurrency(zcb=0.88)
        b = _StubCurrency(zcb=0.97)
        comp = EquityAndDividendYield(a, b)
        p = comp.zcb_price({}, t=0.0, T=5.0)
        assert jnp.isclose(p, 0.88 * 0.97)

    def test_with_real_dividend_yield(self):
        """Composition works with real DividendYield and FactorWedge."""
        dy = DividendYield(dividend_yield=0.03, equity_key="eq")
        fw = FactorWedge(factor_spread=0.01, factor_key="fac")
        comp = EquityAndDividendYield(dy, fw)
        state = {"eq": {"level": 1.0}, "fac": {"level": 1.0}}
        # At t=0, both exchange rates should be 1.0
        q = comp.exchange_to_base(state, t=0.0)
        assert jnp.isclose(q, 1.0)


class TestWedgeCurrencyAndInflationWedge:
    """Tests for WedgeCurrencyAndInflationWedge composition."""

    def test_exchange_rate_is_product(self):
        """Composed exchange rate is product of both."""
        a = _StubCurrency(exchange=1.01)
        b = _StubCurrency(exchange=0.99)
        comp = WedgeCurrencyAndInflationWedge(a, b)
        q = comp.exchange_to_base({}, t=1.0)
        assert jnp.isclose(q, 1.01 * 0.99)

    def test_zcb_price_is_product(self):
        """Composed ZCB price is product of both."""
        a = _StubCurrency(zcb=0.98)
        b = _StubCurrency(zcb=0.99)
        comp = WedgeCurrencyAndInflationWedge(a, b)
        p = comp.zcb_price({}, t=0.0, T=5.0)
        assert jnp.isclose(p, 0.98 * 0.99)

    def test_spot_rate_from_composed_zcb(self):
        """Spot rate is derived from composed ZCB price."""
        a = _StubCurrency(zcb=0.98)
        b = _StubCurrency(zcb=0.99)
        comp = WedgeCurrencyAndInflationWedge(a, b)
        r = comp.spot_rate({}, t=0.0, T=5.0)
        expected = -jnp.log(0.98 * 0.99) / 5.0
        assert jnp.isclose(r, expected, atol=1e-10)
