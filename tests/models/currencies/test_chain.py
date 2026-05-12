"""Tests for ExchangeRateAnalogyChain."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hyesg.models.currencies.chain import ExchangeRateAnalogyChain


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


class TestExchangeRateAnalogyChain:
    """Tests for ExchangeRateAnalogyChain."""

    def test_empty_chain_raises(self):
        """Empty chain raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            ExchangeRateAnalogyChain([])

    def test_single_currency_chain(self):
        """Single-currency chain returns that currency's values."""
        c = _StubCurrency(cash=1.2, exchange=1.5, zcb=0.9)
        chain = ExchangeRateAnalogyChain([c])
        assert jnp.isclose(chain.exchange_to_base({}, t=1.0), 1.5)
        assert jnp.isclose(chain.cash_account({}, t=1.0), 1.2)
        assert jnp.isclose(chain.zcb_price({}, t=0.0, T=5.0), 0.9)

    def test_two_currency_chain_exchange_rate(self):
        """Two-currency chain multiplies exchange rates."""
        a = _StubCurrency(exchange=1.35)
        b = _StubCurrency(exchange=0.85)
        chain = ExchangeRateAnalogyChain([a, b])
        q = chain.exchange_to_base({}, t=1.0)
        assert jnp.isclose(q, 1.35 * 0.85)

    def test_two_currency_chain_zcb(self):
        """Two-currency chain multiplies ZCB prices."""
        a = _StubCurrency(zcb=0.9)
        b = _StubCurrency(zcb=0.95)
        chain = ExchangeRateAnalogyChain([a, b])
        p = chain.zcb_price({}, t=0.0, T=5.0)
        assert jnp.isclose(p, 0.9 * 0.95)

    def test_three_currency_chain(self):
        """Three-currency chain multiplies all exchange rates."""
        a = _StubCurrency(exchange=1.3)
        b = _StubCurrency(exchange=0.9)
        c = _StubCurrency(exchange=1.1)
        chain = ExchangeRateAnalogyChain([a, b, c])
        q = chain.exchange_to_base({}, t=1.0)
        assert jnp.isclose(q, 1.3 * 0.9 * 1.1)

    def test_spot_rate_from_chained_zcb(self):
        """Spot rate is derived from chained ZCB price."""
        a = _StubCurrency(zcb=0.85)
        b = _StubCurrency(zcb=0.92)
        chain = ExchangeRateAnalogyChain([a, b])
        r = chain.spot_rate({}, t=0.0, T=5.0)
        expected = -jnp.log(0.85 * 0.92) / 5.0
        assert jnp.isclose(r, expected, atol=1e-10)

    def test_currencies_property(self):
        """Currencies property returns the list."""
        a = _StubCurrency()
        b = _StubCurrency()
        chain = ExchangeRateAnalogyChain([a, b])
        assert len(chain.currencies) == 2

    def test_cash_account_chain(self):
        """Cash accounts are multiplied across the chain."""
        a = _StubCurrency(cash=1.1)
        b = _StubCurrency(cash=1.2)
        c = _StubCurrency(cash=1.05)
        chain = ExchangeRateAnalogyChain([a, b, c])
        result = chain.cash_account({}, t=1.0)
        assert jnp.isclose(result, 1.1 * 1.2 * 1.05)
