"""Tests for the CurrencyAnalogy protocol compliance."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hyesg.models.currencies.base_nominal import BaseNominal
from hyesg.models.currencies.chain import ExchangeRateAnalogyChain
from hyesg.models.currencies.compositions import (
    EquityAndDividendYield,
    NominalAndExchangeRate,
    RealRateAndInflation,
    WedgeCurrencyAndInflationWedge,
)
from hyesg.models.currencies.dividend_yield import DividendYield
from hyesg.models.currencies.factor_wedge import FactorWedge
from hyesg.models.currencies.protocols import CurrencyAnalogy
from hyesg.models.currencies.real_rate import RealRate
from hyesg.models.currencies.standard_nominal import StandardNominal


class _MockShortRateModel:
    """Minimal mock that satisfies ShortRateModel for zcb_price."""

    def __init__(self, rate: float = 0.05) -> None:
        self._rate = rate

    def zcb_price(self, state, t, T):
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        return jnp.exp(-jnp.asarray(self._rate, dtype=jnp.float64) * tau)

    def short_rate(self, state):
        return jnp.asarray(self._rate, dtype=jnp.float64)

    def spot_rate(self, state, t, T):
        return jnp.asarray(self._rate, dtype=jnp.float64)

    def forward_rate(self, state, t, T):
        return jnp.asarray(self._rate, dtype=jnp.float64)

    def swap_rate(self, state, t, tenors):
        return jnp.asarray(self._rate, dtype=jnp.float64)


def test_base_nominal_is_currency_analogy():
    """BaseNominal satisfies the CurrencyAnalogy protocol."""
    model = _MockShortRateModel()
    bn = BaseNominal(model)
    assert isinstance(bn, CurrencyAnalogy)


def test_standard_nominal_is_currency_analogy():
    """StandardNominal satisfies the CurrencyAnalogy protocol."""
    model = _MockShortRateModel()
    sn = StandardNominal(model)
    assert isinstance(sn, CurrencyAnalogy)


def test_real_rate_is_currency_analogy():
    """RealRate satisfies the CurrencyAnalogy protocol."""
    model = _MockShortRateModel()
    rr = RealRate(model)
    assert isinstance(rr, CurrencyAnalogy)


def test_dividend_yield_is_currency_analogy():
    """DividendYield satisfies the CurrencyAnalogy protocol."""
    dy = DividendYield(dividend_yield=0.02)
    assert isinstance(dy, CurrencyAnalogy)


def test_factor_wedge_is_currency_analogy():
    """FactorWedge satisfies the CurrencyAnalogy protocol."""
    fw = FactorWedge(factor_spread=0.01)
    assert isinstance(fw, CurrencyAnalogy)


def test_composition_nominal_fx_is_currency_analogy():
    """NominalAndExchangeRate satisfies the CurrencyAnalogy protocol."""
    model = _MockShortRateModel()
    bn = BaseNominal(model)
    dy = DividendYield()
    comp = NominalAndExchangeRate(bn, dy)
    assert isinstance(comp, CurrencyAnalogy)


def test_composition_real_inflation_is_currency_analogy():
    """RealRateAndInflation satisfies the CurrencyAnalogy protocol."""
    model = _MockShortRateModel()
    rr = RealRate(model)
    dy = DividendYield()
    comp = RealRateAndInflation(rr, dy)
    assert isinstance(comp, CurrencyAnalogy)


def test_composition_equity_dividend_is_currency_analogy():
    """EquityAndDividendYield satisfies the CurrencyAnalogy protocol."""
    dy = DividendYield(dividend_yield=0.02)
    fw = FactorWedge()
    comp = EquityAndDividendYield(dy, fw)
    assert isinstance(comp, CurrencyAnalogy)


def test_composition_wedge_is_currency_analogy():
    """WedgeCurrencyAndInflationWedge satisfies the CurrencyAnalogy protocol."""
    fw1 = FactorWedge(factor_spread=0.01)
    fw2 = FactorWedge(factor_spread=0.005)
    comp = WedgeCurrencyAndInflationWedge(fw1, fw2)
    assert isinstance(comp, CurrencyAnalogy)


def test_chain_is_currency_analogy():
    """ExchangeRateAnalogyChain satisfies the CurrencyAnalogy protocol."""
    dy = DividendYield()
    chain = ExchangeRateAnalogyChain([dy])
    assert isinstance(chain, CurrencyAnalogy)
