"""Tests for DividendYield FCA type."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import pytest


class _MockEquityState(NamedTuple):
    level: float = 120.0


class TestDividendYield:
    """Tests for DividendYield pseudo-currency."""

    def test_exchange_to_base_dict_state(self):
        """Exchange rate = S(t)/S(0) from dict state."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        dy = DividendYield(
            dividend_yield=0.02,
            equity_key="eq",
            initial_equity_level=100.0,
        )
        state = {"eq": {"level": 120.0}}
        q = dy.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.2)

    def test_exchange_to_base_namedtuple_state(self):
        """Exchange rate reads level from NamedTuple."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        dy = DividendYield(
            dividend_yield=0.02,
            equity_key="eq",
            initial_equity_level=100.0,
        )
        state = {"eq": _MockEquityState(level=80.0)}
        q = dy.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 0.8)

    def test_exchange_to_base_default_level(self):
        """Exchange rate defaults to 1.0 if equity not in state."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        dy = DividendYield(initial_equity_level=1.0)
        state: dict = {}
        q = dy.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.0)

    def test_cash_account_exponential(self):
        """Cash account = exp(q * t)."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        q = 0.03
        dy = DividendYield(dividend_yield=q)
        state: dict = {}
        t = 5.0
        result = dy.cash_account(state, t)
        assert jnp.isclose(result, jnp.exp(q * t))

    def test_cash_account_zero_yield(self):
        """Cash account = 1.0 for zero yield at any time."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        dy = DividendYield(dividend_yield=0.0)
        state: dict = {}
        assert jnp.isclose(dy.cash_account(state, 0.0), 1.0)
        assert jnp.isclose(dy.cash_account(state, 10.0), 1.0)

    def test_zcb_price_constant_yield(self):
        """ZCB price = exp(-q * tau) for constant dividend yield."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        q = 0.025
        dy = DividendYield(dividend_yield=q)
        state: dict = {}
        t, T = 1.0, 6.0
        tau = T - t
        p = dy.zcb_price(state, t, T)
        assert jnp.isclose(p, jnp.exp(-q * tau), atol=1e-12)

    def test_spot_rate_equals_yield(self):
        """Spot rate equals the constant dividend yield."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        q = 0.04
        dy = DividendYield(dividend_yield=q)
        state: dict = {}
        r = dy.spot_rate(state, t=0.0, T=10.0)
        assert jnp.isclose(r, q, atol=1e-10)

    def test_zero_yield_zcb_is_one(self):
        """ZCB price is 1.0 for zero dividend yield."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        dy = DividendYield(dividend_yield=0.0)
        state: dict = {}
        p = dy.zcb_price(state, t=0.0, T=5.0)
        assert jnp.isclose(p, 1.0)

    def test_equity_key_property(self):
        """Equity key is accessible."""
        from hyesg.models.currencies.dividend_yield import DividendYield

        dy = DividendYield(equity_key="my_eq")
        assert dy.equity_key == "my_eq"
