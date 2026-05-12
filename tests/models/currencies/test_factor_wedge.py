"""Tests for FactorWedge FCA type."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import pytest


class _MockFactorState(NamedTuple):
    level: float = 1.05


class TestFactorWedge:
    """Tests for FactorWedge pseudo-currency."""

    def test_exchange_to_base_dict_state(self):
        """Exchange rate = factor_level / initial from dict state."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        fw = FactorWedge(
            factor_spread=0.01,
            factor_key="size",
            initial_factor_level=100.0,
        )
        state = {"size": {"TotalReturnIndex": 110.0}}
        q = fw.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.1)

    def test_exchange_to_base_namedtuple_state(self):
        """Exchange rate reads level from NamedTuple."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        fw = FactorWedge(
            factor_key="val",
            initial_factor_level=1.0,
        )
        state = {"val": _MockFactorState(level=1.05)}
        q = fw.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.05)

    def test_exchange_to_base_default(self):
        """Exchange rate defaults to 1.0 if not in state."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        fw = FactorWedge(initial_factor_level=1.0)
        state: dict = {}
        q = fw.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.0)

    def test_cash_account_exponential(self):
        """Cash account = exp(s * t) for constant spread s."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        s = 0.02
        fw = FactorWedge(factor_spread=s)
        state: dict = {}
        t = 3.0
        result = fw.cash_account(state, t)
        assert jnp.isclose(result, jnp.exp(s * t))

    def test_cash_account_zero_spread(self):
        """Cash account = 1.0 for zero spread."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        fw = FactorWedge(factor_spread=0.0)
        state: dict = {}
        assert jnp.isclose(fw.cash_account(state, 5.0), 1.0)

    def test_zcb_price_constant_spread(self):
        """ZCB price = exp(-s * tau)."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        s = 0.015
        fw = FactorWedge(factor_spread=s)
        state: dict = {}
        t, T = 2.0, 7.0
        tau = T - t
        p = fw.zcb_price(state, t, T)
        assert jnp.isclose(p, jnp.exp(-s * tau), atol=1e-12)

    def test_spot_rate_equals_spread(self):
        """Spot rate equals the constant factor spread."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        s = 0.03
        fw = FactorWedge(factor_spread=s)
        state: dict = {}
        r = fw.spot_rate(state, t=0.0, T=5.0)
        assert jnp.isclose(r, s, atol=1e-10)

    def test_zero_spread_zcb_is_one(self):
        """ZCB price is 1.0 for zero spread."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        fw = FactorWedge(factor_spread=0.0)
        state: dict = {}
        p = fw.zcb_price(state, t=0.0, T=10.0)
        assert jnp.isclose(p, 1.0)

    def test_factor_key_property(self):
        """Factor key is accessible."""
        from hyesg.models.currencies.factor_wedge import FactorWedge

        fw = FactorWedge(factor_key="lowvol")
        assert fw.factor_key == "lowvol"
