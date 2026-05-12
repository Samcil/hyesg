"""Tests for RealRate FCA type."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import pytest


class _MockShortRateModel:
    """Mock short rate model with constant rate."""

    def __init__(self, rate: float = 0.02) -> None:
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


class _MockInflationState(NamedTuple):
    level: float = 1.1


class TestRealRate:
    """Tests for RealRate pseudo-currency."""

    def test_exchange_to_base_dict_state(self):
        """Exchange rate reads inflation index from dict state."""
        from hyesg.models.currencies.real_rate import RealRate

        model = _MockShortRateModel()
        rr = RealRate(model, inflation_key="cpi")
        state = {"cpi": {"InflationIndex": 1.15}, "real_rate": None}
        q = rr.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.15)

    def test_exchange_to_base_namedtuple_state(self):
        """Exchange rate reads level from NamedTuple inflation state."""
        from hyesg.models.currencies.real_rate import RealRate

        model = _MockShortRateModel()
        rr = RealRate(model, inflation_key="infl")
        state = {"infl": _MockInflationState(level=1.2), "real_rate": None}
        q = rr.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.2)

    def test_exchange_to_base_default(self):
        """Exchange rate defaults to 1.0 if not in state."""
        from hyesg.models.currencies.real_rate import RealRate

        model = _MockShortRateModel()
        rr = RealRate(model)
        state: dict = {}
        q = rr.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.0)

    def test_cash_account_from_state(self):
        """Cash account reads from state."""
        from hyesg.models.currencies.real_rate import RealRate

        model = _MockShortRateModel()
        rr = RealRate(model, rate_model_key="real")
        state = {"real": {"cash_account": 1.05}}
        result = rr.cash_account(state, t=1.0)
        assert jnp.isclose(result, 1.05)

    def test_cash_account_default(self):
        """Cash account defaults to 1.0."""
        from hyesg.models.currencies.real_rate import RealRate

        model = _MockShortRateModel()
        rr = RealRate(model)
        state: dict = {}
        result = rr.cash_account(state, t=1.0)
        assert jnp.isclose(result, 1.0)

    def test_zcb_price_delegates(self):
        """ZCB price delegates to the real short rate model."""
        from hyesg.models.currencies.real_rate import RealRate

        rate = 0.015
        model = _MockShortRateModel(rate=rate)
        rr = RealRate(model, rate_model_key="real")
        state = {"real": None}
        p = rr.zcb_price(state, t=0.0, T=5.0)
        expected = jnp.exp(-rate * 5.0)
        assert jnp.isclose(p, expected, atol=1e-12)

    def test_spot_rate_consistent(self):
        """Spot rate is consistent with ZCB price."""
        from hyesg.models.currencies.real_rate import RealRate

        rate = 0.02
        model = _MockShortRateModel(rate=rate)
        rr = RealRate(model, rate_model_key="real")
        state = {"real": None}
        r = rr.spot_rate(state, t=1.0, T=11.0)
        assert jnp.isclose(r, rate, atol=1e-10)

    def test_key_properties(self):
        """Rate and inflation keys are accessible."""
        from hyesg.models.currencies.real_rate import RealRate

        model = _MockShortRateModel()
        rr = RealRate(model, rate_model_key="real", inflation_key="cpi")
        assert rr.rate_model_key == "real"
        assert rr.inflation_key == "cpi"
