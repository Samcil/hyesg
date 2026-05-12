"""Tests for StandardNominal FCA type."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import pytest


class _MockShortRateModel:
    """Mock short rate model with constant rate."""

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


class _MockFXState(NamedTuple):
    level: float = 1.5


class TestStandardNominal:
    """Tests for StandardNominal pseudo-currency."""

    def test_exchange_to_base_dict_state(self):
        """Exchange rate reads FX level from dict state."""
        from hyesg.models.currencies.standard_nominal import StandardNominal

        model = _MockShortRateModel()
        sn = StandardNominal(model, fx_model_key="fx")
        state = {"fx": {"ExchangeRate": 1.35}, "foreign_nominal": None}
        q = sn.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.35)

    def test_exchange_to_base_namedtuple_state(self):
        """Exchange rate reads FX level from NamedTuple state."""
        from hyesg.models.currencies.standard_nominal import StandardNominal

        model = _MockShortRateModel()
        sn = StandardNominal(model, fx_model_key="fx")
        state = {"fx": _MockFXState(level=0.85), "foreign_nominal": None}
        q = sn.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 0.85)

    def test_exchange_to_base_default(self):
        """Exchange rate defaults to 1.0 if FX not in state."""
        from hyesg.models.currencies.standard_nominal import StandardNominal

        model = _MockShortRateModel()
        sn = StandardNominal(model, fx_model_key="fx")
        state: dict = {}
        q = sn.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.0)

    def test_cash_account_from_state(self):
        """Cash account reads from state."""
        from hyesg.models.currencies.standard_nominal import StandardNominal

        model = _MockShortRateModel()
        sn = StandardNominal(model, rate_model_key="usd")
        state = {"usd": {"cash_account": 1.1}}
        result = sn.cash_account(state, t=1.0)
        assert jnp.isclose(result, 1.1)

    def test_zcb_price_delegates(self):
        """ZCB price delegates to the foreign short rate model."""
        from hyesg.models.currencies.standard_nominal import StandardNominal

        rate = 0.06
        model = _MockShortRateModel(rate=rate)
        sn = StandardNominal(model, rate_model_key="usd")
        state = {"usd": None}
        p = sn.zcb_price(state, t=0.0, T=10.0)
        expected = jnp.exp(-rate * 10.0)
        assert jnp.isclose(p, expected, atol=1e-12)

    def test_spot_rate_consistent(self):
        """Spot rate is consistent with ZCB price."""
        from hyesg.models.currencies.standard_nominal import StandardNominal

        rate = 0.04
        model = _MockShortRateModel(rate=rate)
        sn = StandardNominal(model, rate_model_key="usd")
        state = {"usd": None}
        r = sn.spot_rate(state, t=2.0, T=7.0)
        assert jnp.isclose(r, rate, atol=1e-10)

    def test_key_properties(self):
        """Rate and FX model keys are accessible."""
        from hyesg.models.currencies.standard_nominal import StandardNominal

        model = _MockShortRateModel()
        sn = StandardNominal(model, rate_model_key="usd", fx_model_key="gbpusd")
        assert sn.rate_model_key == "usd"
        assert sn.fx_model_key == "gbpusd"
