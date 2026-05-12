"""Tests for BaseNominal FCA type."""

from __future__ import annotations

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


class TestBaseNominal:
    """Tests for BaseNominal pseudo-currency."""

    def test_exchange_to_base_is_one(self):
        """Base currency exchange rate is always 1.0."""
        from hyesg.models.currencies.base_nominal import BaseNominal

        model = _MockShortRateModel()
        bn = BaseNominal(model)
        state: dict = {"nominal": {"cash_account": 1.5}}
        q = bn.exchange_to_base(state, t=1.0)
        assert jnp.isclose(q, 1.0)

    def test_exchange_to_base_is_one_at_various_times(self):
        """Q(t) = 1 regardless of time."""
        from hyesg.models.currencies.base_nominal import BaseNominal

        model = _MockShortRateModel()
        bn = BaseNominal(model)
        state: dict = {}
        for t in [0.0, 0.5, 1.0, 5.0, 10.0]:
            assert jnp.isclose(bn.exchange_to_base(state, t), 1.0)

    def test_cash_account_returns_stored_value(self):
        """Cash account reads from state dict."""
        from hyesg.models.currencies.base_nominal import BaseNominal

        model = _MockShortRateModel()
        bn = BaseNominal(model, model_key="nom")
        state = {"nom": {"cash_account": 1.25}}
        result = bn.cash_account(state, t=1.0)
        assert jnp.isclose(result, 1.25)

    def test_cash_account_default_when_missing(self):
        """Cash account defaults to 1.0 if not in state."""
        from hyesg.models.currencies.base_nominal import BaseNominal

        model = _MockShortRateModel()
        bn = BaseNominal(model)
        state: dict = {}
        result = bn.cash_account(state, t=1.0)
        assert jnp.isclose(result, 1.0)

    def test_zcb_price_delegates_to_model(self):
        """ZCB price delegates to the underlying short rate model."""
        from hyesg.models.currencies.base_nominal import BaseNominal

        rate = 0.04
        model = _MockShortRateModel(rate=rate)
        bn = BaseNominal(model)
        state = {"nominal": None}
        t, T = 0.0, 5.0
        p = bn.zcb_price(state, t, T)
        expected = jnp.exp(-rate * 5.0)
        assert jnp.isclose(p, expected, atol=1e-12)

    def test_spot_rate_consistent_with_zcb(self):
        """Spot rate = -ln(P)/(T-t) matches the model rate."""
        from hyesg.models.currencies.base_nominal import BaseNominal

        rate = 0.03
        model = _MockShortRateModel(rate=rate)
        bn = BaseNominal(model)
        state = {"nominal": None}
        t, T = 1.0, 6.0
        r = bn.spot_rate(state, t, T)
        assert jnp.isclose(r, rate, atol=1e-10)

    def test_model_key_property(self):
        """Model key is accessible."""
        from hyesg.models.currencies.base_nominal import BaseNominal

        model = _MockShortRateModel()
        bn = BaseNominal(model, model_key="my_nom")
        assert bn.model_key == "my_nom"
