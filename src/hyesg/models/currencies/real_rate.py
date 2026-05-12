"""RealRate — pseudo-currency for real interest rates.

In the FCA framework, the real rate is treated as a pseudo-currency
where:
    - Short rate = real short rate (from G2++ model)
    - Exchange rate Q(t) = price index (inflation index)
    - Fisher relation: real_rate ≈ nominal_rate - inflation_rate
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.core.protocols import ShortRateModel


class RealRate:
    """FCA real rate pseudo-currency.

    Wraps a real short rate model (e.g. G2++) and an inflation index
    to present the unified ``CurrencyAnalogy`` interface.

    Args:
        short_rate_model: Real short rate model (e.g. G2++).
        rate_model_key: Key for the real rate model in the state dict.
        inflation_key: Key for the inflation index in the state dict.
    """

    def __init__(
        self,
        short_rate_model: ShortRateModel,
        rate_model_key: str = "real_rate",
        inflation_key: str = "inflation",
    ) -> None:
        self._model = short_rate_model
        self._rate_key = rate_model_key
        self._inflation_key = inflation_key

    @property
    def rate_model_key(self) -> str:
        """Key for the real rate model in the state dict."""
        return self._rate_key

    @property
    def inflation_key(self) -> str:
        """Key for the inflation index in the state dict."""
        return self._inflation_key

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Cash account accumulated at the real short rate.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Real cash account value.
        """
        model_state = state.get(self._rate_key, {})
        if isinstance(model_state, dict):
            return jnp.asarray(
                model_state.get("cash_account", 1.0), dtype=jnp.float64
            )
        return jnp.array(1.0, dtype=jnp.float64)

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Exchange rate Q(t) = inflation price index.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Inflation price index level.
        """
        infl_state = state.get(self._inflation_key, {})
        if isinstance(infl_state, dict):
            return jnp.asarray(
                infl_state.get("index", 1.0), dtype=jnp.float64
            )
        return jnp.asarray(
            getattr(infl_state, "level", 1.0), dtype=jnp.float64
        )

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Zero-coupon bond price in the real rate currency.

        Delegates to the real short rate model's ``zcb_price``.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Real ZCB price.
        """
        model_state = state.get(self._rate_key)
        return self._model.zcb_price(model_state, t, T)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Spot rate: -ln(P_real(t,T)) / (T-t).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Real continuously compounded spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)
