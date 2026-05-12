"""StandardNominal — foreign economy pseudo-currency.

A foreign economy (e.g. USD, EUR) has its own short rate model and
an FX rate against the domestic currency.  The exchange rate Q(t) is
the FX spot rate, and the cash account accumulates at the foreign
nominal short rate.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.core.protocols import ShortRateModel


class StandardNominal:
    """FCA foreign nominal currency.

    Wraps a foreign short rate model and an FX model to present the
    unified ``CurrencyAnalogy`` interface.

    Args:
        short_rate_model: Foreign economy short rate model.
        rate_model_key: Key for the short rate model in the state dict.
        fx_model_key: Key for the FX model in the state dict.
    """

    def __init__(
        self,
        short_rate_model: ShortRateModel,
        rate_model_key: str = "foreign_nominal",
        fx_model_key: str = "fx",
    ) -> None:
        self._model = short_rate_model
        self._rate_key = rate_model_key
        self._fx_key = fx_model_key

    @property
    def rate_model_key(self) -> str:
        """Key for the foreign short rate model in the state dict."""
        return self._rate_key

    @property
    def fx_model_key(self) -> str:
        """Key for the FX model in the state dict."""
        return self._fx_key

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Cash account accumulated at the foreign nominal short rate.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Foreign cash account value.
        """
        model_state = state.get(self._rate_key, {})
        if isinstance(model_state, dict):
            return jnp.asarray(
                model_state.get("cash_account", 1.0), dtype=jnp.float64
            )
        return jnp.array(1.0, dtype=jnp.float64)

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Exchange rate Q(t) = FX spot rate.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            FX spot rate from foreign to domestic currency.
        """
        fx_state = state.get(self._fx_key, {})
        if isinstance(fx_state, dict):
            return jnp.asarray(fx_state.get("level", 1.0), dtype=jnp.float64)
        # NamedTuple state — access .level attribute
        return jnp.asarray(getattr(fx_state, "level", 1.0), dtype=jnp.float64)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Zero-coupon bond price in the foreign nominal currency.

        Delegates to the foreign short rate model's ``zcb_price``.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Foreign ZCB price P_f(t, T).
        """
        model_state = state.get(self._rate_key)
        return self._model.zcb_price(model_state, t, T)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Spot rate: -ln(P_f(t,T)) / (T-t).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Foreign continuously compounded spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)
