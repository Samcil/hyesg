"""BaseNominal — domestic economy pseudo-currency.

The domestic economy (e.g. GBP) is the base currency in the FCA
framework.  Its exchange rate Q(t) is always 1.  The cash account
accumulates at the nominal short rate, and ZCB pricing delegates
to the underlying short rate model (e.g. CIR2++).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.core.protocols import ShortRateModel


class BaseNominal:
    """FCA base (domestic) nominal currency.

    Wraps a ``ShortRateModel`` and presents the unified
    ``CurrencyAnalogy`` interface with Q(t) = 1.

    Args:
        short_rate_model: The underlying nominal short rate model
            (e.g. CIR2++).
        model_key: Key used to look up the model's state in the
            simulation state dict.
    """

    def __init__(
        self,
        short_rate_model: ShortRateModel,
        model_key: str = "nominal",
    ) -> None:
        self._model = short_rate_model
        self._model_key = model_key

    @property
    def model_key(self) -> str:
        """Key for the underlying model in the state dict."""
        return self._model_key

    def cash_account(self, state: dict[str, Any], t: float) -> Array:
        """Cash account accumulated at the nominal short rate.

        Returns the cash account value stored in the simulation state.
        If not yet accumulated, returns 1.0 (initial value).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.

        Returns:
            Nominal cash account value.
        """
        model_state = state.get(self._model_key, {})
        if isinstance(model_state, dict):
            return jnp.asarray(
                model_state.get("cash_account", 1.0), dtype=jnp.float64
            )
        return jnp.array(1.0, dtype=jnp.float64)

    def exchange_to_base(self, state: dict[str, Any], t: float) -> Array:
        """Exchange rate Q(t) = 1 always for the base currency.

        Args:
            state: Simulation state dict (unused).
            t: Current time in years (unused).

        Returns:
            Scalar 1.0.
        """
        return jnp.array(1.0, dtype=jnp.float64)

    def zcb_price(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Zero-coupon bond price in the domestic nominal currency.

        Delegates to the underlying short rate model's ``zcb_price``.

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            ZCB price P(t, T).
        """
        model_state = state.get(self._model_key)
        return self._model.zcb_price(model_state, t, T)

    def spot_rate(self, state: dict[str, Any], t: float, T: float) -> Array:
        """Spot rate: -ln(P(t,T)) / (T-t).

        Args:
            state: Simulation state dict keyed by model name.
            t: Current time in years.
            T: Maturity time in years.

        Returns:
            Continuously compounded spot rate.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, T)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)
