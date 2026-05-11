"""FX exchange rate model via GBM with interest rate differential.

The exchange rate evolves as a log-normal process driven by the
differential between domestic and foreign short rates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import FXState, ShockConfig

if TYPE_CHECKING:
    from hyesg.config.params import GBMParams


@register_model("fx")
class FXRate:
    """Exchange rate model via GBM with interest rate differential.

    SDE: dS/S = (r_domestic - r_foreign)dt + σ dZ
    Euler in log: ln S_{t+dt} = ln S_t + (r_d - r_f - σ²/2)dt + σ·dZ·√dt

    Dependencies: needs domestic and foreign short rates from deps.

    Args:
        params: GBM parameters (sigma, initial_value).
        name: Unique model name.
        domestic_rate_model: Key for domestic rate model in deps.
        foreign_rate_model: Key for foreign rate model in deps.
    """

    def __init__(
        self,
        params: GBMParams,
        name: str = "fx",
        domestic_rate_model: str = "",
        foreign_rate_model: str = "",
    ) -> None:
        self._params = params
        self._name = name
        self._domestic = domestic_rate_model
        self._foreign = foreign_rate_model

    @property
    def name(self) -> str:
        """Unique model name."""
        return self._name

    @property
    def n_shocks(self) -> int:
        """Number of independent Brownian increments."""
        return 1

    @property
    def shock_config(self) -> ShockConfig:
        """Shock metadata for the correlation engine."""
        return ShockConfig(
            n_shocks=1,
            distribution="normal",
            correlate=True,
            names=(f"{self._name}_z",),
        )

    def init_state(self, params: Any = None, market: Any = None) -> FXState:
        """Create initial state from parameters.

        Args:
            params: Optional override parameters.
            market: Optional market data.

        Returns:
            Initial FXState with log_level and level.
        """
        s0 = jnp.array(self._params.initial_value, dtype=jnp.float64)
        return FXState(log_level=jnp.log(s0), level=s0)

    def step(
        self,
        state: FXState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[FXState, dict[str, Any]]:
        """Advance the FX rate by one timestep.

        Args:
            state: Current FXState.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs keyed by model name, each a dict
                  containing ``"short_rate"``.

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        dz = shocks[0]
        zero = jnp.array(0.0, dtype=jnp.float64)

        # Extract domestic and foreign short rates from deps
        r_d = (
            deps.get(self._domestic, {}).get("short_rate", zero)
            if self._domestic
            else zero
        )
        r_f = (
            deps.get(self._foreign, {}).get("short_rate", zero)
            if self._foreign
            else zero
        )

        sigma = self._params.sigma
        log_new = (
            state.log_level
            + (r_d - r_f - 0.5 * sigma**2) * dt
            + sigma * dz * jnp.sqrt(dt)
        )
        level = jnp.exp(log_new)

        new_state = FXState(log_level=log_new, level=level)
        return new_state, {"level": level, "rate": level}
