"""FX exchange rate model via GBM with interest rate differential.

The exchange rate evolves as a log-normal process driven by the
differential between domestic and foreign short rates.

Supports optional SVJD composition: when a vol_model or jump_model
dependency name is configured, the model reads stochastic volatility
and jump contributions from those dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import FXState, ShockConfig
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.config.params import GBMParams


@register_model("fx")
class FXRate:
    """Exchange rate model via GBM with optional SVJD composition.

    SDE (plain): dS/S = (r_d - r_f)dt + σ dZ
    SDE (SVJD):  dS/S = (r_d - r_f - λκ - ½σ²(t))dt + σ(t)dW + J·dN

    When vol_model or jump_model are provided, reads stochastic
    volatility and jump outputs from deps. FX uses the same SVJD
    structure as equity via the FCA framework.

    Args:
        params: GBM parameters (sigma, initial_value).
        name: Unique model name.
        domestic_rate_model: Key for domestic rate model in deps.
        foreign_rate_model: Key for foreign rate model in deps.
        vol_model: Name of volatility dependency in deps.
        jump_model: Name of jump dependency in deps.
    """

    def __init__(
        self,
        params: GBMParams,
        name: str = "fx",
        domestic_rate_model: str = "",
        foreign_rate_model: str = "",
        vol_model: str = "",
        jump_model: str = "",
    ) -> None:
        self._params = params
        self._name = name
        self._domestic = domestic_rate_model
        self._foreign = foreign_rate_model
        self._vol_model = vol_model
        self._jump_model = jump_model

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

        When vol_model/jump_model are configured, reads stochastic
        volatility and jump contributions from deps. Otherwise
        uses constant sigma with no jumps (plain GBM).

        Args:
            state: Current FXState.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs keyed by model name, each a dict
                  containing ``"short_rate"``, and optionally vol/jump outputs.

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        dz = shocks[0]
        zero = jnp.array(0.0, dtype=jnp.float64)

        # Extract domestic and foreign short rates from deps
        r_d = (
            deps.get(self._domestic, {}).get(OutputName.SHORT_RATE, zero)
            if self._domestic
            else zero
        )
        r_f = (
            deps.get(self._foreign, {}).get(OutputName.SHORT_RATE, zero)
            if self._foreign
            else zero
        )

        # Volatility: stochastic (from deps) or constant
        if self._vol_model and self._vol_model in deps:
            sigma = deps[self._vol_model].get(
                OutputName.SIGMA, jnp.array(self._params.sigma, dtype=jnp.float64)
            )
        else:
            sigma = self._params.sigma

        # Jump contributions: from deps or zero
        if self._jump_model and self._jump_model in deps:
            jump = deps[self._jump_model].get(OutputName.JUMP, zero)
            drift_adj = deps[self._jump_model].get(OutputName.DRIFT_ADJUSTMENT, zero)
        else:
            jump = zero
            drift_adj = zero

        log_new = (
            state.log_level
            + (r_d - r_f - 0.5 * sigma**2) * dt
            + drift_adj
            + sigma * dz * jnp.sqrt(dt)
            + jump
        )
        level = jnp.exp(log_new)

        new_state = FXState(log_level=log_new, level=level)
        return new_state, {OutputName.EXCHANGE_RATE: level}
