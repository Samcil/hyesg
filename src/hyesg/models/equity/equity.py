"""GBM equity / property model.

Geometric Brownian Motion for equity indices and property values.
The model evolves log-prices with drift from the domestic short rate
minus a continuous dividend yield.

Supports optional SVJD composition: when a vol_model or jump_model
dependency name is configured, the model reads stochastic volatility
and jump contributions from those dependencies instead of using
constant sigma.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import FXState, ShockConfig
from hyesg.models.equity._helpers import extract_short_rate
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.config.params import GBMParams


@register_model("equity")
class Equity:
    """GBM equity / property model with optional SVJD composition.

    SDE (plain): dS/S = (r - q - σ²/2)dt + σ dZ   (log-normal)
    SDE (SVJD):  dS/S = (r - q - λκ - ½σ²(t))dt + σ(t)dW + J·dN

    When vol_model or jump_model are provided, the model reads
    stochastic volatility and jump outputs from deps, composing
    the full SVJD model. When not provided, falls back to plain
    GBM with constant sigma.

    Args:
        params: GBM parameters (sigma, initial_value).
        name: Unique model name.
        dividend_yield: Continuous dividend yield.
        vol_model: Name of the volatility dependency in deps.
            When set, reads ``deps[vol_model]["volatility"]``
            as σ(t) instead of using constant sigma.
        jump_model: Name of the jump dependency in deps.
            When set, reads ``deps[jump_model]["jump"]`` and
            ``deps[jump_model]["drift_adjustment"]``.
    """

    def __init__(
        self,
        params: GBMParams,
        name: str = "equity",
        dividend_yield: float = 0.0,
        vol_model: str = "",
        jump_model: str = "",
    ) -> None:
        self._params = params
        self._name = name
        self._dividend_yield = dividend_yield
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
        """Advance the equity price by one timestep.

        When vol_model/jump_model are configured, reads stochastic
        volatility and jump contributions from deps. Otherwise
        uses constant sigma with no jumps (plain GBM).

        Args:
            state: Current FXState.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs (expects ``"short_rate"`` key,
                  and optionally vol/jump model outputs).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        dz = shocks[0]

        # Drift from domestic short rate dependency (nested deps format)
        r = extract_short_rate(deps)
        q = jnp.array(self._dividend_yield, dtype=jnp.float64)

        # Volatility: stochastic (from deps) or constant
        if self._vol_model and self._vol_model in deps:
            sigma = deps[self._vol_model].get(
                OutputName.SIGMA, jnp.array(self._params.sigma, dtype=jnp.float64)
            )
        else:
            sigma = self._params.sigma

        # Jump contributions: from deps or zero
        zero = jnp.array(0.0, dtype=jnp.float64)
        if self._jump_model and self._jump_model in deps:
            jump = deps[self._jump_model].get(OutputName.JUMP, zero)
            drift_adj = deps[self._jump_model].get(OutputName.DRIFT_ADJUSTMENT, zero)
        else:
            jump = zero
            drift_adj = zero

        # Log-normal Euler step with optional SVJD terms
        log_new = (
            state.log_level
            + (r - q - 0.5 * sigma**2) * dt
            + drift_adj
            + sigma * dz * jnp.sqrt(dt)
            + jump
        )
        level = jnp.exp(log_new)

        new_state = FXState(log_level=log_new, level=level)
        outputs: dict[str, Any] = {
            OutputName.TOTAL_RETURN_INDEX: level,
            OutputName.LOG_RETURN: log_new - state.log_level,
        }
        return new_state, outputs
