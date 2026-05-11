"""GBM equity / property model.

Geometric Brownian Motion for equity indices and property values.
The model evolves log-prices with drift from the domestic short rate
minus a continuous dividend yield.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import FXState, ShockConfig

if TYPE_CHECKING:
    from hyesg.config.params import GBMParams


@register_model("equity")
class Equity:
    """GBM equity / property model.

    SDE: dS/S = (r - q - σ²/2)dt + σ dZ   (log-normal)
    Euler in log: ln S_{t+dt} = ln S_t + (r - q - σ²/2)dt + σ·dZ·√dt

    Where r = domestic short rate (from deps), q = dividend yield,
    σ can be constant or from a stochastic vol model (from deps).

    Shocks: 1 × N(0,1), model multiplies by √dt.

    Args:
        params: GBM parameters (sigma, initial_value).
        name: Unique model name.
        dividend_yield: Continuous dividend yield.
    """

    def __init__(
        self,
        params: GBMParams,
        name: str = "equity",
        dividend_yield: float = 0.0,
    ) -> None:
        self._params = params
        self._name = name
        self._dividend_yield = dividend_yield

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

        Args:
            state: Current FXState.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs (expects ``"short_rate"`` key).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        dz = shocks[0]

        # Drift from domestic short rate dependency (nested deps format)
        r = jnp.array(0.0, dtype=jnp.float64)
        for dep_out in deps.values():
            if isinstance(dep_out, dict) and "short_rate" in dep_out:
                r = dep_out["short_rate"]
                break
        q = jnp.array(self._dividend_yield, dtype=jnp.float64)
        sigma = self._params.sigma

        # Log-normal Euler step
        log_new = (
            state.log_level + (r - q - 0.5 * sigma**2) * dt + sigma * dz * jnp.sqrt(dt)
        )
        level = jnp.exp(log_new)

        new_state = FXState(log_level=log_new, level=level)
        outputs = {
            "level": level,
            "log_return": log_new - state.log_level,
        }
        return new_state, outputs
