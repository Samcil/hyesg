"""Inflation index model with Fourier seasonality.

Tracks an inflation index (e.g. RPI, CPI) using a GBM-like process
driven by a real rate dependency, with an optional Fourier seasonal
adjustment subtracted from the forward rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import FXState, ShockConfig
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.config.params import GBMParams, SeasonalityParams


@register_model("inflation")
class Inflation:
    """Inflation index model with Fourier seasonality.

    The index evolves as:
        I(t+dt) = I(t) * exp((r_real - seasonal - σ²/2) * dt + σ * dZ * √dt)

    where r_real comes from a real rate model dependency.

    Seasonality is a Fourier series (2 harmonics, 4 coefficients):
        seasonal(shift) = 0.01 * (a1*cos(2π*s) + a2*cos(4π*s)
                                 + b1*sin(2π*s) + b2*sin(4π*s))

    where s = t + 0.5.  Seasonality is SUBTRACTED from the forward rate.

    Args:
        params: GBM parameters (sigma, initial_value).
        name: Unique model name.
        real_rate_model: Key for real rate model in deps.
        seasonality_params: Optional Fourier seasonality coefficients.
    """

    def __init__(
        self,
        params: GBMParams,
        name: str = "inflation",
        real_rate_model: str = "",
        seasonality_params: SeasonalityParams | None = None,
    ) -> None:
        self._params = params
        self._name = name
        self._real_rate = real_rate_model
        self._seasonality = seasonality_params

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

    def seasonal_adjustment(self, t: float) -> Any:
        """Fourier seasonality adjustment at time t.

        Computes the 2-harmonic Fourier series scaled by 0.01:
            0.01 * (a1*cos(2π*s) + a2*cos(4π*s)
                  + b1*sin(2π*s) + b2*sin(4π*s))

        where s = t + 0.5.

        Args:
            t: Time in years.

        Returns:
            Seasonal adjustment (scalar JAX array).
        """
        if self._seasonality is None:
            return jnp.array(0.0, dtype=jnp.float64)

        s = self._seasonality
        shift = t + 0.5
        two_pi = 2.0 * jnp.pi
        result = (
            s.a1 * jnp.cos(two_pi * shift)
            + s.a2 * jnp.cos(2.0 * two_pi * shift)
            + s.b1 * jnp.sin(two_pi * shift)
            + s.b2 * jnp.sin(2.0 * two_pi * shift)
        )
        return 0.01 * result

    def step(
        self,
        state: FXState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[FXState, dict[str, Any]]:
        """Advance the inflation index by one timestep.

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

        # Real rate from dependency
        r_real = (
            deps.get(self._real_rate, {}).get(OutputName.SHORT_RATE, zero)
            if self._real_rate
            else zero
        )

        # Apply seasonality (subtracted from rate)
        seasonal = self.seasonal_adjustment(t)
        adjusted_rate = r_real - seasonal

        sigma = self._params.sigma
        log_new = (
            state.log_level
            + (adjusted_rate - 0.5 * sigma**2) * dt
            + sigma * dz * jnp.sqrt(dt)
        )
        level = jnp.exp(log_new)

        new_state = FXState(log_level=log_new, level=level)
        return new_state, {OutputName.INFLATION_INDEX: level, OutputName.INFLATION_RATE: adjusted_rate}
