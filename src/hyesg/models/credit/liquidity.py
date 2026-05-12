"""Liquidity process using independent CIR dynamics.

Models illiquidity events as an independent Cox process with CIR
intensity.  Three tiers: High (no liquidity process), Medium, Low.
Each tier has its own CIR parameters and an RN→RW transform scaled
from the base credit transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp

from hyesg.math.cir_formulas import cir_euler_step
from hyesg.models.credit.intensity_transform import (
    IntensityTransform,
    ScaledIntensityTransform,
)

if TYPE_CHECKING:
    from jax import Array


class LiquidityState(NamedTuple):
    """State of a liquidity CIR process.

    Attributes:
        intensity: Current CIR liquidity intensity.
        cum_intensity: Cumulative integrated intensity.
        has_triggered: Whether a liquidity event has triggered (1.0/0.0).
        trigger_time: Time at which the event triggered (inf if not).
    """

    intensity: Array
    cum_intensity: Array
    has_triggered: Array
    trigger_time: Array
    threshold: Array


@runtime_checkable
class LiquidityProcess(Protocol):
    """Protocol for liquidity event processes."""

    def init_state(self, key: Array | None = None) -> LiquidityState:
        """Create initial liquidity state.

        Args:
            key: JAX PRNG key for threshold draw. If None, a default
                key is created internally.

        Returns:
            Initial ``LiquidityState``.
        """
        ...

    def step(
        self,
        state: LiquidityState,
        t: float,
        dt: float,
        dz: Array,
    ) -> LiquidityState:
        """Advance the liquidity process by one timestep.

        Args:
            state: Current state.
            t: Current time in years.
            dt: Timestep size.
            dz: N(0,1) Brownian increment.

        Returns:
            Updated ``LiquidityState``.
        """
        ...


class CIRLiquidityProcess:
    """Independent CIR process for liquidity events.

    The liquidity intensity follows CIR dynamics:
        dx = alpha * (mu - x) * dt + sigma * sqrt(x) * dZ

    A liquidity event triggers when the cumulative intensity exceeds
    a uniform threshold drawn at initialisation (Cox process).

    Three tiers exist:
    - High: no liquidity process (handled externally — not instantiated)
    - Medium: (x0=0.04, mu=0.1, alpha=0.0225, sigma=0.1)
    - Low: (x0=0.08, mu=0.3, alpha=0.0225, sigma=0.12)

    The RN→RW transform is the base credit spline scaled by 0.1.

    Args:
        alpha: Mean-reversion speed.
        mu: Long-run mean intensity.
        sigma: CIR volatility.
        x0: Initial intensity value.
        rn_transform: RN→RW intensity transform for this tier.
        scale_factor: Scale factor applied to base transform (default 0.1).
        recovery_rate: Recovery rate on liquidity event (default 0.75).
    """

    def __init__(
        self,
        alpha: float,
        mu: float,
        sigma: float,
        x0: float,
        rn_transform: IntensityTransform,
        scale_factor: float = 0.1,
        recovery_rate: float = 0.75,
    ) -> None:
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma
        self._x0 = x0
        self._transform = ScaledIntensityTransform(rn_transform, scale_factor)
        self._recovery_rate = recovery_rate

    @property
    def recovery_rate(self) -> float:
        """Recovery rate on liquidity event."""
        return self._recovery_rate

    def init_state(self, key: Array | None = None) -> LiquidityState:
        """Create initial liquidity state.

        Draws a uniform threshold for Cox process default detection.

        Args:
            key: JAX PRNG key. If None, a default key is created
                internally.

        Returns:
            Initial ``LiquidityState``.
        """
        import jax

        if key is None:
            key = jax.random.PRNGKey(0)

        u = jax.random.uniform(key, dtype=jnp.float64)
        return LiquidityState(
            intensity=jnp.asarray(self._x0, dtype=jnp.float64),
            cum_intensity=jnp.array(0.0, dtype=jnp.float64),
            has_triggered=jnp.array(0.0, dtype=jnp.float64),
            trigger_time=jnp.array(jnp.inf, dtype=jnp.float64),
            threshold=u,
        )

    def step(
        self,
        state: LiquidityState,
        t: float,
        dt: float,
        dz: Array,
    ) -> LiquidityState:
        """Advance the liquidity CIR process by one timestep.

        Uses Euler-Maruyama discretisation with floored diffusion.

        Args:
            state: Current liquidity state.
            t: Current time in years.
            dt: Timestep size.
            dz: N(0,1) Brownian increment.

        Returns:
            Updated ``LiquidityState``.
        """
        x = state.intensity

        # CIR Euler step with floored diffusion
        _, x_new = cir_euler_step(x, self._alpha, self._mu, self._sigma, dt, dz)

        # Apply RN→RW transform for real-world intensity
        rw_intensity = self._transform.transform(x_new)

        # Cumulative intensity
        new_cum = state.cum_intensity + rw_intensity * dt

        # Cox process trigger: default when exp(-cum_intensity) < U
        survival = jnp.exp(-new_cum)
        newly_triggered = jnp.where(
            state.has_triggered > 0.5,
            state.has_triggered,
            jnp.where(survival < state.threshold, 1.0, 0.0),
        )

        new_trigger_time = jnp.where(
            (newly_triggered > 0.5) & (state.has_triggered < 0.5),
            jnp.asarray(t + dt, dtype=jnp.float64),
            state.trigger_time,
        )

        return LiquidityState(
            intensity=x_new,
            cum_intensity=new_cum,
            has_triggered=newly_triggered,
            trigger_time=new_trigger_time,
            threshold=state.threshold,
        )
