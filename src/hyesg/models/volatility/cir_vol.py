"""CIR-driven stochastic volatility model.

The variance V(t) follows CIR dynamics with time-dependent mean
reversion level μ(t):

    dV = α(μ(t) - V)dt + σ√V dZ

Provides either variance or volatility (√V) as output, and is used
as a dependency for equity and FX models supplying the time-varying
volatility term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import ShockConfig, VolState
from hyesg.math.cir_formulas import cir_euler_step
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.math.curves.protocol import ParametricCurve


@register_model("cir_vol")
class CIRVolatility:
    """CIR-driven stochastic volatility model.

    The variance follows CIR dynamics with time-dependent mean reversion
    level μ(t). Provides either variance or volatility as output.

    This model is used as a dependency for equity and FX models,
    supplying the time-varying volatility term.

    Args:
        alpha: Mean reversion speed.
        sigma: Vol-of-vol.
        v0: Initial variance.
        mu: Constant long-run mean (used when mu_curve is None).
        mu_curve: Time-dependent long-run mean μ(t) as ParametricCurve.
            If None, uses constant mu.
        output_variance: If True, output field is variance.
            If False, volatility (√V).
        name: Model name.
    """

    def __init__(
        self,
        alpha: float,
        sigma: float,
        v0: float,
        mu: float = 0.0,
        mu_curve: ParametricCurve | None = None,
        output_variance: bool = False,
        name: str = "cir_vol",
    ) -> None:
        self._alpha = alpha
        self._sigma = sigma
        self._v0 = v0
        self._mu = mu
        self._mu_curve = mu_curve
        self._output_variance = output_variance
        self._name = name

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

    def init_state(self, params: Any = None, market: Any = None) -> VolState:
        """Create initial state from parameters.

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial VolState with variance=v0, volatility=√v0.
        """
        v0 = jnp.array(self._v0, dtype=jnp.float64)
        return VolState(
            variance=v0,
            volatility=jnp.sqrt(jnp.maximum(v0, 0.0)),
        )

    def step(
        self,
        state: VolState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[VolState, dict[str, Any]]:
        """Advance the CIR variance process by one timestep.

        Uses the Euler-Maruyama scheme with a floored diffusion:
            V_{t+dt} = V_t + α(μ(t) - V_t)dt + σ√max(0, V_t) · √dt · dZ

        Args:
            state: Current VolState.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs (unused).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        v = state.variance
        dz = shocks[0]

        # Time-dependent mu: curve is evaluated with concrete t (not traced)
        if self._mu_curve is not None:
            mu_t = self._mu_curve.evaluate(float(t))
        else:
            mu_t = self._mu

        # CIR Euler step with floored diffusion
        v_new, v_floor_new = cir_euler_step(v, self._alpha, mu_t, self._sigma, dt, dz)

        vol_new = jnp.sqrt(v_floor_new)

        new_state = VolState(variance=v_new, volatility=vol_new)

        outputs: dict[str, Any] = {
            OutputName.VARIANCE: v_floor_new,
            OutputName.SIGMA: vol_new,
        }

        return new_state, outputs
