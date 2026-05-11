"""Credit default model using CIR++ intensity and Cox survival.

The default intensity follows a CIR++ process:
    λ(t) = y(t) + ψ(t)

where y follows the CIR dynamics:
    dy = α(μ - y)dt + σ√y dZ

and ψ(t) is a deterministic shift calibrated to match market credit
spreads.  When no market curve is supplied, ψ(t) = 0.

The Cox survival process:
    S(t) = exp(-∫₀ᵗ λ(s)ds)

is discretised as:
    S(t+dt) = S(t) · exp(-λ(t)·dt)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import CreditState, ShockConfig
from hyesg.math.cir_formulas import cir_A, cir_B, cir_forward_rate, cir_zcb_price

if TYPE_CHECKING:
    from hyesg.config.params import CreditParams
    from hyesg.math.curves.protocol import ParametricCurve


@register_model("credit")
class Credit:
    """Credit default model using CIR++ intensity and Cox survival.

    Implements the ``CreditModel`` protocol.  The underlying CIR factor
    is evolved with Euler-Maruyama and a floored diffusion (√max(0, y))
    to prevent negative square-root arguments.

    Args:
        params: Credit model parameters.
        market_curve: Optional market credit spread curve for ψ calibration.
            If ``None``, ψ(t) = 0 (no market fitting).
        name: Unique model name.
    """

    def __init__(
        self,
        params: CreditParams,
        market_curve: ParametricCurve | None = None,
        name: str = "credit",
    ) -> None:
        self._params = params
        self._market_curve = market_curve
        self._name = name

    # ─── Properties ───

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

    @property
    def recovery_rate(self) -> float:
        """Recovery rate on default."""
        return self._params.recovery_rate

    # ─── State management ───

    def init_state(self, params: Any = None, market: Any = None) -> CreditState:
        """Create initial state from parameters.

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial ``CreditState`` with zero cumulative intensity
            and no default.
        """
        intensity = jnp.array(self._params.initial_intensity, dtype=jnp.float64)
        cum_intensity = jnp.array(0.0, dtype=jnp.float64)
        has_defaulted = jnp.array(0.0, dtype=jnp.float64)
        return CreditState(
            intensity=intensity,
            cum_intensity=cum_intensity,
            has_defaulted=has_defaulted,
        )

    # ─── Step function ───

    def step(
        self,
        state: CreditState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[CreditState, dict[str, Any]]:
        """Advance the credit intensity process by one timestep.

        Algorithm:
            1. CIR Euler step for y(t) with floored diffusion.
            2. λ(t) = y(t) + ψ(t).
            3. cum_intensity += λ(t) · dt.
            4. survival = exp(-cum_intensity).

        Args:
            state: Current credit state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape ``(1,)`` with N(0,1) shock.
            deps: Dependency outputs (unused for credit).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma

        dz = shocks[0]

        # Current CIR factor (intensity minus psi at current time)
        psi_t = self._psi(t)
        y = state.intensity - psi_t

        # CIR Euler step with floored diffusion
        drift = alpha * (mu - y) * dt
        diffusion = sigma * jnp.sqrt(jnp.maximum(y, 0.0) * dt) * dz
        y_new = y + drift + diffusion
        y_new_floored = jnp.maximum(y_new, 0.0)

        # New intensity = CIR factor + psi at new time
        psi_new = self._psi(t + dt)
        intensity_new = y_new_floored + psi_new

        # Cumulative intensity and survival
        cum_intensity_new = state.cum_intensity + intensity_new * dt
        survival = jnp.exp(-cum_intensity_new)

        new_state = CreditState(
            intensity=intensity_new,
            cum_intensity=cum_intensity_new,
            has_defaulted=state.has_defaulted,
        )
        outputs = {
            "intensity": intensity_new,
            "survival_probability": survival,
            "cum_intensity": cum_intensity_new,
        }
        return new_state, outputs

    # ─── CreditModel protocol methods ───

    def default_intensity(self, state: CreditState, t: float) -> jnp.ndarray:
        """Current default intensity from state.

        Args:
            state: Current credit state.
            t: Current time (unused — intensity stored in state).

        Returns:
            Default intensity λ(t).
        """
        return state.intensity

    def survival_probability(
        self, state: CreditState, t: float, T: float
    ) -> jnp.ndarray:
        """Survival probability from current state to maturity T.

        Uses the CIR analytical ZCB formula applied to the intensity
        process to compute the conditional survival probability:
            S(t, T) = A(τ) · exp(-B(τ) · y(t)) · exp(-∫ₜᵀ ψ(s)ds)

        where τ = T - t and y(t) is the CIR factor.

        Args:
            state: Current credit state.
            t: Current time.
            T: Target time.

        Returns:
            Survival probability S(t, T).
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)

        # CIR factor from intensity
        psi_t = self._psi(t)
        y = jnp.maximum(state.intensity - psi_t, 0.0)

        # CIR ZCB price gives exp(-E[∫ y ds]) component
        cir_component = cir_zcb_price(
            tau, y, self._params.alpha, self._params.mu, self._params.sigma
        )

        # Integral of psi from t to T (numerical)
        int_psi = self._integrate_psi(t, T)

        return cir_component * jnp.exp(-int_psi)

    def has_defaulted(self, state: CreditState) -> jnp.ndarray:
        """Whether default has occurred.

        Args:
            state: Current credit state.

        Returns:
            1.0 if defaulted, 0.0 otherwise.
        """
        return state.has_defaulted

    # ─── Analytical helpers ───

    def credit_spread(
        self, state: CreditState, t: float, T: float
    ) -> jnp.ndarray:
        """Credit spread derived from survival probability.

        spread(t, T) ≈ -ln(S(t,T)) / (T - t)

        This gives the hazard-rate component of the spread. For the
        full credit spread over risk-free rates, subtract the risk-free
        rate externally.

        Args:
            state: Current credit state.
            t: Current time.
            T: Maturity time.

        Returns:
            Credit spread.
        """
        tau = jnp.asarray(T - t, dtype=jnp.float64)
        surv = self.survival_probability(state, t, T)
        return -jnp.log(surv) / jnp.maximum(tau, 1e-12)

    # ─── Private helpers ───

    def _psi(self, t: float) -> jnp.ndarray:
        """Deterministic shift ψ(t) from market curve.

        When a market curve is provided, ψ(t) = market(t) - f_CIR(0, t; y₀)
        where f_CIR is the CIR forward rate.  Otherwise ψ(t) = 0.

        Args:
            t: Time point.

        Returns:
            Shift value at time t.
        """
        if self._market_curve is None:
            return jnp.array(0.0, dtype=jnp.float64)
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        f_market = self._market_curve.evaluate(float(t_arr))
        f_cir = cir_forward_rate(
            t_arr,
            jnp.array(self._params.initial_intensity, dtype=jnp.float64),
            self._params.alpha,
            self._params.mu,
            self._params.sigma,
        )
        return jnp.asarray(f_market, dtype=jnp.float64) - f_cir

    def _integrate_psi(self, t: float, T: float) -> jnp.ndarray:
        """Numerical integral of ψ(s) from t to T.

        Args:
            t: Start time.
            T: End time.

        Returns:
            ∫ₜᵀ ψ(s) ds.
        """
        if self._market_curve is None:
            return jnp.array(0.0, dtype=jnp.float64)
        n_points = max(100, int(50 * (T - t)) + 1)
        s_values = jnp.linspace(t, T, n_points)
        psi_values = jnp.array(
            [float(self._psi(float(s))) for s in s_values],
            dtype=jnp.float64,
        )
        return jnp.trapezoid(psi_values, s_values)
