"""Single-factor CIR short rate model.

The Cox-Ingersoll-Ross process:
    dx = α(μ - x)dt + σ√x dZ

Provides analytic ZCB pricing via closed-form A(τ) and B(τ) functions,
which are delegated to the pure math module ``hyesg.math.cir_formulas``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import CIRState, ShockConfig
from hyesg.math.cir_formulas import (
    cir_A,
    cir_B,
    cir_euler_step,
    cir_forward_rate,
    cir_zcb_price,
)
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.config.params import CIRParams


@register_model("cir")
class CIR:
    """Single-factor CIR short rate model.

    Implements the ``ShortRateModel`` protocol with analytic bond pricing.
    The Euler step uses a floored diffusion (√max(0, x)) to prevent
    negative arguments to the square root.

    Args:
        params: CIR process parameters.
        name: Unique model name.
    """

    def __init__(self, params: CIRParams, name: str = "cir") -> None:
        self._params = params
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

    def init_state(self, params: Any = None, market: Any = None) -> CIRState:
        """Create initial state from parameters.

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial CIRState.
        """
        x0 = jnp.array(self._params.initial_value, dtype=jnp.float64)
        return CIRState(x=x0, state_var=jnp.maximum(x0, 0.0), short_rate=x0)

    def step(
        self,
        state: CIRState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[CIRState, dict[str, Any]]:
        """Advance the CIR process by one timestep.

        Uses the Euler-Maruyama scheme with a floored diffusion:
            x_{t+dt} = x_t + α(μ - x_t)dt + σ√max(0, x_t) · dZ · √dt

        Args:
            state: Current CIR state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs (unused for CIR).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma

        dz = shocks[0]
        x = state.x
        x_new, state_var = cir_euler_step(x, alpha, mu, sigma, dt, dz)
        short_rate = state_var

        new_state = CIRState(x=x_new, state_var=state_var, short_rate=short_rate)
        outputs = {OutputName.SHORT_RATE: short_rate}
        return new_state, outputs

    # ─── ShortRateModel analytics ───

    def short_rate(self, state: CIRState) -> jnp.ndarray:
        """Current short rate from state.

        Args:
            state: Current CIR state.

        Returns:
            Short rate value.
        """
        return state.short_rate

    def zcb_price(self, state: CIRState, t: float, maturity: float) -> jnp.ndarray:
        """Zero-coupon bond price P(t, T).

        Delegates to ``cir_zcb_price`` from the math module.

        Args:
            state: Current CIR state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            ZCB price.
        """
        tau = maturity - t
        return cir_zcb_price(
            tau,
            state.state_var,
            self._params.alpha,
            self._params.mu,
            self._params.sigma,
        )

    def spot_rate(self, state: CIRState, t: float, maturity: float) -> jnp.ndarray:
        """Continuously compounded spot rate R(t, T).

        R(t,T) = -ln P(t,T) / (T - t)

        Args:
            state: Current CIR state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            Spot rate.
        """
        tau = jnp.asarray(maturity - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, maturity)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)

    def forward_rate(self, state: CIRState, t: float, maturity: float) -> jnp.ndarray:
        """Instantaneous forward rate f(t, T).

        Delegates to ``cir_forward_rate`` from the math module.

        Args:
            state: Current CIR state.
            t: Current time.
            maturity: Forward time.

        Returns:
            Instantaneous forward rate.
        """
        tau = maturity - t
        return cir_forward_rate(
            tau,
            state.state_var,
            self._params.alpha,
            self._params.mu,
            self._params.sigma,
        )

    def swap_rate(
        self, state: CIRState, t: float, tenor: float, freq: float
    ) -> jnp.ndarray:
        """Par swap rate S(t; tenor, freq).

        S = (1 - P(t, t+tenor)) / Σ_{i=1}^{n} freq · P(t, t + i·freq)

        Args:
            state: Current CIR state.
            t: Current time.
            tenor: Swap tenor in years.
            freq: Payment frequency in years (e.g. 0.5 for semi-annual).

        Returns:
            Par swap rate.
        """
        n_payments = int(jnp.round(tenor / freq))
        annuity = jnp.array(0.0, dtype=jnp.float64)
        for i in range(1, n_payments + 1):
            annuity = annuity + freq * self.zcb_price(state, t, t + i * freq)
        numerator = 1.0 - self.zcb_price(state, t, t + tenor)
        return numerator / jnp.maximum(annuity, 1e-12)

    # ─── StochasticProcess interface ───

    def analytic_a(self, tau: float) -> jnp.ndarray:
        """CIR A(τ) coefficient for bond pricing.

        Args:
            tau: Time to maturity.

        Returns:
            A(τ).
        """
        return cir_A(tau, self._params.alpha, self._params.mu, self._params.sigma)

    def analytic_b(self, tau: float) -> jnp.ndarray:
        """CIR B(τ) coefficient for bond pricing.

        Args:
            tau: Time to maturity.

        Returns:
            B(τ).
        """
        return cir_B(tau, self._params.alpha, self._params.sigma)
