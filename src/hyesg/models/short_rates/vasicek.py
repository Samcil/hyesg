"""Vasicek (OU) short rate model.

The Vasicek process:
    dx = α(μ - x)dt + σ dZ

A Gaussian short rate model with analytic ZCB pricing.
The short rate can go negative, which is realistic for some
rate environments but should be noted for calibration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import OUState, ShockConfig
from hyesg.math.gaussian_helpers import b_func
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.config.params import OUParams


@register_model("vasicek")
class Vasicek:
    """Vasicek (OU) short rate model.

    Implements the ``ShortRateModel`` protocol with analytic bond pricing.
    The Vasicek model has a non-zero long-run mean μ, unlike the G1++
    sub-factor which uses μ = 0 and a shift function.

    ZCB pricing:
        P(t,T) = A(τ) · exp(-B(τ) · r(t))

    where:
        B(τ) = (1 - e^{-ατ}) / α
        ln A(τ) = (B(τ) - τ)(μ - σ²/(2α²)) - σ²B(τ)²/(4α)

    Args:
        params: OU process parameters (model_type must be "vasicek").
        name: Unique model name.
    """

    def __init__(self, params: OUParams, name: str = "vasicek") -> None:
        if params.model_type != "vasicek":
            raise ValueError(
                f"Vasicek requires model_type='vasicek', got '{params.model_type}'"
            )
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

    def init_state(self, params: Any = None, market: Any = None) -> OUState:
        """Create initial state from parameters.

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial OUState.
        """
        x0 = jnp.array(self._params.initial_value, dtype=jnp.float64)
        return OUState(x=x0, short_rate=x0)

    def step(
        self,
        state: OUState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[OUState, dict[str, Any]]:
        """Advance the Vasicek process by one timestep.

        Uses the Euler-Maruyama scheme:
            x_{t+dt} = x_t + α(μ - x_t)dt + σ · dZ · √dt

        Args:
            state: Current OU state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs (unused for Vasicek).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma

        dz = shocks[0]
        x = state.x
        x_new = x + alpha * (mu - x) * dt + sigma * jnp.sqrt(dt) * dz
        short_rate = x_new

        new_state = OUState(x=x_new, short_rate=short_rate)
        outputs = {OutputName.SHORT_RATE: short_rate}
        return new_state, outputs

    # ─── ShortRateModel analytics ───

    def short_rate(self, state: OUState) -> jnp.ndarray:
        """Current short rate from state.

        Args:
            state: Current OU state.

        Returns:
            Short rate value.
        """
        return state.short_rate

    def zcb_price(self, state: OUState, t: float, maturity: float) -> jnp.ndarray:
        """Zero-coupon bond price P(t, T).

        P(t,T) = A(τ) · exp(-B(τ) · r(t))

        where τ = T - t,
            B(τ) = (1 - e^{-ατ}) / α
            ln A(τ) = (B(τ) - τ)(μ - σ²/(2α²)) - σ²B(τ)²/(4α)

        Args:
            state: Current OU state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            ZCB price.
        """
        tau = jnp.asarray(maturity - t, dtype=jnp.float64)
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma

        b = b_func(alpha, tau)
        ln_a = (b - tau) * (mu - sigma**2 / (2.0 * alpha**2)) - (
            sigma**2 * b**2 / (4.0 * alpha)
        )
        return jnp.exp(ln_a - b * state.short_rate)

    def spot_rate(self, state: OUState, t: float, maturity: float) -> jnp.ndarray:
        """Continuously compounded spot rate R(t, T).

        R(t,T) = -ln P(t,T) / (T - t)

        Args:
            state: Current OU state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            Spot rate.
        """
        tau = jnp.asarray(maturity - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, maturity)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)

    def forward_rate(self, state: OUState, t: float, maturity: float) -> jnp.ndarray:
        """Instantaneous forward rate f(t, T).

        f(t,T) = -∂/∂T ln P(t,T)

        Computed analytically:
            f(t,T) = μ + (r - μ)e^{-ατ} - (σ²/2α²)(1 - e^{-ατ})²

        Args:
            state: Current OU state.
            t: Current time.
            maturity: Forward time.

        Returns:
            Instantaneous forward rate.
        """
        tau = jnp.asarray(maturity - t, dtype=jnp.float64)
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma

        exp_at = jnp.exp(-alpha * tau)
        return (
            mu
            + (state.short_rate - mu) * exp_at
            - (sigma**2 / (2.0 * alpha**2)) * (1.0 - exp_at) ** 2
        )

    def swap_rate(
        self, state: OUState, t: float, tenor: float, freq: float
    ) -> jnp.ndarray:
        """Par swap rate S(t; tenor, freq).

        S = (1 - P(t, t+tenor)) / Σ freq · P(t, t + i·freq)

        Args:
            state: Current OU state.
            t: Current time.
            tenor: Swap tenor in years.
            freq: Payment frequency in years.

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
        """Vasicek A(τ) coefficient for bond pricing.

        Args:
            tau: Time to maturity.

        Returns:
            A(τ).
        """
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma
        b = b_func(alpha, tau)
        ln_a = (b - tau) * (mu - sigma**2 / (2.0 * alpha**2)) - (
            sigma**2 * b**2 / (4.0 * alpha)
        )
        return jnp.exp(ln_a)

    def analytic_b(self, tau: float) -> jnp.ndarray:
        """Vasicek B(τ) = (1 - e^{-ατ}) / α.

        Args:
            tau: Time to maturity.

        Returns:
            B(τ).
        """
        return b_func(self._params.alpha, tau)
