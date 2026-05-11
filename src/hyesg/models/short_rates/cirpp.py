"""CIR++ (market-fitted CIR) short rate model.

The CIR++ model adds a deterministic shift φ(t) to the CIR process
so that the model matches the initial market yield curve exactly:

    r(t) = x(t) + φ(t)

where x follows the CIR process:
    dx = α(μ - x)dt + σ√x dZ

and φ(t) = f_market(0,t) - f_CIR(0,t; x₀) is chosen to match
the market instantaneous forward curve at time 0.

ZCB pricing uses the FROM TIME 0 formulation:
    P(t,T) = exp(-∫ₜᵀ φ(s)ds) · [A(0,T)/A(0,t)] · exp(-(B(0,T)-B(0,t))·x(t))

This differs from plain CIR which uses A(T-t), B(T-t) directly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import CIRState, ShockConfig
from hyesg.math.cir_formulas import (
    cir_A,
    cir_B,
    cir_forward_rate,
    cir_integral_phi,
    cir_phi_from_curves,
)
from hyesg.math.transforms import forward_to_zcbp

if TYPE_CHECKING:
    from hyesg.config.params import CIRParams
    from hyesg.math.curves.protocol import ParametricCurve


@register_model("cirpp")
class CIRPlusPlus:
    """CIR++ model: CIR + deterministic shift to match market curve.

    The model evolves a CIR factor x(t) and adds a pre-computed
    deterministic shift φ(t) so that the initial forward curve is
    matched exactly. ZCB pricing uses the FROM TIME 0 formulation
    with A(0,T)/A(0,t) rather than A(T-t).

    Args:
        params: CIR process parameters (α, μ, σ, x₀).
        market_curve: Market instantaneous forward rate curve f(0, t).
        name: Unique model name.
    """

    def __init__(
        self,
        params: CIRParams,
        market_curve: ParametricCurve,
        name: str = "cirpp",
    ) -> None:
        self._params = params
        self._market_curve = market_curve
        self._zcb_curve = forward_to_zcbp(market_curve)
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

    def _phi(self, t: float) -> jnp.ndarray:
        """Compute the deterministic shift φ(t).

        φ(t) = f_market(0,t) - f_CIR(0,t; x₀)

        Applies non-negativity clamping: values in (-1e-4, 0) are
        clamped to 0; values below -1e-4 trigger a warning.

        Args:
            t: Time point.

        Returns:
            φ(t) value.
        """
        phi_val = cir_phi_from_curves(
            t,
            self._market_curve.evaluate,
            self._params.alpha,
            self._params.mu,
            self._params.sigma,
            self._params.initial_value,
        )
        phi_val = jnp.asarray(phi_val, dtype=jnp.float64)

        phi_float = float(phi_val)
        if phi_float < -1e-4:
            warnings.warn(
                f"CIR++ phi({float(t):.4f}) = {phi_float:.6f} is significantly "
                f"negative. Market curve may be inconsistent with CIR parameters.",
                UserWarning,
                stacklevel=2,
            )
        # Clamp small negative values to zero
        return jnp.maximum(phi_val, 0.0)

    def _forward_curve_fn(self, t: float) -> float:
        """Evaluate the market forward curve at time t.

        Args:
            t: Time point.

        Returns:
            Market instantaneous forward rate f(0, t).
        """
        return self._market_curve.evaluate(t)

    def init_state(self, params: Any = None, market: Any = None) -> CIRState:
        """Create initial state from parameters.

        The initial short rate is x₀ + φ(0).

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial CIRState with r(0) = x₀ + φ(0).
        """
        x0 = jnp.array(self._params.initial_value, dtype=jnp.float64)
        state_var = jnp.maximum(x0, 0.0)
        phi_0 = self._phi(0.0)
        short_rate = state_var + phi_0
        return CIRState(x=x0, state_var=state_var, short_rate=short_rate)

    def step(
        self,
        state: CIRState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[CIRState, dict[str, Any]]:
        """Advance the CIR++ process by one timestep.

        The CIR factor evolves via Euler-Maruyama:
            x_{t+dt} = x_t + α(μ - x_t)dt + σ√max(0, x_t) · dZ · √dt

        Then r(t+dt) = max(0, x_{t+dt}) + φ(t+dt).

        Args:
            state: Current CIR state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs (unused for CIR++).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma

        dz = shocks[0]
        x = state.x
        drift = alpha * (mu - x) * dt
        diffusion = sigma * jnp.sqrt(jnp.maximum(x, 0.0) * dt) * dz
        x_new = x + drift + diffusion
        state_var = jnp.maximum(x_new, 0.0)

        t_new = t + dt
        phi_new = self._phi(t_new)
        short_rate = state_var + phi_new

        new_state = CIRState(x=x_new, state_var=state_var, short_rate=short_rate)
        outputs = {"short_rate": short_rate}
        return new_state, outputs

    # ─── ShortRateModel analytics ───

    def short_rate(self, state: CIRState) -> jnp.ndarray:
        """Current short rate from state.

        Args:
            state: Current CIR state.

        Returns:
            Short rate value r(t) = x(t) + φ(t).
        """
        return state.short_rate

    def zcb_price(self, state: CIRState, t: float, maturity: float) -> jnp.ndarray:
        """Zero-coupon bond price P(t, T) using FROM TIME 0 formulation.

        P(t,T) = exp(-∫ₜᵀ φ(s)ds) · [A(0,T)/A(0,t)] · exp(-(B(0,T)-B(0,t))·x(t))

        This uses A(0,T)/A(0,t) and B(0,T)-B(0,t) computed from time 0,
        NOT the plain CIR A(T-t), B(T-t) formulas.

        Args:
            state: Current CIR state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            ZCB price.
        """
        alpha = self._params.alpha
        mu = self._params.mu
        sigma = self._params.sigma
        x0 = self._params.initial_value

        # Integral of phi from t to T
        int_phi = cir_integral_phi(
            t, maturity, alpha, mu, sigma, x0, self._forward_curve_fn
        )

        # A(0,T) / A(0,t) — FROM TIME 0 formulation
        a_0_T = cir_A(maturity, alpha, mu, sigma)
        a_0_t = cir_A(t, alpha, mu, sigma)
        a_ratio = a_0_T / jnp.maximum(a_0_t, 1e-30)

        # B(0,T) - B(0,t) — FROM TIME 0 formulation
        b_0_T = cir_B(maturity, alpha, sigma)
        b_0_t = cir_B(t, alpha, sigma)
        b_diff = b_0_T - b_0_t

        return jnp.exp(-int_phi) * a_ratio * jnp.exp(-b_diff * state.state_var)

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

        Computed via numerical differentiation of -ln P(t, T).

        Args:
            state: Current CIR state.
            t: Current time.
            maturity: Forward time.

        Returns:
            Instantaneous forward rate.
        """
        eps = 1e-6
        p_plus = self.zcb_price(state, t, maturity + eps)
        p_minus = self.zcb_price(state, t, maturity - eps)
        return -(jnp.log(p_plus) - jnp.log(p_minus)) / (2.0 * eps)

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
