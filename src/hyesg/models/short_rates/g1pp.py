"""G1++ (Hull-White one-factor) short rate model.

The G1++ model evolves a zero-mean OU factor x(t):
    dx = -α·x·dt + σ dZ

The short rate is then:
    r(t) = x(t) + φ(t)

where φ(t) is a deterministic shift function calibrated to
match the initial market forward curve exactly:
    φ(t) = f_market(0, t) + (σ²/2α²)(1 - e^{-αt})²

ZCB pricing uses the integral of φ and the OU variance:
    ln P(t,T) = -IntegralPhi(t,T) - B(τ)·x(t) + ½V²(t,T)

where:
    B(τ) = (1 - e^{-ατ})/α
    V²(t,T) = (σ²/α²)[τ - B(τ) - ½αB(τ)²]
    IntegralPhi(t,T) = -ln[P₀(T)/P₀(t)] + ½V²(0,T) - ½V²(0,t)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import OUState, ShockConfig
from hyesg.math.gaussian_helpers import b_func, variance_integral_ou
from hyesg.math.transforms import forward_to_zcbp

if TYPE_CHECKING:
    from hyesg.config.params import OUParams
    from hyesg.math.curves.protocol import ParametricCurve


@register_model("g1pp")
class G1PP:
    """G1++ (Hull-White one-factor) short rate model.

    Implements the ``ShortRateModel`` protocol with analytic bond pricing
    calibrated to an initial market forward curve.

    The model requires a ``ParametricCurve`` representing the market
    instantaneous forward rate f(0, t). This is used to compute
    the shift function φ(t) and the initial ZCB prices P₀(t).

    Args:
        params: OU process parameters (model_type must be "g1pp").
        market_curve: Market forward rate curve f(0, t).
        name: Unique model name.
    """

    def __init__(
        self,
        params: OUParams,
        market_curve: ParametricCurve,
        name: str = "g1pp",
    ) -> None:
        if params.model_type != "g1pp":
            raise ValueError(
                f"G1PP requires model_type='g1pp', got '{params.model_type}'"
            )
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
        """Compute the shift function φ(t).

        φ(t) = f_market(0, t) + (σ²/2α²)(1 - e^{-αt})²

        Args:
            t: Time point.

        Returns:
            φ(t).
        """
        alpha = self._params.alpha
        sigma = self._params.sigma
        f_market = self._market_curve.evaluate(t)
        exp_term = (1.0 - jnp.exp(-alpha * t)) ** 2
        return f_market + (sigma**2 / (2.0 * alpha**2)) * exp_term

    def _integral_phi(self, t: float, maturity: float) -> jnp.ndarray:
        """Integral of φ from t to T for bond pricing.

        ∫ₜᵀ φ(s)ds = -ln[P₀(T)/P₀(t)] + ½V²(0,T) - ½V²(0,t)

        Args:
            t: Start time.
            maturity: End time.

        Returns:
            ∫ₜᵀ φ(s)ds.
        """
        alpha = self._params.alpha
        sigma = self._params.sigma

        p0_t = self._zcb_curve.evaluate(t)
        p0_maturity = self._zcb_curve.evaluate(maturity)
        ln_ratio = -jnp.log(jnp.maximum(p0_maturity, 1e-30) / jnp.maximum(p0_t, 1e-30))

        v2_0_maturity = variance_integral_ou(sigma, alpha, maturity)
        v2_0t = variance_integral_ou(sigma, alpha, t)

        return ln_ratio + 0.5 * v2_0_maturity - 0.5 * v2_0t

    def init_state(self, params: Any = None, market: Any = None) -> OUState:
        """Create initial state from parameters.

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial OUState with x=x₀ and r = x₀ + φ(0).
        """
        x0 = jnp.array(self._params.initial_value, dtype=jnp.float64)
        phi_0 = self._phi(0.0)
        short_rate = x0 + phi_0
        return OUState(x=x0, short_rate=short_rate)

    def step(
        self,
        state: OUState,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[OUState, dict[str, Any]]:
        """Advance the G1++ process by one timestep.

        The OU factor evolves as:
            x_{t+dt} = x_t - α·x_t·dt + σ·dZ·√dt

        Then r(t+dt) = x(t+dt) + φ(t+dt).

        Args:
            state: Current OU state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (1,) with N(0,1) shock.
            deps: Dependency outputs (unused for G1++).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        alpha = self._params.alpha
        sigma = self._params.sigma

        dz = shocks[0]
        x = state.x
        x_new = x - alpha * x * dt + sigma * jnp.sqrt(dt) * dz

        t_new = t + dt
        phi_new = self._phi(t_new)
        short_rate = x_new + phi_new

        new_state = OUState(x=x_new, short_rate=short_rate)
        outputs = {"short_rate": short_rate}
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

        ln P(t,T) = -IntegralPhi(t,T) - B(τ)·x(t) + ½V²(t,T)

        Args:
            state: Current OU state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            ZCB price.
        """
        alpha = self._params.alpha
        sigma = self._params.sigma
        tau = maturity - t

        int_phi = self._integral_phi(t, maturity)
        b = b_func(alpha, tau)
        v2 = variance_integral_ou(sigma, alpha, tau)

        ln_p = -int_phi - b * state.x + 0.5 * v2
        return jnp.exp(ln_p)

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

        Computed via numerical differentiation of -ln P(t, T).

        Args:
            state: Current OU state.
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
        """G1++ A(τ) coefficient (depends on market curve, evaluated at t=0).

        For general use, see ``zcb_price`` which handles arbitrary t.

        Args:
            tau: Time to maturity.

        Returns:
            A(τ) evaluated from t=0.
        """
        alpha = self._params.alpha
        sigma = self._params.sigma
        int_phi = self._integral_phi(0.0, tau)
        v2 = variance_integral_ou(sigma, alpha, tau)
        return jnp.exp(-int_phi + 0.5 * v2)

    def analytic_b(self, tau: float) -> jnp.ndarray:
        """G1++ B(τ) = (1 - e^{-ατ}) / α.

        Args:
            tau: Time to maturity.

        Returns:
            B(τ).
        """
        return b_func(self._params.alpha, tau)
