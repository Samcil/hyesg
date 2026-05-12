"""G2++ (two-factor Gaussian) short rate model.

The G2++ model evolves two correlated zero-mean OU factors:
    dx₁ = -α₁·x₁·dt + σ₁·dZ₁
    dx₂ = -α₂·x₂·dt + σ₂·dZ₂
    dZ₁·dZ₂ = ρ·dt

The short rate is then:
    r(t) = x₁(t) + x₂(t) + φ(t)

where φ(t) is a deterministic shift calibrated to the initial market
forward curve:
    φ(t) = f_market(0,t)
           + (σ₁²/2α₁²)(1-e^{-α₁t})²
           + (σ₂²/2α₂²)(1-e^{-α₂t})²
           + ρσ₁σ₂/(α₁α₂)(1-e^{-α₁t})(1-e^{-α₂t})

ZCB pricing uses the two-factor generalisation:
    ln P(t,T) = -IntegralPhi(t,T) - M₁·x₁ - M₂·x₂ + ½V²(t,T)

where V² includes variance contributions from both factors plus
a cross-correlation term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import G2State, ShockConfig
from hyesg.math.gaussian_helpers import b_func, variance_integral_ou
from hyesg.math.transforms import forward_to_zcbp
from hyesg.outputs import OutputName

if TYPE_CHECKING:
    from hyesg.config.params import G2PPParams
    from hyesg.math.curves.protocol import ParametricCurve


def _v_squared_full(
    sigma1: float,
    alpha1: float,
    sigma2: float,
    alpha2: float,
    rho: float,
    tau: jnp.ndarray | float,
) -> jnp.ndarray:
    """Full V²(t, T) including cross-correlation term.

    V² = V₁² + V₂² + 2ρσ₁σ₂/(α₁α₂)[τ - B₁ - B₂ + B₁₂]

    where:
        V_i² = variance_integral_ou(σᵢ, αᵢ, τ)
        B₁ = (1-e^{-α₁τ})/α₁
        B₂ = (1-e^{-α₂τ})/α₂
        B₁₂ = (1-e^{-(α₁+α₂)τ})/(α₁+α₂)

    Args:
        sigma1: Volatility of factor 1.
        alpha1: Mean-reversion speed of factor 1.
        sigma2: Volatility of factor 2.
        alpha2: Mean-reversion speed of factor 2.
        rho: Correlation between the two factors.
        tau: Time to maturity T - t.

    Returns:
        Full V² value.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)

    v2_1 = variance_integral_ou(sigma1, alpha1, tau)
    v2_2 = variance_integral_ou(sigma2, alpha2, tau)

    b1 = b_func(alpha1, tau)
    b2 = b_func(alpha2, tau)
    b12 = b_func(alpha1 + alpha2, tau)

    cross = 2.0 * rho * sigma1 * sigma2 / (alpha1 * alpha2) * (
        tau - b1 - b2 + b12
    )

    return v2_1 + v2_2 + cross


@register_model("g2pp")
class G2PP:
    """G2++ (two-factor Gaussian) short rate model.

    Implements the ``ShortRateModel`` protocol with analytic bond pricing
    calibrated to an initial market forward curve. Extends the G1++ model
    to two correlated OU factors for richer term-structure dynamics.

    The model requires a ``ParametricCurve`` representing the market
    instantaneous forward rate f(0, t). This is used to compute the shift
    function φ(t) and the initial ZCB prices P₀(t).

    Args:
        params: G2++ model parameters.
        market_curve: Market forward rate curve f(0, t).
        name: Unique model name.
    """

    def __init__(
        self,
        params: G2PPParams,
        market_curve: ParametricCurve,
        name: str = "g2pp",
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
        return 2

    @property
    def shock_config(self) -> ShockConfig:
        """Shock metadata for the correlation engine."""
        return ShockConfig(
            n_shocks=2,
            distribution="normal",
            correlate=True,
            names=(f"{self._name}_z1", f"{self._name}_z2"),
        )

    def _phi(self, t: float) -> jnp.ndarray:
        """Compute the shift function φ(t).

        φ(t) = f_market(0,t)
               + (σ₁²/2α₁²)(1-e^{-α₁t})²
               + (σ₂²/2α₂²)(1-e^{-α₂t})²
               + ρσ₁σ₂/(α₁α₂)(1-e^{-α₁t})(1-e^{-α₂t})

        Args:
            t: Time point.

        Returns:
            φ(t).
        """
        p = self._params
        f_market = self._market_curve.evaluate(t)

        e1 = 1.0 - jnp.exp(-p.alpha1 * t)
        e2 = 1.0 - jnp.exp(-p.alpha2 * t)

        term1 = (p.sigma1**2 / (2.0 * p.alpha1**2)) * e1**2
        term2 = (p.sigma2**2 / (2.0 * p.alpha2**2)) * e2**2
        cross = p.rho * p.sigma1 * p.sigma2 / (p.alpha1 * p.alpha2) * e1 * e2

        return f_market + term1 + term2 + cross

    def _integral_phi(self, t: float, maturity: float) -> jnp.ndarray:
        """Integral of φ from t to T for bond pricing.

        ∫ₜᵀ φ(s)ds = -ln[P₀(T)/P₀(t)] + ½V²(0,T) - ½V²(0,t)

        where V² is the full two-factor variance including cross-term.

        Args:
            t: Start time.
            maturity: End time.

        Returns:
            ∫ₜᵀ φ(s)ds.
        """
        p = self._params

        p0_t = self._zcb_curve.evaluate(t)
        p0_maturity = self._zcb_curve.evaluate(maturity)
        ln_ratio = -jnp.log(
            jnp.maximum(p0_maturity, 1e-30) / jnp.maximum(p0_t, 1e-30)
        )

        v2_0_T = _v_squared_full(
            p.sigma1, p.alpha1, p.sigma2, p.alpha2, p.rho, maturity
        )
        v2_0_t = _v_squared_full(
            p.sigma1, p.alpha1, p.sigma2, p.alpha2, p.rho, t
        )

        return ln_ratio + 0.5 * v2_0_T - 0.5 * v2_0_t

    def init_state(self, params: Any = None, market: Any = None) -> G2State:
        """Create initial state from parameters.

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial G2State with x₁=x₁₀, x₂=x₂₀, and r = x₁₀ + x₂₀ + φ(0).
        """
        x1_0 = jnp.array(self._params.x1_initial, dtype=jnp.float64)
        x2_0 = jnp.array(self._params.x2_initial, dtype=jnp.float64)
        phi_0 = self._phi(0.0)
        short_rate = x1_0 + x2_0 + phi_0
        return G2State(x1=x1_0, x2=x2_0, short_rate=short_rate)

    def step(
        self,
        state: G2State,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[G2State, dict[str, Any]]:
        """Advance the G2++ process by one timestep.

        Each OU factor evolves independently:
            x₁_{t+dt} = x₁_t - α₁·x₁_t·dt + σ₁·dZ₁·√dt
            x₂_{t+dt} = x₂_t - α₂·x₂_t·dt + σ₂·dZ₂·√dt

        Then r(t+dt) = x₁(t+dt) + x₂(t+dt) + φ(t+dt).

        Note: Shocks arrive already correlated from the correlation engine.
        Do NOT apply ρ here.

        Args:
            state: Current G2 state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (2,) with N(0,1) shocks.
            deps: Dependency outputs (unused for G2++).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        p = self._params
        sqrt_dt = jnp.sqrt(dt)

        dz1 = shocks[0]
        dz2 = shocks[1]

        x1_new = state.x1 - p.alpha1 * state.x1 * dt + p.sigma1 * sqrt_dt * dz1
        x2_new = state.x2 - p.alpha2 * state.x2 * dt + p.sigma2 * sqrt_dt * dz2

        t_new = t + dt
        phi_new = self._phi(t_new)
        short_rate = x1_new + x2_new + phi_new

        new_state = G2State(x1=x1_new, x2=x2_new, short_rate=short_rate)
        outputs = {OutputName.SHORT_RATE: short_rate}
        return new_state, outputs

    # ─── ShortRateModel analytics ───

    def short_rate(self, state: G2State) -> jnp.ndarray:
        """Current short rate from state.

        Args:
            state: Current G2 state.

        Returns:
            Short rate value.
        """
        return state.short_rate

    def zcb_price(
        self, state: G2State, t: float, maturity: float
    ) -> jnp.ndarray:
        """Zero-coupon bond price P(t, T).

        ln P(t,T) = -IntegralPhi(t,T) - M₁·x₁ - M₂·x₂ + ½V²(t,T)

        where M_i = B(αᵢ, τ) and V² includes the cross-correlation term.

        Args:
            state: Current G2 state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            ZCB price.
        """
        p = self._params
        tau = maturity - t

        int_phi = self._integral_phi(t, maturity)
        m1 = b_func(p.alpha1, tau)
        m2 = b_func(p.alpha2, tau)
        v2 = _v_squared_full(
            p.sigma1, p.alpha1, p.sigma2, p.alpha2, p.rho, tau
        )

        ln_p = -int_phi - m1 * state.x1 - m2 * state.x2 + 0.5 * v2
        return jnp.exp(ln_p)

    def spot_rate(
        self, state: G2State, t: float, maturity: float
    ) -> jnp.ndarray:
        """Continuously compounded spot rate R(t, T).

        R(t,T) = -ln P(t,T) / (T - t)

        Args:
            state: Current G2 state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            Spot rate.
        """
        tau = jnp.asarray(maturity - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, maturity)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)

    def forward_rate(
        self, state: G2State, t: float, maturity: float
    ) -> jnp.ndarray:
        """Instantaneous forward rate f(t, T).

        Computed via numerical differentiation of -ln P(t, T).

        Args:
            state: Current G2 state.
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
        self, state: G2State, t: float, tenor: float, freq: float
    ) -> jnp.ndarray:
        """Par swap rate S(t; tenor, freq).

        S = (1 - P(t, t+tenor)) / Σ freq · P(t, t + i·freq)

        Args:
            state: Current G2 state.
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
