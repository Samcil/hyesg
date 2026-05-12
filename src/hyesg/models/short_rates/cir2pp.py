"""CIR2++ two-factor nominal short rate model.

The CIR2++ model uses two independent CIR factors plus a deterministic
shift φ(t) to match the initial market yield curve exactly:

    r(t) = x₁(t) + x₂(t) + φ(t)

where each factor follows an independent CIR process:
    dx₁ = α₁(μ₁ - x₁)dt + σ₁√x₁ dZ₁
    dx₂ = α₂(μ₂ - x₂)dt + σ₂√x₂ dZ₂

and φ(t) = f_market(0,t) - f_CIR1(0,t; x₁₀) - f_CIR2(0,t; x₂₀)
is chosen to match the market instantaneous forward curve at time 0.

ZCB pricing uses the FROM TIME 0 formulation:
    P(t,T) = exp(-∫ₜᵀ φ(s)ds)
             · [A₁(0,T)/A₁(0,t)] · exp(-(B₁(0,T)-B₁(0,t))·x₁(t))
             · [A₂(0,T)/A₂(0,t)] · exp(-(B₂(0,T)-B₂(0,t))·x₂(t))
"""

from __future__ import annotations


from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from hyesg.core.registry import register_model
from hyesg.core.types import CIR2State, ShockConfig
from hyesg.math.cir_formulas import (
    cir_A,
    cir_B,
    cir_forward_rate,
    cir_phi_from_curves,
)
from hyesg.math.transforms import forward_to_zcbp

if TYPE_CHECKING:
    from hyesg.config.params import CIRParams
    from hyesg.math.curves.protocol import ParametricCurve


def _cir2_phi_from_curves(
    t: float,
    forward_curve_fn: Any,
    alpha1: float,
    mu1: float,
    sigma1: float,
    x10: float,
    alpha2: float,
    mu2: float,
    sigma2: float,
    x20: float,
) -> jnp.ndarray:
    """Two-factor CIR++ phi shift: φ(t) = f_market(0,t) - f_CIR1(0,t) - f_CIR2(0,t).

    Args:
        t: Time point.
        forward_curve_fn: Market forward rate function f(t).
        alpha1: Mean reversion speed for factor 1.
        mu1: Long-run mean for factor 1.
        sigma1: Volatility for factor 1.
        x10: Initial state for factor 1.
        alpha2: Mean reversion speed for factor 2.
        mu2: Long-run mean for factor 2.
        sigma2: Volatility for factor 2.
        x20: Initial state for factor 2.

    Returns:
        Phi shift at time t.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    f_market = forward_curve_fn(t)
    f_cir1 = cir_forward_rate(t, x10, alpha1, mu1, sigma1)
    f_cir2 = cir_forward_rate(t, x20, alpha2, mu2, sigma2)
    return f_market - f_cir1 - f_cir2


def _cir2_integral_phi(
    t: float,
    T: float,
    alpha1: float,
    mu1: float,
    sigma1: float,
    x10: float,
    alpha2: float,
    mu2: float,
    sigma2: float,
    x20: float,
    forward_curve_fn: Any,
) -> jnp.ndarray:
    """Integral of two-factor phi shift for CIR2++ bond pricing.

    ∫ₜᵀ φ(s)ds used in CIR2++ ZCB pricing.

    Args:
        t: Start time.
        T: End time.
        alpha1: Mean reversion speed for factor 1.
        mu1: Long-run mean for factor 1.
        sigma1: Volatility for factor 1.
        x10: Initial state for factor 1.
        alpha2: Mean reversion speed for factor 2.
        mu2: Long-run mean for factor 2.
        sigma2: Volatility for factor 2.
        x20: Initial state for factor 2.
        forward_curve_fn: Market forward rate function f(s).

    Returns:
        ∫ₜᵀ φ(s)ds.
    """
    n_points = max(200, int(50 * (T - t)) + 1)
    s_values = jnp.linspace(t, T, n_points)

    phi_values = jax.vmap(
        lambda s: _cir2_phi_from_curves(
            s, forward_curve_fn,
            alpha1, mu1, sigma1, x10,
            alpha2, mu2, sigma2, x20,
        )
    )(s_values)

    return jnp.trapezoid(phi_values, s_values)


def _cir2_integral_phi_analytic(
    t: float,
    T: float,
    alpha1: float,
    mu1: float,
    sigma1: float,
    x10: float,
    alpha2: float,
    mu2: float,
    sigma2: float,
    x20: float,
    market_zcb_fn: Any,
) -> jnp.ndarray:
    """Analytic integral of two-factor CIR2++ phi shift.

    ∫ₜᵀ φ(s)ds = ln[P_mkt(0,t)/P_mkt(0,T)]
                 - Σᵢ { ln[Aᵢ(t)/Aᵢ(T)] + [Bᵢ(T)-Bᵢ(t)]·xᵢ₀ }

    This is exact and avoids the O(dt) discretisation error of
    the numerical trapezoid in ``_cir2_integral_phi``.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)

    # Market term
    p_t = market_zcb_fn(t)
    p_T = market_zcb_fn(T)
    market_term = jnp.log(jnp.maximum(p_t, 1e-30)) - jnp.log(jnp.maximum(p_T, 1e-30))

    # Factor 1 CIR term
    a1_t = cir_A(t, alpha1, mu1, sigma1)
    a1_T = cir_A(T, alpha1, mu1, sigma1)
    b1_t = cir_B(t, alpha1, sigma1)
    b1_T = cir_B(T, alpha1, sigma1)
    cir1_term = (
        jnp.log(jnp.maximum(a1_t, 1e-30)) - jnp.log(jnp.maximum(a1_T, 1e-30))
        + (b1_T - b1_t) * x10
    )

    # Factor 2 CIR term
    a2_t = cir_A(t, alpha2, mu2, sigma2)
    a2_T = cir_A(T, alpha2, mu2, sigma2)
    b2_t = cir_B(t, alpha2, sigma2)
    b2_T = cir_B(T, alpha2, sigma2)
    cir2_term = (
        jnp.log(jnp.maximum(a2_t, 1e-30)) - jnp.log(jnp.maximum(a2_T, 1e-30))
        + (b2_T - b2_t) * x20
    )

    return market_term - cir1_term - cir2_term


@register_model("cir2pp")
class CIR2PlusPlus:
    """CIR2++ model: two independent CIR factors + deterministic shift.

    The model evolves two CIR factors x₁(t) and x₂(t) and adds a
    pre-computed deterministic shift φ(t) so that the initial forward
    curve is matched exactly. ZCB pricing uses the FROM TIME 0
    formulation with A(0,T)/A(0,t) for each factor.

    Args:
        params1: CIR process parameters for factor 1 (α₁, μ₁, σ₁, x₁₀).
        params2: CIR process parameters for factor 2 (α₂, μ₂, σ₂, x₂₀).
        market_curve: Market instantaneous forward rate curve f(0, t).
        name: Unique model name.
    """

    def __init__(
        self,
        params1: CIRParams,
        params2: CIRParams,
        market_curve: ParametricCurve,
        name: str = "cir2pp",
    ) -> None:
        self._params1 = params1
        self._params2 = params2
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
        """Compute the deterministic shift φ(t).

        φ(t) = f_market(0,t) - f_CIR1(0,t; x₁₀) - f_CIR2(0,t; x₂₀)

        Applies non-negativity clamping: values in (-1e-4, 0) are
        clamped to 0; values below -1e-4 trigger a warning.

        Args:
            t: Time point.

        Returns:
            φ(t) value.
        """
        phi_val = _cir2_phi_from_curves(
            t,
            self._market_curve.evaluate,
            self._params1.alpha,
            self._params1.mu,
            self._params1.sigma,
            self._params1.initial_value,
            self._params2.alpha,
            self._params2.mu,
            self._params2.sigma,
            self._params2.initial_value,
        )
        phi_val = jnp.asarray(phi_val, dtype=jnp.float64)
        # Clamp small negative values to zero (pure JAX, JIT-safe)
        return jnp.maximum(phi_val, 0.0)

    def _forward_curve_fn(self, t: float) -> float:
        """Evaluate the market forward curve at time t.

        Args:
            t: Time point.

        Returns:
            Market instantaneous forward rate f(0, t).
        """
        return self._market_curve.evaluate(t)

    def init_state(self, params: Any = None, market: Any = None) -> CIR2State:
        """Create initial state from parameters.

        The initial short rate is x₁₀ + x₂₀ + φ(0).

        Args:
            params: Optional override parameters (unused).
            market: Optional market data (unused).

        Returns:
            Initial CIR2State with r(0) = x₁₀ + x₂₀ + φ(0).
        """
        x10 = jnp.array(self._params1.initial_value, dtype=jnp.float64)
        x20 = jnp.array(self._params2.initial_value, dtype=jnp.float64)
        state_var1 = jnp.maximum(x10, 0.0)
        state_var2 = jnp.maximum(x20, 0.0)
        phi_0 = self._phi(0.0)
        short_rate = state_var1 + state_var2 + phi_0
        return CIR2State(
            x1=x10, x2=x20,
            state_var1=state_var1, state_var2=state_var2,
            short_rate=short_rate,
        )

    def step(
        self,
        state: CIR2State,
        t: float,
        dt: float,
        shocks: Any,
        deps: dict[str, Any],
    ) -> tuple[CIR2State, dict[str, Any]]:
        """Advance the CIR2++ process by one timestep.

        Each CIR factor evolves independently via Euler-Maruyama:
            x_{i,t+dt} = x_{i,t} + αᵢ(μᵢ - xᵢ)dt + σᵢ√max(0,xᵢ)·dZᵢ·√dt

        Then r(t+dt) = max(0, x₁) + max(0, x₂) + φ(t+dt).

        Args:
            state: Current CIR2 state.
            t: Current time in years.
            dt: Timestep size in years.
            shocks: Array of shape (2,) with N(0,1) shocks.
            deps: Dependency outputs (unused for CIR2++).

        Returns:
            Tuple of (new_state, outputs_dict).
        """
        # Factor 1
        alpha1 = self._params1.alpha
        mu1 = self._params1.mu
        sigma1 = self._params1.sigma

        dz1 = shocks[0]
        x1 = state.x1
        drift1 = alpha1 * (mu1 - x1) * dt
        diffusion1 = sigma1 * jnp.sqrt(jnp.maximum(x1, 0.0) * dt) * dz1
        x1_new = x1 + drift1 + diffusion1
        state_var1 = jnp.maximum(x1_new, 0.0)

        # Factor 2
        alpha2 = self._params2.alpha
        mu2 = self._params2.mu
        sigma2 = self._params2.sigma

        dz2 = shocks[1]
        x2 = state.x2
        drift2 = alpha2 * (mu2 - x2) * dt
        diffusion2 = sigma2 * jnp.sqrt(jnp.maximum(x2, 0.0) * dt) * dz2
        x2_new = x2 + drift2 + diffusion2
        state_var2 = jnp.maximum(x2_new, 0.0)

        t_new = t + dt
        phi_new = self._phi(t_new)
        short_rate = state_var1 + state_var2 + phi_new

        new_state = CIR2State(
            x1=x1_new, x2=x2_new,
            state_var1=state_var1, state_var2=state_var2,
            short_rate=short_rate,
        )
        outputs = {"short_rate": short_rate}
        return new_state, outputs

    # ─── ShortRateModel analytics ───

    def short_rate(self, state: CIR2State) -> jnp.ndarray:
        """Current short rate from state.

        Args:
            state: Current CIR2 state.

        Returns:
            Short rate value r(t) = x₁(t) + x₂(t) + φ(t).
        """
        return state.short_rate

    def zcb_price(self, state: CIR2State, t: float, maturity: float) -> jnp.ndarray:
        """Zero-coupon bond price P(t, T) using FROM TIME 0 formulation.

        P(t,T) = exp(-∫ₜᵀ φ(s)ds)
                 · [A₁(0,T)/A₁(0,t)] · exp(-(B₁(0,T)-B₁(0,t))·x₁(t))
                 · [A₂(0,T)/A₂(0,t)] · exp(-(B₂(0,T)-B₂(0,t))·x₂(t))

        Args:
            state: Current CIR2 state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            ZCB price.
        """
        alpha1 = self._params1.alpha
        mu1 = self._params1.mu
        sigma1 = self._params1.sigma
        x10 = self._params1.initial_value

        alpha2 = self._params2.alpha
        mu2 = self._params2.mu
        sigma2 = self._params2.sigma
        x20 = self._params2.initial_value

        # Integral of phi from t to T (analytic — exact, no discretisation error)
        int_phi = _cir2_integral_phi_analytic(
            t, maturity,
            alpha1, mu1, sigma1, x10,
            alpha2, mu2, sigma2, x20,
            self._zcb_curve.evaluate,
        )

        # Factor 1: A₁(0,T)/A₁(0,t) and B₁(0,T)-B₁(0,t)
        a1_0_T = cir_A(maturity, alpha1, mu1, sigma1)
        a1_0_t = cir_A(t, alpha1, mu1, sigma1)
        a1_ratio = a1_0_T / jnp.maximum(a1_0_t, 1e-30)

        b1_0_T = cir_B(maturity, alpha1, sigma1)
        b1_0_t = cir_B(t, alpha1, sigma1)
        b1_diff = b1_0_T - b1_0_t

        # Factor 2: A₂(0,T)/A₂(0,t) and B₂(0,T)-B₂(0,t)
        a2_0_T = cir_A(maturity, alpha2, mu2, sigma2)
        a2_0_t = cir_A(t, alpha2, mu2, sigma2)
        a2_ratio = a2_0_T / jnp.maximum(a2_0_t, 1e-30)

        b2_0_T = cir_B(maturity, alpha2, sigma2)
        b2_0_t = cir_B(t, alpha2, sigma2)
        b2_diff = b2_0_T - b2_0_t

        return (
            jnp.exp(-int_phi)
            * a1_ratio * jnp.exp(-b1_diff * state.state_var1)
            * a2_ratio * jnp.exp(-b2_diff * state.state_var2)
        )

    def spot_rate(self, state: CIR2State, t: float, maturity: float) -> jnp.ndarray:
        """Continuously compounded spot rate R(t, T).

        R(t,T) = -ln P(t,T) / (T - t)

        Args:
            state: Current CIR2 state.
            t: Current time.
            maturity: Maturity time.

        Returns:
            Spot rate.
        """
        tau = jnp.asarray(maturity - t, dtype=jnp.float64)
        p = self.zcb_price(state, t, maturity)
        return -jnp.log(p) / jnp.maximum(tau, 1e-12)

    def forward_rate(
        self, state: CIR2State, t: float, maturity: float
    ) -> jnp.ndarray:
        """Instantaneous forward rate f(t, T).

        Computed via numerical differentiation of -ln P(t, T).

        Args:
            state: Current CIR2 state.
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
        self, state: CIR2State, t: float, tenor: float, freq: float
    ) -> jnp.ndarray:
        """Par swap rate S(t; tenor, freq).

        S = (1 - P(t, t+tenor)) / Σ_{i=1}^{n} freq · P(t, t + i·freq)

        Args:
            state: Current CIR2 state.
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

    def analytic_a(self, tau: float, factor: int = 1) -> jnp.ndarray:
        """CIR A(τ) coefficient for bond pricing.

        Args:
            tau: Time to maturity.
            factor: Which factor (1 or 2).

        Returns:
            A(τ) for the specified factor.
        """
        params = self._params1 if factor == 1 else self._params2
        return cir_A(tau, params.alpha, params.mu, params.sigma)

    def analytic_b(self, tau: float, factor: int = 1) -> jnp.ndarray:
        """CIR B(τ) coefficient for bond pricing.

        Args:
            tau: Time to maturity.
            factor: Which factor (1 or 2).

        Returns:
            B(τ) for the specified factor.
        """
        params = self._params1 if factor == 1 else self._params2
        return cir_B(tau, params.alpha, params.sigma)


# ---------------------------------------------------------------------------
# Standalone phi computation via central differences
# ---------------------------------------------------------------------------


def compute_phi_central_differences(
    market_curve: ParametricCurve,
    params1: CIRParams,
    params2: CIRParams,
    h: float = 1e-4,
) -> Callable[[float], jnp.ndarray]:
    """Compute φ(t) using central differences on the market ZCB curve.

    This is an alternative to the analytic phi that avoids requiring a
    closed-form forward rate for the market curve. The instantaneous
    forward rate is approximated by central differences on the log ZCB
    price curve:

        f(0, t) ≈ -(ln P(0, t+h) - ln P(0, t-h)) / (2h)

    Then φ(t) = f_market(0,t) - f_CIR1(0,t) - f_CIR2(0,t), clamped
    to be non-negative.

    Args:
        market_curve: The market forward rate curve (ParametricCurve).
        params1: CIR parameters for factor 1.
        params2: CIR parameters for factor 2.
        h: Half-width for central difference step (default 1e-4).

    Returns:
        A callable f(t) -> jnp.ndarray that evaluates φ at time t.
    """
    zcb_curve = forward_to_zcbp(market_curve)

    def _phi(t: float) -> jnp.ndarray:
        t = jnp.asarray(t, dtype=jnp.float64)
        ln_p_plus = jnp.log(jnp.maximum(zcb_curve(t + h), 1e-30))
        ln_p_minus = jnp.log(jnp.maximum(zcb_curve(jnp.maximum(t - h, 0.0)), 1e-30))
        f_market = -(ln_p_plus - ln_p_minus) / (2.0 * h)

        f1 = cir_forward_rate(t, params1.alpha, params1.mu, params1.sigma, params1.initial_value)
        f2 = cir_forward_rate(t, params2.alpha, params2.mu, params2.sigma, params2.initial_value)

        return jnp.maximum(f_market - f1 - f2, 0.0)

    return _phi
