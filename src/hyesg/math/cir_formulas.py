"""CIR process closed-form solutions.

All functions are pure JAX, vectorizable with vmap.
Uses float64 precision for financial accuracy.

The CIR process: dx = α(μ - x)dt + σ√x dZ

Key formulas:
    h = √(α² + 2σ²)
    B(τ) = 2(eʰᵗ - 1) / ((h + α)(eʰᵗ - 1) + 2h)
    A(τ) = [2h·exp((α+h)τ/2) / ((h+α)(eʰᵗ-1) + 2h)]^(2αμ/σ²)
    P(τ, x) = A(τ) · exp(-B(τ) · x)

Edge cases:
    σ → 0 (zero vol): B(τ) → (1-e^(-ατ))/α, A(τ) → exp(-μτ + μB(τ))
    h → 0: handle via Taylor expansion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float


def cir_h(alpha: float, sigma: float) -> Float[Array, ""]:
    """Compute h = √(α² + 2σ²).

    Args:
        alpha: Mean reversion speed.
        sigma: Volatility.

    Returns:
        The auxiliary parameter h.
    """
    return jnp.sqrt(alpha**2 + 2.0 * sigma**2)


def cir_B(
    tau: Float[Array, ...] | float,
    alpha: float,
    sigma: float,
) -> Float[Array, ...]:
    """CIR B(τ) coefficient for bond pricing.

    B(τ) = 2(eʰᵗ - 1) / ((h + α)(eʰᵗ - 1) + 2h)

    For σ→0: B(τ) = (1 - e^(-ατ)) / α

    Args:
        tau: Time to maturity T-t.
        alpha: Mean reversion speed.
        sigma: Volatility.

    Returns:
        B(τ) coefficient.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    h = cir_h(alpha, sigma)

    exp_ht = jnp.exp(h * tau)
    numerator = 2.0 * (exp_ht - 1.0)
    denominator = (h + alpha) * (exp_ht - 1.0) + 2.0 * h

    # Zero-vol limit: B(τ) = (1 - e^(-ατ)) / α
    zero_vol_B = jnp.where(
        alpha > 1e-12,
        (1.0 - jnp.exp(-alpha * tau)) / alpha,
        tau,
    )

    return jnp.where(sigma < 1e-7, zero_vol_B, numerator / denominator)


def cir_A(
    tau: Float[Array, ...] | float,
    alpha: float,
    mu: float,
    sigma: float,
) -> Float[Array, ...]:
    """CIR A(τ) coefficient for bond pricing.

    A(τ) = [2h·exp((α+h)τ/2) / ((h+α)(eʰᵗ-1) + 2h)]^(2αμ/σ²)

    For σ→0: A(τ) = exp(-μτ + μ·B(τ))

    Args:
        tau: Time to maturity T-t.
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.

    Returns:
        A(τ) coefficient.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    h = cir_h(alpha, sigma)

    exp_ht = jnp.exp(h * tau)
    denominator = (h + alpha) * (exp_ht - 1.0) + 2.0 * h
    base = 2.0 * h * jnp.exp((alpha + h) * tau / 2.0) / denominator
    exponent = 2.0 * alpha * mu / jnp.maximum(sigma**2, 1e-30)

    A_standard = jnp.power(base, exponent)

    # Zero-vol limit: A(τ) = exp(-μτ + μ·B(τ))
    B_val = cir_B(tau, alpha, sigma)
    A_zero_vol = jnp.exp(-mu * tau + mu * B_val)

    return jnp.where(sigma < 1e-7, A_zero_vol, A_standard)


def cir_zcb_price(
    tau: Float[Array, ...] | float,
    x: Float[Array, ...] | float,
    alpha: float,
    mu: float,
    sigma: float,
) -> Float[Array, ...]:
    """CIR zero-coupon bond price P(τ, x) = A(τ) · exp(-B(τ) · x).

    Args:
        tau: Time to maturity T-t.
        x: Current state variable.
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.

    Returns:
        Zero-coupon bond price.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    A = cir_A(tau, alpha, mu, sigma)
    B = cir_B(tau, alpha, sigma)
    return A * jnp.exp(-B * x)


def cir_forward_rate(
    tau: Float[Array, ...] | float,
    x: Float[Array, ...] | float,
    alpha: float,
    mu: float,
    sigma: float,
) -> Float[Array, ...]:
    """CIR instantaneous forward rate f(0,τ) = -d/dτ ln P(0,τ).

    Computed analytically: f(τ,x) = B'(τ)·x - d/dτ ln A(τ)

    where:
        γ = √(α² + 2σ²)
        denom = (α+γ)(e^{γτ}-1) + 2γ
        B'(τ) = [2γe^{γτ}·denom - 2(e^{γτ}-1)·(α+γ)γe^{γτ}] / denom²
        d/dτ ln A(τ) = (2αμ/σ²) · [(α+γ)/2 - (α+γ)γe^{γτ}/denom]

    At τ=0: B'(0)=1 and d/dτ ln A(0)=0, so f(0,x)=x. ✓

    Args:
        tau: Time to maturity T-t.
        x: Current state variable.
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.

    Returns:
        Instantaneous forward rate.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    gamma = cir_h(alpha, sigma)

    exp_gt = jnp.exp(gamma * tau)
    denom = (alpha + gamma) * (exp_gt - 1.0) + 2.0 * gamma

    # B'(τ) via quotient rule on B(τ) = 2(e^{γτ}-1)/denom
    denom_prime = (alpha + gamma) * gamma * exp_gt
    b_prime = (2.0 * gamma * exp_gt * denom - 2.0 * (exp_gt - 1.0) * denom_prime) / (
        denom * denom
    )

    # d/dτ ln A(τ) = (2αμ/σ²) · [(α+γ)/2 - (α+γ)γe^{γτ}/denom]
    feller = 2.0 * alpha * mu / jnp.maximum(sigma**2, 1e-30)
    d_ln_a = feller * ((alpha + gamma) / 2.0 - (alpha + gamma) * gamma * exp_gt / denom)

    # Zero-vol limit: f(τ,x) = μ + (x-μ)e^{-ατ} (Vasicek forward)
    zero_vol_fwd = mu + (x - mu) * jnp.exp(-alpha * tau)

    standard_fwd = b_prime * x - d_ln_a
    return jnp.where(sigma < 1e-7, zero_vol_fwd, standard_fwd)


def cir_expectation(
    tau: Float[Array, ...] | float,
    x: Float[Array, ...] | float,
    alpha: float,
    mu: float,
) -> Float[Array, ...]:
    """Expected value E[x(t+τ) | x(t) = x] = μ + (x - μ)e^(-ατ).

    Args:
        tau: Time horizon.
        x: Current state.
        alpha: Mean reversion speed.
        mu: Long-run mean.

    Returns:
        Conditional expectation of x at time t+τ.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    return mu + (x - mu) * jnp.exp(-alpha * tau)


def cir_variance(
    tau: Float[Array, ...] | float,
    x: Float[Array, ...] | float,
    alpha: float,
    mu: float,
    sigma: float,
) -> Float[Array, ...]:
    """Variance Var[x(t+τ) | x(t) = x].

    Var = x·σ²/α·(e^(-ατ) - e^(-2ατ)) + μσ²/(2α)·(1 - e^(-ατ))²

    Args:
        tau: Time horizon.
        x: Current state.
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.

    Returns:
        Conditional variance of x at time t+τ.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    exp_at = jnp.exp(-alpha * tau)
    term1 = x * sigma**2 / alpha * (exp_at - exp_at**2)
    term2 = mu * sigma**2 / (2.0 * alpha) * (1.0 - exp_at) ** 2
    return term1 + term2


def cir_bond_option(
    t: float,
    T: float,
    S: float,
    K: float,
    x: Float[Array, ...] | float,
    alpha: float,
    mu: float,
    sigma: float,
    is_call: bool,
) -> Float[Array, ...]:
    """CIR bond option price (placeholder for Jamshidian's formula).

    Price of a European option on a ZCB: option to buy/sell
    P(T,S) at price K at time T. Current time is t.

    Args:
        t: Current time.
        T: Option expiry time.
        S: Bond maturity time.
        K: Strike price.
        x: Current state variable.
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.
        is_call: True for call, False for put.

    Returns:
        Option price (placeholder returning zeros).
    """
    # Full implementation requires non-central chi-squared CDF
    raise NotImplementedError(
        "CIR bond option pricing (Jamshidian) not yet implemented"
    )


def cir_phi_from_curves(
    t: Float[Array, ...] | float,
    forward_curve_fn: Callable,
    alpha: float,
    mu: float,
    sigma: float,
    x0: float,
) -> Float[Array, ...]:
    """CIR++ phi shift: φ(t) = f_market(0,t) - f_CIR(0,t; x₀).

    The shift that makes the CIR++ model match the initial
    yield curve exactly.

    Args:
        t: Time point.
        forward_curve_fn: Market forward rate function f(t).
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.
        x0: Initial CIR state.

    Returns:
        Phi shift at time t.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    f_market = forward_curve_fn(t)
    f_cir = cir_forward_rate(t, x0, alpha, mu, sigma)
    return f_market - f_cir


def cir_integral_phi(
    t: float,
    T: float,
    alpha: float,
    mu: float,
    sigma: float,
    x0: float,
    forward_curve_fn: Callable,
) -> Float[Array, ...]:
    """Integral of phi shift for CIR++ bond pricing.

    ∫ₜᵀ φ(s)ds used in CIR++ ZCB pricing.

    Args:
        t: Start time.
        T: End time.
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.
        x0: Initial CIR state.
        forward_curve_fn: Market forward rate function f(s).

    Returns:
        ∫ₜᵀ φ(s)ds.
    """
    n_points = max(200, int(50 * (T - t)) + 1)
    s_values = jnp.linspace(t, T, n_points)

    phi_values = jax.vmap(
        lambda s: cir_phi_from_curves(s, forward_curve_fn, alpha, mu, sigma, x0)
    )(s_values)

    return jnp.trapezoid(phi_values, s_values)


def cir_integral_phi_analytic(
    t: float,
    T: float,
    alpha: float,
    mu: float,
    sigma: float,
    x0: float,
    market_zcb_fn: Callable,
) -> Float[Array, ...]:
    """Analytic integral of CIR++ phi shift for bond pricing.

    ∫ₜᵀ φ(s)ds = ln[P_mkt(0,t)/P_mkt(0,T)]
                 - ln[A(t)/A(T)] - [B(T)-B(t)]·x₀

    where A, B are the CIR affine bond price coefficients.

    This is exact (no discretisation error) and much faster than
    the numerical trapezoid in ``cir_integral_phi``.

    Args:
        t: Start time.
        T: End time.
        alpha: Mean reversion speed.
        mu: Long-run mean.
        sigma: Volatility.
        x0: Initial CIR state.
        market_zcb_fn: Market ZCB price function P(0, s).

    Returns:
        ∫ₜᵀ φ(s)ds.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)

    # Market term: ln[P(0,t)/P(0,T)]
    p_t = market_zcb_fn(t)
    p_T = market_zcb_fn(T)
    market_term = jnp.log(jnp.maximum(p_t, 1e-30)) - jnp.log(jnp.maximum(p_T, 1e-30))

    # CIR model term: ln[A(t)/A(T)] + [B(T)-B(t)]·x₀
    A_t = cir_A(t, alpha, mu, sigma)
    A_T = cir_A(T, alpha, mu, sigma)
    B_t = cir_B(t, alpha, sigma)
    B_T = cir_B(T, alpha, sigma)

    cir_term = (
        jnp.log(jnp.maximum(A_t, 1e-30)) - jnp.log(jnp.maximum(A_T, 1e-30))
        + (B_T - B_t) * x0
    )

    return market_term - cir_term


def check_cir_timestep_stability(
    alpha: float,
    sigma: float,
    mu: float,
    dt: float,
) -> bool:
    """Check that the Euler-Maruyama timestep is stable for a CIR process.

    The standard stability requirement for explicit Euler on the CIR
    drift is α·dt < 1 (for the mean-reversion not to overshoot).
    The Feller condition 2αμ ≥ σ² ensures non-negativity in continuous
    time; we also check that the discrete approximation does not
    violate this excessively.

    Args:
        alpha: Mean reversion speed.
        sigma: Volatility parameter.
        mu: Long-run mean.
        dt: Timestep size.

    Returns:
        True if the timestep is stable.
    """
    mean_reversion_ok = alpha * dt < 1.0
    feller_ratio = 2.0 * alpha * mu / max(sigma**2, 1e-30)
    return mean_reversion_ok and feller_ratio > 0.5
