"""Model-specific objective functions for calibration.

All objective functions return residual vectors suitable for
Levenberg-Marquardt optimisation. They are pure JAX and JIT-compatible.

Convention: ``params_arr`` packs model parameters into a flat array.
The functions unpack, compute model-implied quantities, and return
the difference from market-observed values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from hyesg.math.cir_formulas import cir_zcb_price
from hyesg.math.gaussian_helpers import b_func

if TYPE_CHECKING:
    from jax import Array


# ─── CIR Curve Objective ───


def cir_curve_objective(
    params_arr: Array,
    target_prices: Array,
    tenors: Array,
) -> Array:
    """Residuals for fitting CIR to market zero-coupon bond prices.

    Packs [alpha, mu, sigma] and computes CIR ZCB prices at each tenor.
    Uses x0 = mu as the initial state (equilibrium assumption).

    Args:
        params_arr: Array of [alpha, mu, sigma].
        target_prices: Market ZCB prices at each tenor.
        tenors: Maturities in years.

    Returns:
        Residual vector (model_prices - target_prices).
    """
    alpha = params_arr[0]
    mu = params_arr[1]
    sigma = params_arr[2]

    # Ensure positivity via softplus
    alpha_pos = jnp.log1p(jnp.exp(alpha))
    mu_pos = jnp.log1p(jnp.exp(mu))
    sigma_pos = jnp.log1p(jnp.exp(sigma))

    # Use mu as initial state (equilibrium)
    x0 = mu_pos
    model_prices = cir_zcb_price(tenors, x0, alpha_pos, mu_pos, sigma_pos)

    return model_prices - target_prices


def cir_curve_objective_direct(
    params_arr: Array,
    target_prices: Array,
    tenors: Array,
) -> Array:
    """Residuals for fitting CIR to ZCB prices (direct parameterisation).

    Uses raw [alpha, mu, sigma] without softplus transform.
    Caller must ensure parameters are valid (positive).

    Args:
        params_arr: Array of [alpha, mu, sigma] (all positive).
        target_prices: Market ZCB prices at each tenor.
        tenors: Maturities in years.

    Returns:
        Residual vector (model_prices - target_prices).
    """
    alpha = params_arr[0]
    mu = params_arr[1]
    sigma = params_arr[2]

    x0 = mu
    model_prices = cir_zcb_price(tenors, x0, alpha, mu, sigma)

    return model_prices - target_prices


# ─── OU / Vasicek Curve Objective ───


def ou_zcb_price(
    tenors: Array,
    x0: Array,
    alpha: Array,
    mu: Array,
    sigma: Array,
) -> Array:
    """OU/Vasicek ZCB price: P(0,T) = exp(-B(T)*x0 + V(T)/2 - mu*(T-B(T))).

    Actually: P(0,T) = exp(A(T) - B(T)*x0)
    where A(T) = (mu - sigma^2/(2*alpha^2))*(B(T) - T) - sigma^2*B(T)^2/(4*alpha)
    and B(T) = (1 - exp(-alpha*T))/alpha.

    Args:
        tenors: Maturities in years.
        x0: Initial short rate.
        alpha: Mean-reversion speed.
        mu: Long-run mean.
        sigma: Volatility.

    Returns:
        Array of ZCB prices.
    """
    B = b_func(alpha, tenors)
    # A(T) for Vasicek
    theta = mu - sigma**2 / (2.0 * alpha**2)
    A = theta * (B - tenors) - sigma**2 * B**2 / (4.0 * alpha)
    return jnp.exp(A - B * x0)


def ou_curve_objective(
    params_arr: Array,
    target_prices: Array,
    tenors: Array,
) -> Array:
    """Residuals for fitting OU/Vasicek to market ZCB prices.

    Packs [alpha, mu, sigma]. Uses x0 = mu (equilibrium).

    Args:
        params_arr: Array of [alpha, mu, sigma].
        target_prices: Market ZCB prices at each tenor.
        tenors: Maturities in years.

    Returns:
        Residual vector (model_prices - target_prices).
    """
    alpha = params_arr[0]
    mu = params_arr[1]
    sigma = params_arr[2]

    # Softplus for positivity on alpha and sigma
    alpha_pos = jnp.log1p(jnp.exp(alpha))
    sigma_pos = jnp.log1p(jnp.exp(sigma))

    x0 = mu
    model_prices = ou_zcb_price(tenors, x0, alpha_pos, mu, sigma_pos)

    return model_prices - target_prices


def ou_curve_objective_direct(
    params_arr: Array,
    target_prices: Array,
    tenors: Array,
) -> Array:
    """Residuals for fitting OU/Vasicek (direct params, no transform).

    Args:
        params_arr: Array of [alpha, mu, sigma] (alpha, sigma > 0).
        target_prices: Market ZCB prices at each tenor.
        tenors: Maturities in years.

    Returns:
        Residual vector.
    """
    alpha = params_arr[0]
    mu = params_arr[1]
    sigma = params_arr[2]

    x0 = mu
    model_prices = ou_zcb_price(tenors, x0, alpha, mu, sigma)

    return model_prices - target_prices


# ─── Credit Spread Objective ───


def credit_survival_probability(
    tenors: Array,
    alpha: Array,
    mu: Array,
    sigma: Array,
    lambda0: Array,
) -> Array:
    """CIR-based survival probability Q(0,T) = A(T)*exp(-B(T)*lambda0).

    Uses the same CIR ZCB formula but applied to default intensity.

    Args:
        tenors: Maturities in years.
        alpha: Mean-reversion speed of intensity.
        mu: Long-run mean intensity.
        sigma: Intensity volatility.
        lambda0: Initial default intensity.

    Returns:
        Array of survival probabilities.
    """
    return cir_zcb_price(tenors, lambda0, alpha, mu, sigma)


def credit_spread_from_survival(
    tenors: Array,
    survival_probs: Array,
    recovery_rate: Array,
) -> Array:
    """Approximate credit spread from survival probabilities.

    s(T) ≈ -(1-R) * ln(Q(T)) / T

    This is the hazard-rate based approximation assuming constant
    recovery rate and continuous premium payments.

    Args:
        tenors: Maturities in years.
        survival_probs: Survival probabilities Q(0,T).
        recovery_rate: Recovery rate (0 to 1).

    Returns:
        Array of credit spreads.
    """
    lgd = jnp.asarray(1.0) - recovery_rate
    # Avoid log(0) with clipping
    safe_surv = jnp.clip(survival_probs, 1e-30, 1.0)
    hazard_rate = -jnp.log(safe_surv) / jnp.maximum(tenors, jnp.asarray(1e-10))
    return lgd * hazard_rate


def credit_spread_objective(
    params_arr: Array,
    target_spreads: Array,
    tenors: Array,
    recovery_rate: float = 0.4,
) -> Array:
    """Residuals for fitting credit intensity to CDS spreads.

    Packs [alpha, mu, sigma, lambda0].

    Args:
        params_arr: Array of [alpha, mu, sigma, lambda0].
        target_spreads: Market CDS spreads at each tenor.
        tenors: Maturities in years.
        recovery_rate: Recovery rate assumption.

    Returns:
        Residual vector (model_spreads - target_spreads).
    """
    alpha = params_arr[0]
    mu = params_arr[1]
    sigma = params_arr[2]
    lambda0 = params_arr[3]

    # Softplus for positivity
    alpha_pos = jnp.log1p(jnp.exp(alpha))
    mu_pos = jnp.log1p(jnp.exp(mu))
    sigma_pos = jnp.log1p(jnp.exp(sigma))
    lambda0_pos = jnp.log1p(jnp.exp(lambda0))

    surv = credit_survival_probability(tenors, alpha_pos, mu_pos, sigma_pos, lambda0_pos)
    model_spreads = credit_spread_from_survival(
        tenors, surv, jnp.asarray(recovery_rate)
    )

    return model_spreads - target_spreads


def credit_spread_objective_direct(
    params_arr: Array,
    target_spreads: Array,
    tenors: Array,
    recovery_rate: float = 0.4,
) -> Array:
    """Residuals for credit calibration (direct params, no transform).

    Args:
        params_arr: Array of [alpha, mu, sigma, lambda0] (all > 0).
        target_spreads: Market CDS spreads.
        tenors: Maturities.
        recovery_rate: Recovery rate.

    Returns:
        Residual vector.
    """
    alpha = params_arr[0]
    mu = params_arr[1]
    sigma = params_arr[2]
    lambda0 = params_arr[3]

    surv = credit_survival_probability(tenors, alpha, mu, sigma, lambda0)
    model_spreads = credit_spread_from_survival(
        tenors, surv, jnp.asarray(recovery_rate)
    )

    return model_spreads - target_spreads
