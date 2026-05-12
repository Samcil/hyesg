"""G2++ analytic pricing formulas for inflation-linked instruments.

Pure JAX implementations of G2++ ZCB pricing, index-linked ZCB,
forward CPI, ZCIIS, and YYIIS rates.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from hyesg.math.gaussian_helpers import b_func


class G2PPAnalyticParams(NamedTuple):
    """Parameters for G2++ analytic pricing.

    Attributes:
        a1: Mean reversion speed for factor 1.
        a2: Mean reversion speed for factor 2.
        sigma1: Volatility for factor 1.
        sigma2: Volatility for factor 2.
        rho: Correlation between the two factors.
    """

    a1: float
    a2: float
    sigma1: float
    sigma2: float
    rho: float


def _variance_integral(
    t: float,
    T: float,
    params: G2PPAnalyticParams,
) -> Array:
    """Compute variance integral for G2++ ZCB pricing.

    V(t,T) = V1(t,T) + V2(t,T) + 2*rho*V12(t,T)

    where Vi uses the OU variance integral for factor i and V12
    is the cross-factor covariance integral.

    Args:
        t: Current time.
        T: Maturity time.
        params: G2++ model parameters.

    Returns:
        Variance integral value.
    """
    tau = T - t
    a1, a2 = params.a1, params.a2
    s1, s2 = params.sigma1, params.sigma2
    rho = params.rho

    B1 = b_func(a1, tau)
    B2 = b_func(a2, tau)

    # Var from factor 1: (sigma1^2 / a1^2) * (tau - B1 - 0.5*a1*B1^2)
    V1 = (s1**2 / a1**2) * (tau - B1 - 0.5 * a1 * B1**2)

    # Var from factor 2: (sigma2^2 / a2^2) * (tau - B2 - 0.5*a2*B2^2)
    V2 = (s2**2 / a2**2) * (tau - B2 - 0.5 * a2 * B2**2)

    # Cross term: 2*rho*sigma1*sigma2 / (a1*a2) * (tau - B1 - B2 + B12)
    # where B12 = (1 - exp(-(a1+a2)*tau)) / (a1+a2)
    B12 = b_func(a1 + a2, tau)
    V12 = (s1 * s2 / (a1 * a2)) * (tau - B1 - B2 + B12)

    return V1 + V2 + 2.0 * rho * V12


def g2pp_zcb_price(
    t: float,
    T: float,
    x1: Array,
    x2: Array,
    params: G2PPAnalyticParams,
    phi_t: float,
    phi_T: float,
) -> Array:
    """G2++ analytic zero-coupon bond price.

    P(t,T) = exp(A(t,T) - B1(t,T)*x1 - B2(t,T)*x2)

    where B_i = (1 - exp(-a_i*(T-t)))/a_i and A includes the
    phi adjustment and variance terms.

    Args:
        t: Current time.
        T: Maturity time.
        x1: Factor 1 state variable.
        x2: Factor 2 state variable.
        params: G2++ model parameters.
        phi_t: Shift function value at time t.
        phi_T: Shift function value at time T.

    Returns:
        Zero-coupon bond price P(t,T).
    """
    tau = T - t
    B1 = b_func(params.a1, tau)
    B2 = b_func(params.a2, tau)

    V = _variance_integral(t, T, params)
    A = phi_T - phi_t - 0.5 * V

    return jnp.exp(A - B1 * x1 - B2 * x2)


def il_zcb_price(
    t: float,
    T: float,
    nominal_x1: Array,
    nominal_x2: Array,
    real_x1: Array,
    real_x2: Array,
    nominal_params: G2PPAnalyticParams,
    real_params: G2PPAnalyticParams,
    nominal_phi_t: float,
    nominal_phi_T: float,
    real_phi_t: float,
    real_phi_T: float,
    inflation_index: Array,
) -> Array:
    """Index-linked zero-coupon bond price.

    P_IL(t,T) = P_real(t,T) * I(t)

    The real ZCB prices the real purchasing power, and multiplying
    by the current inflation index converts to nominal terms.

    Args:
        t: Current time.
        T: Maturity time.
        nominal_x1: Nominal factor 1 state.
        nominal_x2: Nominal factor 2 state.
        real_x1: Real factor 1 state.
        real_x2: Real factor 2 state.
        nominal_params: Nominal G2++ parameters.
        real_params: Real G2++ parameters.
        nominal_phi_t: Nominal shift at t.
        nominal_phi_T: Nominal shift at T.
        real_phi_t: Real shift at t.
        real_phi_T: Real shift at T.
        inflation_index: Current inflation index value I(t).

    Returns:
        Index-linked ZCB price.
    """
    real_zcb = g2pp_zcb_price(
        t, T, real_x1, real_x2, real_params, real_phi_t, real_phi_T
    )
    return real_zcb * inflation_index


def forward_cpi(
    inflation_index: Array,
    nominal_zcb: Array,
    real_zcb: Array,
) -> Array:
    """Forward CPI implied by nominal and real ZCB prices.

    E[CPI(T)|F_t] = CPI(t) * P_nom(t,T) / P_real(t,T)

    This is the breakeven inflation-adjusted forward price level.

    Args:
        inflation_index: Current CPI level I(t).
        nominal_zcb: Nominal zero-coupon bond price P_nom(t,T).
        real_zcb: Real zero-coupon bond price P_real(t,T).

    Returns:
        Forward CPI level at maturity T.
    """
    return inflation_index * nominal_zcb / real_zcb


def zciis_rate(
    t: float,
    T: float,
    nominal_zcb: Array,
    real_zcb: Array,
) -> Array:
    """Zero-coupon inflation-indexed swap rate.

    ZCIIS(t,T) = (P_real(t,T) / P_nom(t,T))^(1/(T-t)) - 1

    This is the annualised breakeven inflation rate implied
    by the ratio of real to nominal discount factors.

    Args:
        t: Current time.
        T: Maturity time.
        nominal_zcb: Nominal ZCB price P_nom(t,T).
        real_zcb: Real ZCB price P_real(t,T).

    Returns:
        Annualised ZCIIS rate.
    """
    tau = T - t
    return jnp.power(real_zcb / nominal_zcb, 1.0 / tau) - 1.0


def yyiis_rate(
    t: float,
    T: float,
    nominal_zcb: Array,
    real_zcb: Array,
    convexity_adj: Array,
) -> Array:
    """Year-on-year inflation-indexed swap rate with convexity adjustment.

    YYIIS(t,T) = ZCIIS(t,T) + convexity_adj

    The convexity adjustment accounts for the difference between
    the ZCIIS rate and the year-on-year rate due to the Jensen
    inequality when compounding.

    Args:
        t: Current time.
        T: Maturity time.
        nominal_zcb: Nominal ZCB price P_nom(t,T).
        real_zcb: Real ZCB price P_real(t,T).
        convexity_adj: Convexity adjustment term.

    Returns:
        Year-on-year inflation swap rate.
    """
    base_rate = zciis_rate(t, T, nominal_zcb, real_zcb)
    return base_rate + convexity_adj
