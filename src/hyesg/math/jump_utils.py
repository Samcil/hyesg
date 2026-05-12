"""Jump-diffusion utilities for equity and FX models.

Pure JAX functions for:
- Jump-adjusted initial volatility (removing jump variance from observed vol)
- Poisson inverse CDF (exact integer and continuous interpolation)
- E[sigma] Taylor approximation for CIR variance processes

All functions are JIT-compatible, vmappable, and use float64 precision.

C# reference:
    Calibration.cs  GetJumpAdjustedInitialVolatility (lines 1550-1561)
    PoissonDistribution.cs  InverseCdf
    PoissonDistributionContinuousApproximation.cs  InverseCdf
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def jump_adjusted_sigma(
    sigma: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
    floor: float = 0.01,
) -> float:
    """Remove jump variance contribution from total observed volatility.

    The total observed variance of a jump-diffusion is the sum of
    diffusion variance and jump variance.  This function backs out the
    pure-diffusion volatility by subtracting the jump component:

        sigma_adj = sqrt(max(sigma^2 - lambda * (mu_j^2 + sigma_j^2), floor^2))

    Args:
        sigma: Total (unadjusted) observed volatility.
        lambda_: Jump intensity (expected number of jumps per year).
        mu_j: Mean of the log-normal jump size distribution.
        sigma_j: Std dev of the log-normal jump size distribution.
        floor: Minimum adjusted volatility (prevents zero/negative variance).

    Returns:
        Jump-adjusted volatility, floored at *floor*.
    """
    unadjusted_variance = sigma * sigma
    jump_variance = lambda_ * (mu_j * mu_j + sigma_j * sigma_j)
    adjusted_variance = jnp.maximum(unadjusted_variance - jump_variance, floor * floor)
    return jnp.sqrt(adjusted_variance)


def poisson_inverse_cdf(u: float, lambda_: float) -> int:
    """Exact Poisson inverse CDF (smallest k such that CDF(k) >= u).

    Iteratively accumulates the Poisson PMF until the CDF exceeds *u*.
    Returns an integer number of jumps.

    Uses ``jax.lax.while_loop`` for JIT compatibility.

    Args:
        u: Uniform random sample in [0, 1).
        lambda_: Poisson rate parameter (must be >= 0).

    Returns:
        Smallest non-negative integer k with P(X <= k) >= u.
    """
    exp_neg_lambda = jnp.exp(-lambda_)

    # State: (k, cumulative_cdf, current_pmf_term)
    # current_pmf_term = exp(-lambda) * lambda^k / k!  (iteratively updated)
    init_state = (0, exp_neg_lambda, exp_neg_lambda)

    def cond_fn(state):
        k, cdf, _term = state
        return cdf <= u

    def body_fn(state):
        k, cdf, term = state
        k_next = k + 1
        # P(X=k+1) = P(X=k) * lambda / (k+1)
        term_next = term * lambda_ / k_next
        cdf_next = cdf + term_next
        return (k_next, cdf_next, term_next)

    k_final, _cdf, _term = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return k_final


def poisson_inverse_cdf_continuous(u: float, lambda_: float) -> float:
    """Continuous interpolation of the Poisson inverse CDF.

    Uses a second-order recurrence on the cumulative sum of squared
    Poisson PMF terms to produce a piecewise-linear interpolation
    between integer values.  This matches the C# implementation in
    ``PoissonDistributionContinuousApproximation.InverseCdf``.

    The interpolation satisfies:
        - At CDF boundaries it equals the integer value
        - Between boundaries it linearly interpolates

    Uses ``jax.lax.while_loop`` for JIT compatibility.

    Args:
        u: Uniform random sample in [0, 1).
        lambda_: Poisson rate parameter (must be >= 0).

    Returns:
        Continuously interpolated Poisson quantile (non-negative float).
    """
    exp_neg_lambda = jnp.exp(-lambda_)

    # The C# recurrence:
    #   xn_0 = exp(-lambda)^2
    #   accumulator_0 = 2 * exp(-lambda)
    #   xnm1_0 = -xn_0
    #   For n = 1, 2, ...:
    #     xnm2 = xnm1;  xnm1 = xn;  xn = xnm2 + accumulator
    #     if u < xn: return n - 1 + (u - xnm1) / (xn - xnm1)
    #     accumulator *= lambda / n

    xn_init = exp_neg_lambda * exp_neg_lambda
    accumulator_init = 2.0 * exp_neg_lambda
    xnm1_init = -xn_init

    # Check u < xn_init (the n=0 case)
    # State: (n, xnm1, xn, accumulator, found, result)
    init_state = (
        jnp.int32(1),
        xnm1_init,
        xn_init,
        accumulator_init,
        u < xn_init,  # found flag
        jnp.float64(0.0),  # result (0.0 if found at n=0)
    )

    def cond_fn(state):
        _n, _xnm1, _xn, _acc, found, _result = state
        return ~found

    def body_fn(state):
        n, xnm1, xn, accumulator, _found, _result = state
        xnm2 = xnm1
        xnm1_new = xn
        xn_new = xnm2 + accumulator

        # Linear interpolation within this interval
        interp = (n - 1) + (u - xnm1_new) / (xn_new - xnm1_new)
        found_new = u < xn_new

        accumulator_new = accumulator * lambda_ / n
        return (n + 1, xnm1_new, xn_new, accumulator_new, found_new, interp)

    _n, _xnm1, _xn, _acc, _found, result = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )
    return result


def expected_sigma_taylor(mu_v: float, sigma_v: float) -> float:
    """Second-order Taylor approximation for E[sqrt(V)] given V ~ CIR.

    When variance V follows a CIR process with long-run mean *mu_v*
    and volatility-of-variance *sigma_v*, the expected instantaneous
    volatility is:

        E[sigma] = E[sqrt(V)] ≈ sqrt(mu_v) - sigma_v^2 / (8 * mu_v^(3/2))

    This is the second-order Taylor expansion of sqrt(·) around mu_v.

    Args:
        mu_v: Long-run mean of the CIR variance process (must be > 0).
        sigma_v: Volatility of the variance process.

    Returns:
        Approximate expected instantaneous volatility.
    """
    sqrt_mu = jnp.sqrt(mu_v)
    correction = sigma_v * sigma_v / (8.0 * mu_v * sqrt_mu)
    return sqrt_mu - correction
