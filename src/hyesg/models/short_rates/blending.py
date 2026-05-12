"""CIR2++ real-world blending for drift adjustment.

Implements the blending mechanism that transitions the short-rate drift
from its risk-neutral (Q-measure) dynamics to a real-world (P-measure)
target over a configurable time window.

The blended expected rate is:
    E^P[r(t)] = w(t) * E^Q[r(t)] + (1 - w(t)) * S*(t)

where w(t) ramps linearly from 0 to ``blend_strength`` over
[blend_start, blend_end], and S*(t) is a per-regime long-term target
curve.

The real-world OU parameters (alpha_RW, mu_RW) are solved from a 2×2
linear system that matches the derivative of the blended target.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from hyesg.math.curves.protocol import ParametricCurve


class BlendingConfig(NamedTuple):
    """Per-regime blending parameters.

    Attributes:
        blend_start: Time blending begins (years).
        blend_end: Time blending reaches full strength (years).
        blend_strength: Maximum weight, typically 1.0.
        target_curve: S*(t) long-term target curve.
    """

    blend_start: float
    blend_end: float
    blend_strength: float
    target_curve: ParametricCurve


def blending_weight(t: float, config: BlendingConfig) -> Array:
    """Compute blending weight w_S(t).

    The weight ramps linearly from 0 to ``blend_strength``:
        w_S = 0                                           for t < blend_start
        w_S = strength * (t - start) / (end - start)     for start <= t <= end
        w_S = strength                                    for t > end

    Args:
        t: Current time in years.
        config: Blending configuration for the current regime.

    Returns:
        Blending weight as a scalar JAX array.
    """
    fraction = jnp.clip(
        (t - config.blend_start) / (config.blend_end - config.blend_start + 1e-30),
        0.0,
        1.0,
    )
    return config.blend_strength * fraction


def blended_expected_rate(
    rn_expected: Array,
    target_rate: Array,
    weight: Array,
) -> Array:
    """Compute blended expected short rate under P-measure.

    E^P[r(t)] = w(t) * E^Q[r(t)] + (1 - w(t)) * S*(t)

    Args:
        rn_expected: Risk-neutral expected rate E^Q[r(t)].
        target_rate: Long-term target rate S*(t).
        weight: Blending weight w(t) in [0, 1].

    Returns:
        Blended expected rate.
    """
    return weight * rn_expected + (1.0 - weight) * target_rate


def solve_rw_params(
    expected_x1: Array,
    expected_x2: Array,
    target_deriv_1: Array,
    target_deriv_2: Array,
) -> tuple[Array, Array]:
    """Solve 2×2 linear system for real-world OU parameters.

    Given two time points with expected factor values and target
    derivatives, solve for alpha_RW and mu_RW:

        [E[x1], 1] [alpha_RW      ]   [d1]
        [E[x2], 1] [alpha_RW*mu_RW] = [d2]

    Args:
        expected_x1: Expected factor value at time point 1.
        expected_x2: Expected factor value at time point 2.
        target_deriv_1: Target derivative at time point 1.
        target_deriv_2: Target derivative at time point 2.

    Returns:
        Tuple of (alpha_RW, mu_RW).
    """
    det = expected_x2 - expected_x1
    safe_det = jnp.where(jnp.abs(det) > 1e-30, det, 1e-30)
    alpha_rw = (target_deriv_1 - target_deriv_2) / safe_det
    alpha_mu_rw = target_deriv_1 + alpha_rw * expected_x1
    safe_alpha = jnp.where(jnp.abs(alpha_rw) > 1e-30, alpha_rw, 1e-30)
    mu_rw = alpha_mu_rw / safe_alpha
    return alpha_rw, mu_rw
