"""Salary wedge model using G2++ (two-factor Gaussian) dynamics.

The salary rate evolves as:
    salary_rate(t) = phi(t) + x1(t) + x2(t)

where x1, x2 are mean-reverting OU factors:
    dx1 = -alpha1 * x1 * dt + sigma1 * dW1
    dx2 = -alpha2 * x2 * dt + sigma2 * dW2

and phi(t) is calibrated so E[salary_rate(t)] = target(t).

The salary index accumulates:
    S(t) = S(0) * exp(integral_0^t salary_rate(s) ds)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from hyesg.math.curves.protocol import ParametricCurve


class SalaryWedgeState(NamedTuple):
    """State for salary wedge G2++ model."""

    x1: Array  # short-term factor
    x2: Array  # long-term factor
    salary_rate: Array  # phi(t) + x1 + x2
    salary_index: Array  # exp(integral of salary_rate)


class SalaryWedgeParams(NamedTuple):
    """Parameters for salary wedge model.

    Attributes:
        alpha1: Mean reversion speed for factor 1.
        alpha2: Mean reversion speed for factor 2.
        sigma1: Volatility for factor 1.
        sigma2: Volatility for factor 2.
        rho: Correlation between factors.
        initial_x1: Initial value for factor 1.
        initial_x2: Initial value for factor 2.
    """

    alpha1: float
    alpha2: float
    sigma1: float
    sigma2: float
    rho: float
    initial_x1: float = 0.0
    initial_x2: float = 0.0


class SalaryWedgeModel:
    """Salary growth modelled as G2++ with two OU factors.

    salary_rate(t) = phi(t) + x1(t) + x2(t)

    dx1 = -alpha1 * x1 * dt + sigma1 * dW1
    dx2 = -alpha2 * x2 * dt + sigma2 * dW2

    where phi(t) is calibrated so E[salary_rate(t)] = target(t).

    The salary index accumulates:
        S(t) = S(0) * exp(integral_0^t salary_rate(s) ds)

    Args:
        params: Salary wedge model parameters.
        target_curve: Target salary growth rate curve.
    """

    def __init__(
        self,
        params: SalaryWedgeParams,
        target_curve: ParametricCurve,
    ) -> None:
        self.params = params
        self.target_curve = target_curve

    def phi(self, t: float) -> Array:
        """Calibrated shift: phi(t) = target(t) - E[x1(t)] - E[x2(t)].

        Since E[x_i(t)] = x_i(0) * exp(-alpha_i * t), and we typically
        start at x_i(0) = 0, phi(t) = target(t).

        Args:
            t: Time point.

        Returns:
            The deterministic shift phi(t).
        """
        return jnp.asarray(self.target_curve.evaluate(t), dtype=jnp.float64)

    def init_state(self) -> SalaryWedgeState:
        """Initialize salary wedge state.

        Returns:
            Initial state with x1, x2 at their initial values,
            salary_rate = phi(0) + x1_0 + x2_0, and salary_index = 1.0.
        """
        p = self.params
        initial_rate = self.phi(0.0) + p.initial_x1 + p.initial_x2
        return SalaryWedgeState(
            x1=jnp.asarray(p.initial_x1, dtype=jnp.float64),
            x2=jnp.asarray(p.initial_x2, dtype=jnp.float64),
            salary_rate=jnp.asarray(initial_rate, dtype=jnp.float64),
            salary_index=jnp.asarray(1.0, dtype=jnp.float64),
        )

    def step(
        self,
        state: SalaryWedgeState,
        t: float,
        dt: float,
        dw1: Array,
        dw2: Array,
    ) -> SalaryWedgeState:
        """Euler step for salary wedge.

        Args:
            state: Current state.
            t: Current time.
            dt: Time step.
            dw1: Standard normal Brownian increment for factor 1.
            dw2: Standard normal Brownian increment for factor 2.

        Returns:
            Updated state after one Euler step.
        """
        p = self.params
        sqrt_dt = jnp.sqrt(dt)

        # OU dynamics for each factor
        new_x1 = state.x1 - p.alpha1 * state.x1 * dt + p.sigma1 * sqrt_dt * dw1
        new_x2 = state.x2 - p.alpha2 * state.x2 * dt + p.sigma2 * sqrt_dt * dw2

        # Salary rate with phi shift
        new_rate = self.phi(t + dt) + new_x1 + new_x2

        # Accumulate salary index using left-point rule
        new_index = state.salary_index * jnp.exp(state.salary_rate * dt)

        return SalaryWedgeState(
            x1=new_x1,
            x2=new_x2,
            salary_rate=new_rate,
            salary_index=new_index,
        )

    def output(self, state: SalaryWedgeState) -> dict[str, Array]:
        """Extract model output from state.

        Args:
            state: Current salary wedge state.

        Returns:
            Dictionary with 'salary_rate' and 'salary_index' keys.
        """
        return {
            "salary_rate": state.salary_rate,
            "salary_index": state.salary_index,
        }
