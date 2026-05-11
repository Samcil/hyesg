"""Parametric yield curve models."""

from __future__ import annotations

import math

from hyesg.math.curves.protocol import ParametricCurve


class NelsonSiegelCurve(ParametricCurve):
    """Nelson-Siegel yield curve model.

    f(t) = β₀ + β₁·(1-e^(-t/τ))/(t/τ)
         + β₂·((1-e^(-t/τ))/(t/τ) - e^(-t/τ))

    At t=0, uses L'Hôpital's rule: f(0) = β₀ + β₁.

    Args:
        beta0: Long-term level.
        beta1: Short-term component.
        beta2: Medium-term (hump) component.
        tau: Decay parameter (must be positive).
    """

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        tau: float,
    ) -> None:
        if tau <= 0:
            raise ValueError("tau must be positive")
        self._beta0 = beta0
        self._beta1 = beta1
        self._beta2 = beta2
        self._tau = tau

    def evaluate(self, x: float) -> float:
        """Evaluate the Nelson-Siegel curve at maturity x.

        Args:
            x: Time to maturity (years).

        Returns:
            Yield or forward rate at maturity x.
        """
        if abs(x) < 1e-12:
            return self._beta0 + self._beta1

        t_over_tau = x / self._tau
        exp_term = math.exp(-t_over_tau)
        factor1 = (1.0 - exp_term) / t_over_tau
        factor2 = factor1 - exp_term

        return self._beta0 + self._beta1 * factor1 + self._beta2 * factor2

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (beta0, beta1, beta2, tau)."""
        return (self._beta0, self._beta1, self._beta2, self._tau)

    def with_parameters(self, params: tuple[float, ...]) -> NelsonSiegelCurve:
        """Return new NelsonSiegelCurve with updated parameters.

        Args:
            params: Tuple of (beta0, beta1, beta2, tau).

        Returns:
            New NelsonSiegelCurve instance.
        """
        return NelsonSiegelCurve(*params[:4])


class GeneralizedLogistic(ParametricCurve):
    """Generalized logistic function.

    f(x) = L + (U - L) / (1 + exp(-k*(x-x0)))^(1/nu)

    Args:
        lower: Lower asymptote L.
        upper: Upper asymptote U.
        k: Growth rate.
        x0: Midpoint (inflection point).
        nu: Asymmetry parameter.
    """

    def __init__(
        self,
        lower: float,
        upper: float,
        k: float,
        x0: float,
        nu: float,
    ) -> None:
        if nu <= 0:
            raise ValueError("nu must be positive")
        self._lower = lower
        self._upper = upper
        self._k = k
        self._x0 = x0
        self._nu = nu

    def evaluate(self, x: float) -> float:
        """Evaluate the generalized logistic at x.

        Args:
            x: The input value.

        Returns:
            Logistic function value at x.
        """
        exp_val = math.exp(-self._k * (x - self._x0))
        denom = (1.0 + exp_val) ** (1.0 / self._nu)
        return self._lower + (self._upper - self._lower) / denom

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (L, U, k, x0, nu)."""
        return (
            self._lower,
            self._upper,
            self._k,
            self._x0,
            self._nu,
        )

    def with_parameters(self, params: tuple[float, ...]) -> GeneralizedLogistic:
        """Return new GeneralizedLogistic with updated parameters.

        Args:
            params: Tuple of (L, U, k, x0, nu).

        Returns:
            New GeneralizedLogistic instance.
        """
        return GeneralizedLogistic(*params[:5])
