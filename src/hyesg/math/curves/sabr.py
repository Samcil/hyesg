"""SABR-related parametric curves using tanh-wrapped Nelson-Siegel.

Each curve wraps the C# exponential-polynomial Nelson-Siegel form:

    NS(x) = β₀ + (β₁ + β₂·x) · exp(-λ·x)

through tanh with curve-specific normalisation to match the C#
``Hymans.FinancialMaths`` SABR parameter curves exactly.
"""

from __future__ import annotations

import math

from hyesg.math.curves.protocol import ParametricCurve


def _nelson_siegel_exp_poly(
    x: float,
    beta0: float,
    beta1: float,
    beta2: float,
    lam: float,
) -> float:
    """Evaluate the C# exponential-polynomial Nelson-Siegel form.

    NS(x) = β₀ + (β₁ + β₂·x) · exp(-λ·x)

    This is the form used in ``NelsonSiegelParametricCurve.cs``,
    *not* the textbook Nelson-Siegel (which uses decay ratios).

    Args:
        x: Evaluation point (maturity in years).
        beta0: Long-term level.
        beta1: Short-term component.
        beta2: Slope component.
        lam: Exponential decay rate (lambda).

    Returns:
        Nelson-Siegel value at x.
    """
    return beta0 + (beta1 + beta2 * x) * math.exp(-lam * x)


class SabrAtmVolCurve(ParametricCurve):
    """SABR ATM volatility term structure.

    Matches C# ``SabrAtmVolCurve``:

        σ_ATM(T) = (tanh(NS(T)) + 1) / 2

    where NS is the exponential-polynomial Nelson-Siegel form.
    Output is bounded in (0, 1).

    Args:
        beta0: Long-term NS level.
        beta1: Short-term NS component.
        beta2: Slope NS component.
        lam: NS decay parameter (lambda).
    """

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        lam: float,
    ) -> None:
        self._beta0 = beta0
        self._beta1 = beta1
        self._beta2 = beta2
        self._lam = lam

    def evaluate(self, x: float) -> float:
        """Evaluate the SABR ATM vol at maturity x.

        Args:
            x: Time to maturity (years).

        Returns:
            ATM volatility at maturity x, in (0, 1).
        """
        ns = _nelson_siegel_exp_poly(
            max(x, 0.0), self._beta0, self._beta1, self._beta2, self._lam
        )
        return (math.tanh(ns) + 1.0) / 2.0

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (beta0, beta1, beta2, lam)."""
        return (self._beta0, self._beta1, self._beta2, self._lam)


class SabrNuCurve(ParametricCurve):
    """SABR vol-of-vol (ν) term structure.

    Matches C# ``SabrNuCurve``:

        ν(T) = max_nu · (tanh(NS(T)) + 1) / 2

    Output is bounded in (0, max_nu).

    Args:
        beta0: Long-term NS level.
        beta1: Short-term NS component.
        beta2: Slope NS component.
        lam: NS decay parameter (lambda).
        max_nu: Upper bound for vol-of-vol.
    """

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        lam: float,
        max_nu: float,
    ) -> None:
        self._beta0 = beta0
        self._beta1 = beta1
        self._beta2 = beta2
        self._lam = lam
        self._max_nu = max_nu

    def evaluate(self, x: float) -> float:
        """Evaluate the SABR vol-of-vol at maturity x.

        Args:
            x: Time to maturity (years).

        Returns:
            Vol-of-vol ν at maturity x, in (0, max_nu).
        """
        ns = _nelson_siegel_exp_poly(
            max(x, 0.0), self._beta0, self._beta1, self._beta2, self._lam
        )
        return self._max_nu * (math.tanh(ns) + 1.0) / 2.0

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (beta0, beta1, beta2, lam, max_nu)."""
        return (
            self._beta0,
            self._beta1,
            self._beta2,
            self._lam,
            self._max_nu,
        )


class SabrRhoCurve(ParametricCurve):
    """SABR correlation (ρ) term structure.

    Matches C# ``SabrRhoCurve``:

        ρ(T) = tanh(NS(T)) · rho_cap

    Output is bounded in (-rho_cap, rho_cap), keeping ρ away from ±1.

    Args:
        beta0: Long-term NS level.
        beta1: Short-term NS component.
        beta2: Slope NS component.
        lam: NS decay parameter (lambda).
        rho_cap: Maximum absolute correlation (default 0.95).
    """

    RHO_CAP_DEFAULT: float = 0.95

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        lam: float,
        rho_cap: float = RHO_CAP_DEFAULT,
    ) -> None:
        self._beta0 = beta0
        self._beta1 = beta1
        self._beta2 = beta2
        self._lam = lam
        self._rho_cap = rho_cap

    def evaluate(self, x: float) -> float:
        """Evaluate the SABR correlation at maturity x.

        Args:
            x: Time to maturity (years).

        Returns:
            Correlation ρ at maturity x, in (-rho_cap, rho_cap).
        """
        ns = _nelson_siegel_exp_poly(
            max(x, 0.0), self._beta0, self._beta1, self._beta2, self._lam
        )
        return math.tanh(ns) * self._rho_cap

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (beta0, beta1, beta2, lam, rho_cap)."""
        return (
            self._beta0,
            self._beta1,
            self._beta2,
            self._lam,
            self._rho_cap,
        )
