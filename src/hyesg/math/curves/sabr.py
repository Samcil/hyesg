"""SABR-related parametric curves using tanh-wrapped Nelson-Siegel."""

from __future__ import annotations

import math

from hyesg.math.curves.protocol import ParametricCurve


class SabrAtmVolCurve(ParametricCurve):
    """SABR ATM volatility term structure.

    Uses Nelson-Siegel form wrapped through tanh to bound the output:

        raw(T) = β₀ + β₁·(1-exp(-λT))/(λT)
               + β₂·((1-exp(-λT))/(λT) - exp(-λT))

        σ_ATM(T) = α + β_scale · tanh(raw(T))

    This ensures the output stays in (α - β_scale, α + β_scale).

    At T=0, uses L'Hôpital: raw(0) = β₀ + β₁.

    Args:
        beta0: Long-term NS level.
        beta1: Short-term NS component.
        beta2: Medium-term NS hump component.
        lam: NS decay parameter (lambda).
        alpha: Vertical shift after tanh.
        beta_scale: Scale factor after tanh.
    """

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        lam: float,
        alpha: float = 0.0,
        beta_scale: float = 1.0,
    ) -> None:
        self._beta0 = beta0
        self._beta1 = beta1
        self._beta2 = beta2
        self._lam = lam
        self._alpha = alpha
        self._beta_scale = beta_scale

    def evaluate(self, x: float) -> float:
        """Evaluate the SABR ATM vol at maturity x.

        Args:
            x: Time to maturity (years).

        Returns:
            ATM volatility at maturity x.
        """
        if x <= 0:
            raw = self._beta0 + self._beta1
            return self._alpha + self._beta_scale * math.tanh(raw)
        lx = self._lam * x
        exp_lx = math.exp(-lx)
        ns1 = (1.0 - exp_lx) / lx
        ns2 = ns1 - exp_lx
        raw = self._beta0 + self._beta1 * ns1 + self._beta2 * ns2
        return self._alpha + self._beta_scale * math.tanh(raw)

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (beta0, beta1, beta2, lam, alpha, beta_scale)."""
        return (
            self._beta0,
            self._beta1,
            self._beta2,
            self._lam,
            self._alpha,
            self._beta_scale,
        )


class SabrNuCurve(ParametricCurve):
    """SABR vol-of-vol (ν) term structure.

    Same Nelson-Siegel + tanh form as SabrAtmVolCurve but without the
    additional alpha/beta_scale wrapping:

        ν(T) = tanh(β₀ + β₁·ns1(T) + β₂·ns2(T))

    This ensures ν ∈ (-1, 1). In practice ν > 0 is enforced by
    calibration bounds.

    Args:
        beta0: Long-term NS level.
        beta1: Short-term NS component.
        beta2: Medium-term NS hump component.
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
        """Evaluate the SABR vol-of-vol at maturity x.

        Args:
            x: Time to maturity (years).

        Returns:
            Vol-of-vol ν at maturity x.
        """
        if x <= 0:
            return math.tanh(self._beta0 + self._beta1)
        lx = self._lam * x
        exp_lx = math.exp(-lx)
        ns1 = (1.0 - exp_lx) / lx
        ns2 = ns1 - exp_lx
        return math.tanh(
            self._beta0 + self._beta1 * ns1 + self._beta2 * ns2
        )

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (beta0, beta1, beta2, lam)."""
        return (self._beta0, self._beta1, self._beta2, self._lam)


class SabrRhoCurve(ParametricCurve):
    """SABR correlation (ρ) term structure.

    Uses tanh(Nelson-Siegel) to ensure ρ ∈ (-1, 1):

        ρ(T) = tanh(β₀ + β₁·ns1(T) + β₂·ns2(T))

    Args:
        beta0: Long-term NS level.
        beta1: Short-term NS component.
        beta2: Medium-term NS hump component.
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
        """Evaluate the SABR correlation at maturity x.

        Args:
            x: Time to maturity (years).

        Returns:
            Correlation ρ at maturity x.
        """
        if x <= 0:
            return math.tanh(self._beta0 + self._beta1)
        lx = self._lam * x
        exp_lx = math.exp(-lx)
        ns1 = (1.0 - exp_lx) / lx
        ns2 = ns1 - exp_lx
        return math.tanh(
            self._beta0 + self._beta1 * ns1 + self._beta2 * ns2
        )

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return (beta0, beta1, beta2, lam)."""
        return (self._beta0, self._beta1, self._beta2, self._lam)
