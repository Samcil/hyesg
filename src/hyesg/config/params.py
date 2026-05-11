"""Model parameter schemas for hyesg.

Pydantic v2 BaseModel classes for all model parameter types.
These are the user-facing configuration objects — validated early,
then converted to frozen internal representations for the engine.
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CopulaType(StrEnum):
    """Copula type for shock correlation."""

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"


class RebalanceStrategy(StrEnum):
    """Portfolio rebalancing strategy."""

    FIXED = "fixed"
    MOMENTUM = "momentum"
    BUY_AND_HOLD = "buy_and_hold"
    TARGET_WEIGHT = "target_weight"


class RecoveryType(StrEnum):
    """Credit recovery assumption."""

    FACE = "face_value"
    MARKET = "market_value"
    TREASURY = "treasury"
    NONE = "no_recovery"


class CIRParams(BaseModel):
    """Parameters for a single-factor CIR process.

    The Feller condition (2*alpha*mu > sigma**2) ensures the process
    stays strictly positive. When violated, the process can hit zero.

    Attributes:
        alpha: Mean-reversion speed (must be > 0).
        mu: Long-run mean level (must be >= 0).
        sigma: Volatility (must be >= 0).
        initial_value: Starting value (must be >= 0).
        strict_feller: If True, raise on Feller violation.
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(gt=0)
    mu: float = Field(ge=0)
    sigma: float = Field(ge=0)
    initial_value: float = Field(ge=0)
    strict_feller: bool = False

    @model_validator(mode="after")
    def _check_feller(self) -> CIRParams:
        """Check the Feller condition: 2*alpha*mu > sigma**2."""
        feller_lhs = 2.0 * self.alpha * self.mu
        feller_rhs = self.sigma**2
        if feller_lhs <= feller_rhs:
            msg = (
                f"Feller condition violated: "
                f"2*alpha*mu={feller_lhs:.6f} <= "
                f"sigma^2={feller_rhs:.6f}. "
                f"Process may hit zero."
            )
            if self.strict_feller:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
        return self


class OUParams(BaseModel):
    """Parameters for an Ornstein-Uhlenbeck process.

    For G1++/G2++ sub-factors, mu must be zero (the shift function
    handles the mean level).

    Attributes:
        alpha: Mean-reversion speed (must be > 0).
        mu: Long-run mean level.
        sigma: Volatility (must be >= 0).
        initial_value: Starting value.
        model_type: One of "vasicek", "g1pp", "g2pp".
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(gt=0)
    mu: float = 0.0
    sigma: float = Field(ge=0)
    initial_value: float = 0.0
    model_type: Literal["vasicek", "g1pp", "g2pp"] = "vasicek"

    @model_validator(mode="after")
    def _enforce_zero_mu(self) -> OUParams:
        """Enforce mu=0 for G1++/G2++ sub-factors."""
        if self.model_type in ("g1pp", "g2pp") and self.mu != 0.0:
            raise ValueError(
                f"mu must be 0 for {self.model_type} "
                f"(shift function handles mean level), "
                f"got mu={self.mu}"
            )
        return self


class GBMParams(BaseModel):
    """Parameters for geometric Brownian motion.

    Attributes:
        sigma: Volatility (must be >= 0).
        initial_value: Starting value (must be > 0).
    """

    model_config = ConfigDict(frozen=True)

    sigma: float = Field(ge=0)
    initial_value: float = Field(gt=0)


class SeasonalityParams(BaseModel):
    """Fourier seasonality coefficients (2 harmonics, 4 coefficients).

    The seasonal adjustment is:
        0.01 * (a1*cos(2π*s) + a2*cos(4π*s) + b1*sin(2π*s) + b2*sin(4π*s))

    where s is the fractional-year shift.

    Attributes:
        a1: First cosine harmonic coefficient.
        a2: Second cosine harmonic coefficient.
        b1: First sine harmonic coefficient.
        b2: Second sine harmonic coefficient.
    """

    model_config = ConfigDict(frozen=True)

    a1: float = 0.0
    a2: float = 0.0
    b1: float = 0.0
    b2: float = 0.0


class PhiConfig(BaseModel):
    """Configuration for the phi/psi shift function.

    Attributes:
        source: How to obtain the shift function.
        curve_params: Extra params for calibrated curve source.
    """

    model_config = ConfigDict(frozen=True)

    source: Literal["analytic", "calibrated_curve"] = "analytic"
    curve_params: dict[str, float] | None = None


class G2PPParams(BaseModel):
    """Parameters for a two-factor G2++ model.

    Attributes:
        alpha1: Mean-reversion speed of factor 1.
        sigma1: Volatility of factor 1.
        alpha2: Mean-reversion speed of factor 2.
        sigma2: Volatility of factor 2.
        rho: Correlation between the two factors.
        x1_initial: Initial value of factor 1.
        x2_initial: Initial value of factor 2.
    """

    model_config = ConfigDict(frozen=True)

    alpha1: float = Field(gt=0)
    sigma1: float = Field(ge=0)
    alpha2: float = Field(gt=0)
    sigma2: float = Field(ge=0)
    rho: float = Field(ge=-1, le=1)
    x1_initial: float = 0.0
    x2_initial: float = 0.0


class CreditParams(BaseModel):
    """Parameters for a credit default intensity model.

    The intensity follows CIR++ dynamics: λ(t) = y(t) + ψ(t)
    where y follows CIR with these parameters.

    Attributes:
        alpha: Mean-reversion speed for intensity CIR factor.
        mu: Long-run mean intensity.
        sigma: Intensity volatility.
        initial_intensity: Starting intensity value.
        recovery_rate: Recovery rate on default (0-1).
        recovery_type: Recovery assumption type.
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(gt=0)
    mu: float = Field(ge=0)
    sigma: float = Field(ge=0)
    initial_intensity: float = Field(ge=0, default=0.01)
    recovery_rate: float = Field(ge=0, le=1, default=0.4)
    recovery_type: str = "face_value"


class CIR2PPParams(BaseModel):
    """Parameters for a two-factor CIR2++ model.

    Each factor follows CIR dynamics independently.

    Attributes:
        factor1: CIR parameters for factor 1.
        factor2: CIR parameters for factor 2.
    """

    model_config = ConfigDict(frozen=True)

    factor1: CIRParams
    factor2: CIRParams
