"""Calibration parameter schemas for hyesg.

Pydantic v2 configuration models matching the C# MinorCalibrationParameters
structure. These define the complete specification for an ESG calibration run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegimeDefinition(BaseModel):
    """Single regime definition with trial allocation and parameter overrides.

    C# ESG uses 3 regimes: Strong (2500 trials), Moderate (1500), Weak (1000).
    Each regime can override model parameters.

    Attributes:
        name: Human-readable regime identifier.
        trials: Number of Monte Carlo trials for this regime.
        weight: Blending weight for regime-weighted outputs.
        overrides: Per-model parameter overrides keyed by parameter name.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    trials: int = Field(gt=0)
    weight: float = Field(ge=0, le=1, default=0.5)
    overrides: dict[str, Any] = Field(default_factory=dict)


class YieldCurveSpec(BaseModel):
    """Specification for a yield curve.

    Attributes:
        knots: Maturity points in years.
        spot_rates: Continuously compounded spot rates at each knot.
        extrapolation: Extrapolation method beyond the last knot.
    """

    model_config = ConfigDict(frozen=True)

    knots: tuple[float, ...] = (
        0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
    )
    spot_rates: tuple[float, ...]
    extrapolation: str = "flat"

    @model_validator(mode="after")
    def _validate_lengths(self) -> YieldCurveSpec:
        """Validate knots and spot_rates have the same length."""
        if len(self.knots) != len(self.spot_rates):
            raise ValueError(
                f"knots ({len(self.knots)}) and spot_rates "
                f"({len(self.spot_rates)}) must have the same length"
            )
        return self


class EquityCalibrationParams(BaseModel):
    """Per-equity calibration parameters.

    Attributes:
        dividend_yield: Annualised dividend yield.
        volatility: Annualised volatility.
        market_price_of_risk: Market price of risk for this equity.
    """

    model_config = ConfigDict(frozen=True)

    dividend_yield: float = Field(ge=0)
    volatility: float = Field(ge=0)
    market_price_of_risk: float = 0.0


class FXCalibrationParams(BaseModel):
    """Per-FX pair calibration parameters.

    Attributes:
        spot_rate: Current spot exchange rate.
        volatility: Annualised FX volatility.
    """

    model_config = ConfigDict(frozen=True)

    spot_rate: float = Field(gt=0)
    volatility: float = Field(ge=0)


class CreditCalibrationParams(BaseModel):
    """Per-credit class calibration parameters.

    Attributes:
        initial_intensity: Starting default intensity.
        alpha: Mean-reversion speed for CIR++ intensity.
        sigma: Intensity volatility.
        recovery_rate: Recovery rate on default (0 to 1).
    """

    model_config = ConfigDict(frozen=True)

    initial_intensity: float = Field(ge=0)
    alpha: float = Field(gt=0)
    sigma: float = Field(ge=0)
    recovery_rate: float = Field(ge=0, le=1, default=0.4)


class CIR2PPStructuralParams(BaseModel):
    """Structural parameters for the CIR2++ nominal rate model.

    Attributes:
        factor1_alpha: Mean-reversion speed of factor 1.
        factor1_sigma: Volatility of factor 1.
        factor2_alpha: Mean-reversion speed of factor 2.
        factor2_sigma: Volatility of factor 2.
        blending_alpha: Blending weight between factors.
    """

    model_config = ConfigDict(frozen=True)

    factor1_alpha: float = Field(gt=0)
    factor1_sigma: float = Field(ge=0)
    factor2_alpha: float = Field(gt=0)
    factor2_sigma: float = Field(ge=0)
    blending_alpha: float = Field(ge=0, le=1, default=0.5)


class G2PPStructuralParams(BaseModel):
    """Structural parameters for the G2++ real rate model.

    Attributes:
        alpha1: Mean-reversion speed of factor 1.
        sigma1: Volatility of factor 1.
        alpha2: Mean-reversion speed of factor 2.
        sigma2: Volatility of factor 2.
        rho: Correlation between the two factors.
    """

    model_config = ConfigDict(frozen=True)

    alpha1: float = Field(gt=0)
    sigma1: float = Field(ge=0)
    alpha2: float = Field(gt=0)
    sigma2: float = Field(ge=0)
    rho: float = Field(ge=-1, le=1)


class CorrelationSpec(BaseModel):
    """Specification for a correlation matrix.

    Attributes:
        labels: Asset/factor labels for rows and columns.
        matrix: Row-major correlation matrix entries.
    """

    model_config = ConfigDict(frozen=True)

    labels: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def _validate_matrix(self) -> CorrelationSpec:
        """Validate matrix is square, matches labels, and is symmetric."""
        n_labels = len(self.labels)
        n_rows = len(self.matrix)
        if n_rows != n_labels:
            raise ValueError(
                f"matrix has {n_rows} rows but {n_labels} labels"
            )
        for i, row in enumerate(self.matrix):
            if len(row) != n_labels:
                raise ValueError(
                    f"row {i} has {len(row)} entries but expected {n_labels}"
                )
        # Check symmetry
        for i in range(n_labels):
            for j in range(i + 1, n_labels):
                if abs(self.matrix[i][j] - self.matrix[j][i]) > 1e-10:
                    raise ValueError(
                        f"matrix is not symmetric: "
                        f"[{i}][{j}]={self.matrix[i][j]} != "
                        f"[{j}][{i}]={self.matrix[j][i]}"
                    )
        return self


class CalibrationParameters(BaseModel):
    """Top-level calibration parameters matching C# MinorCalibrationParameters.

    This is the comprehensive schema for configuring an ESG calibration run.

    Attributes:
        seed: Master RNG seed.
        inverse_dt: Time steps per year (e.g. 12 for monthly).
        horizon: Projection horizon in years.
        trials: Total number of Monte Carlo trials.
        regimes: Regime definitions with trial allocations.
        nominal_curves: Named nominal yield curve specifications.
        real_curves: Named real yield curve specifications.
        inflation_targets: Currency-keyed inflation target rates.
        equity_params: Named equity calibration parameters.
        fx_params: Named FX pair calibration parameters.
        credit_params: Named credit class calibration parameters.
        correlation_specs: Named correlation matrix specifications.
        cir2pp_structural: CIR2++ structural parameters (optional).
        g2pp_structural: G2++ structural parameters (optional).
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 27
    inverse_dt: int = Field(default=12, gt=0)
    horizon: int = Field(default=100, gt=0)
    trials: int = Field(default=5000, gt=0)

    regimes: tuple[RegimeDefinition, ...] = ()

    nominal_curves: dict[str, YieldCurveSpec] = Field(default_factory=dict)
    real_curves: dict[str, YieldCurveSpec] = Field(default_factory=dict)
    inflation_targets: dict[str, float] = Field(default_factory=dict)

    equity_params: dict[str, EquityCalibrationParams] = Field(
        default_factory=dict,
    )
    fx_params: dict[str, FXCalibrationParams] = Field(default_factory=dict)
    credit_params: dict[str, CreditCalibrationParams] = Field(
        default_factory=dict,
    )

    correlation_specs: dict[str, CorrelationSpec] = Field(default_factory=dict)

    cir2pp_structural: CIR2PPStructuralParams | None = None
    g2pp_structural: G2PPStructuralParams | None = None
