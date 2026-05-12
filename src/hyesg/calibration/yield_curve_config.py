"""Pydantic configuration models for the yield curve pipeline.

Defines all configurable parameters for Akima yield curve construction,
long-end extension, RPI reform blending, and CPI breakeven derivation.
Matches the C# ``CalibrationRegime`` and ``Calibration`` configuration.
"""

from __future__ import annotations

import math
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LongEndExtensionConfig(BaseModel):
    """Configuration for long-end yield curve extension.

    The C# pipeline discards fitted Akima knots at maturities ≥ transition_start,
    then appends a single knot at transition_end with the long-term target rate.
    The Akima spline naturally creates a smooth graduated transition.

    Attributes:
        transition_start: Maturity (years) where extension begins (default 61y).
        transition_end: Maturity (years) for the long-term target knot (default 90y).
    """

    model_config = ConfigDict(frozen=True)

    transition_start: float = Field(default=61.0, gt=0)
    transition_end: float = Field(default=90.0, gt=0)

    @model_validator(mode="after")
    def _validate_order(self) -> LongEndExtensionConfig:
        if self.transition_end <= self.transition_start:
            raise ValueError(
                f"transition_end ({self.transition_end}) must be greater "
                f"than transition_start ({self.transition_start})"
            )
        return self


class RpiReformConfig(BaseModel):
    """RPI reform transition parameters.

    Matches C# ``RpiReformTransitionParameters``. Controls how the
    breakeven RPI curve is adjusted around the Feb 2030 reform date.

    Attributes:
        effective_date: RPI reform effective date (default 2030-02-15).
        pre_reform_rpi_minus_cpi: Pre-reform RPI−CPI wedge as continuous rate.
        post_reform_rpi_minus_cpi: Post-reform RPI−CPI wedge as continuous rate.
        breakeven_transition_pre: Years before reform for breakeven transition.
        breakeven_transition_post: Years after reform for breakeven transition.
        breakeven_transition_strength: Polynomial blending strength for breakeven.
        realised_transition_post: Years after reform for realised inflation transition.
        realised_transition_strength: Polynomial blending strength for realised.
        adjustment_period_pre: Years before reform for adjustment sampling.
        adjustment_period_post: Years after reform for adjustment sampling.
        market_rpi_sampling_gap: Gap for market RPI sampling.
        market_rpi_sampling_period: Period for market RPI sampling.
    """

    model_config = ConfigDict(frozen=True)

    effective_date: date = Field(default_factory=lambda: date(2030, 2, 15))
    pre_reform_rpi_minus_cpi: float = Field(
        default_factory=lambda: math.log(1.01)
    )
    post_reform_rpi_minus_cpi: float = Field(
        default_factory=lambda: math.log(1.0)
    )
    breakeven_transition_pre: float = 2.0
    breakeven_transition_post: float = 2.0
    breakeven_transition_strength: float = 1.0
    realised_transition_post: float = Field(
        default_factory=lambda: 1.0 / 12.0
    )
    realised_transition_strength: float = 2.0
    adjustment_period_pre: float = 2.0
    adjustment_period_post: float = 5.0
    market_rpi_sampling_gap: float = 0.5
    market_rpi_sampling_period: float = 1.0

    def time_to_effective_date(self, simulation_date: date) -> float:
        """Compute years from simulation date to reform effective date.

        Uses complete-years-and-months convention matching C#
        ``DateDiffCompleteYearsAndMonths``.

        Args:
            simulation_date: The calibration/simulation date.

        Returns:
            Time in years to the reform effective date.
        """
        delta = self.effective_date - simulation_date
        return delta.days / 365.25


class YieldCurvePipelineConfig(BaseModel):
    """Master configuration for the yield curve calibration pipeline.

    Attributes:
        standard_knots: Standard Akima knot points in years.
        long_end: Long-end extension configuration.
        rpi_reform: RPI reform transition parameters.
        inflation_maturities_quarterly_end: End of quarterly grid (years).
        inflation_maturities_annual_start: Start of annual grid (years).
        inflation_maturities_annual_end: End of annual grid (years).
    """

    model_config = ConfigDict(frozen=True)

    standard_knots: tuple[float, ...] = (
        0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
    )
    long_end: LongEndExtensionConfig = Field(
        default_factory=LongEndExtensionConfig
    )
    rpi_reform: RpiReformConfig = Field(default_factory=RpiReformConfig)
    inflation_maturities_quarterly_end: float = 10.0
    inflation_maturities_annual_start: int = 11
    inflation_maturities_annual_end: int = 100

    def inflation_maturities(self) -> list[float]:
        """Build the inflation maturity grid.

        0 to 10y in 0.25y steps, then 11y to 100y in 1.0y steps.
        Matches C# ``Enumerable.Range(0, 41).Select(i => i * 0.25)
        .Concat(Enumerable.Range(11, 90))``.

        Returns:
            Sorted list of inflation maturities.
        """
        quarterly = [
            i * 0.25
            for i in range(int(self.inflation_maturities_quarterly_end / 0.25) + 1)
        ]
        annual = list(
            range(
                self.inflation_maturities_annual_start,
                self.inflation_maturities_annual_end + 1,
            )
        )
        return quarterly + [float(m) for m in annual]
