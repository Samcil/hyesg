"""Factor equity derivation: clone-override pattern for equity variants.

Matches the C# ``GetFactorEquityParameters`` pattern where factor equities
(Income, Momentum, Quality, LowVolatility, etc.) derive their parametric
curves from a benchmark equity.  Each override is blended back to the
benchmark over a configurable horizon using a
:class:`~hyesg.math.curves.PolynomialBlendingCurve`.

Typical usage::

    from hyesg.config.factor_equity import (
        UK_FACTOR_OVERRIDES,
        EquityCurveSet,
        FactorType,
        derive_factor_equity_curves,
    )

    benchmark = EquityCurveSet(
        dy_mu=ConstantCurve(0.038),
        vol_mu=ConstantCurve(0.18),
        mpr=ConstantCurve(0.30),
        vol_x0=0.18,
    )
    income_curves = derive_factor_equity_curves(
        benchmark,
        UK_FACTOR_OVERRIDES[FactorType.INCOME],
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from hyesg.math.curves.blending import PolynomialBlendingCurve
from hyesg.math.curves.primitives import BlendedCurve, ConstantCurve
from hyesg.math.curves.protocol import ParametricCurve


class FactorType(StrEnum):
    """Equity factor types matching the C# ESG factor index universe.

    Each factor represents a systematic equity style that may have
    adjusted dividend yield, volatility, or market-price-of-risk
    parameters relative to the benchmark index.
    """

    SIZE = "size"
    SIZE_MID = "size_mid"
    VALUE = "value"
    INCOME = "income"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"


class FactorEquityOverrides(BaseModel):
    """Override specification for deriving factor equity curves.

    Each non-``None`` field triggers a
    :class:`~hyesg.math.curves.BlendedCurve` that transitions from the
    adjusted value at short horizons back to the benchmark at long horizons.
    The blending region defaults to years 5–7 with strength 2, matching the
    C# ``GetFactorEquityBlendingCurve()`` call.

    Attributes:
        dy_mu: Replacement constant for dividend-yield long-run mean.
            When set, a ``ConstantCurve(dy_mu)`` is blended to the
            benchmark DY Mu.
        vol_multiplier: Scaling factor for volatility Mu curve and X₀.
            For example, 0.8 reduces both by 20%.
        mpr_multiplier: Scaling factor for the market-price-of-risk curve.
            For example, 1.09823 adds a ~10% uplift.
        blend_start: Start of the blending region in years.
        blend_end: End of the blending region in years.
        blend_strength: Power parameter for the
            :class:`~hyesg.math.curves.PolynomialBlendingCurve`.
    """

    model_config = ConfigDict(frozen=True)

    dy_mu: float | None = None
    vol_multiplier: float | None = Field(default=None, gt=0)
    mpr_multiplier: float | None = Field(default=None, gt=0)

    blend_start: float = Field(default=5.0, ge=0)
    blend_end: float = Field(default=7.0, gt=0)
    blend_strength: float = Field(default=2.0, gt=0)


# ── Predefined factor overrides (C# Calibration.cs) ─────────────────

UK_FACTOR_OVERRIDES: dict[FactorType, FactorEquityOverrides] = {
    FactorType.SIZE: FactorEquityOverrides(),
    FactorType.SIZE_MID: FactorEquityOverrides(),
    FactorType.VALUE: FactorEquityOverrides(),
    FactorType.INCOME: FactorEquityOverrides(dy_mu=0.0611431),
    FactorType.MOMENTUM: FactorEquityOverrides(mpr_multiplier=1.09823),
    FactorType.QUALITY: FactorEquityOverrides(mpr_multiplier=1.09823),
    FactorType.LOW_VOLATILITY: FactorEquityOverrides(
        vol_multiplier=0.8,
        mpr_multiplier=1.15613,
    ),
}
"""UK factor equity overrides from C# ``Calibration.cs``."""

US_FACTOR_OVERRIDES: dict[FactorType, FactorEquityOverrides] = {
    FactorType.SIZE: FactorEquityOverrides(),
    FactorType.SIZE_MID: FactorEquityOverrides(),
    FactorType.VALUE: FactorEquityOverrides(),
    FactorType.INCOME: FactorEquityOverrides(dy_mu=0.03223972),
    FactorType.MOMENTUM: FactorEquityOverrides(),
    FactorType.QUALITY: FactorEquityOverrides(),
    FactorType.LOW_VOLATILITY: FactorEquityOverrides(
        vol_multiplier=0.7,
        mpr_multiplier=1.27583,
    ),
}
"""US factor equity overrides from C# ``Calibration.cs``."""


# ── Runtime curve containers ────────────────────────────────────────


@dataclass(frozen=True)
class EquityCurveSet:
    """Parametric curves defining an equity's stochastic parameters.

    This is the runtime object produced by
    :func:`derive_factor_equity_curves`.  It holds the curves that feed
    into the CIR dividend-yield process, CIR volatility process, and
    market-price-of-risk for a single equity index.

    Attributes:
        dy_mu: Dividend-yield long-run mean curve.
        vol_mu: Volatility long-run mean curve.
        mpr: Market-price-of-risk curve.
        vol_x0: Volatility initial value (scalar, not a curve).
    """

    dy_mu: ParametricCurve
    vol_mu: ParametricCurve
    mpr: ParametricCurve
    vol_x0: float


# ── Factory ─────────────────────────────────────────────────────────


def create_factor_equity_blending_curve(
    start: float = 5.0,
    end: float = 7.0,
    strength: float = 2.0,
) -> PolynomialBlendingCurve:
    """Create the standard factor-equity blending weight curve.

    Args:
        start: Year at which the factor override begins decaying.
        end: Year at which the factor override fully reverts to benchmark.
        strength: Power parameter sharpening the decay.

    Returns:
        A :class:`PolynomialBlendingCurve` going from 1 → 0.
    """
    return PolynomialBlendingCurve(start, end, strength)


def derive_factor_equity_curves(
    benchmark: EquityCurveSet,
    overrides: FactorEquityOverrides,
) -> EquityCurveSet:
    """Derive factor equity curves from a benchmark via clone-override.

    Implements the C# ``GetFactorEquityParameters`` pattern:

    1. Clone all benchmark curves.
    2. For each non-``None`` override, create an adjusted curve and blend
       it with the benchmark using a :class:`PolynomialBlendingCurve`.
    3. At short horizons the factor-adjusted value dominates; at long
       horizons the benchmark value is restored.

    Args:
        benchmark: Benchmark equity curve set to derive from.
        overrides: Factor-specific adjustments (any combination of
            ``dy_mu``, ``vol_multiplier``, ``mpr_multiplier``).

    Returns:
        New :class:`EquityCurveSet` with blended curves where overrides
        were specified, and cloned benchmark curves elsewhere.
    """
    blend = PolynomialBlendingCurve(
        overrides.blend_start,
        overrides.blend_end,
        overrides.blend_strength,
    )

    # Dividend-yield Mu: replace with constant, blend to benchmark
    if overrides.dy_mu is not None:
        adjusted_dy = ConstantCurve(overrides.dy_mu)
        dy_mu: ParametricCurve = BlendedCurve(
            adjusted_dy, benchmark.dy_mu, blend,
        )
    else:
        dy_mu = benchmark.dy_mu

    # Volatility Mu: scale curve, blend to benchmark; also scale X0
    if overrides.vol_multiplier is not None:
        adjusted_vol = benchmark.vol_mu * overrides.vol_multiplier
        vol_mu: ParametricCurve = BlendedCurve(
            adjusted_vol, benchmark.vol_mu, blend,
        )
        vol_x0 = benchmark.vol_x0 * overrides.vol_multiplier
    else:
        vol_mu = benchmark.vol_mu
        vol_x0 = benchmark.vol_x0

    # Market price of risk: scale curve, blend to benchmark
    if overrides.mpr_multiplier is not None:
        adjusted_mpr = benchmark.mpr * overrides.mpr_multiplier
        mpr: ParametricCurve = BlendedCurve(
            adjusted_mpr, benchmark.mpr, blend,
        )
    else:
        mpr = benchmark.mpr

    return EquityCurveSet(
        dy_mu=dy_mu,
        vol_mu=vol_mu,
        mpr=mpr,
        vol_x0=vol_x0,
    )
