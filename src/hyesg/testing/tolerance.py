"""Tolerance tiers for parity comparisons.

Defines four tolerance levels reflecting the expected agreement between
C# and Python ESG outputs depending on the type of computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToleranceTier(Enum):
    """Tolerance tiers for different comparison types.

    Attributes:
        EXACT: Config parsing, validation — exact match.
        ANALYTICAL: CIR A/B, ZCB pricing — atol=1e-12.
        MONTE_CARLO: Simulation outputs — statistical tests.
        DISTRIBUTIONAL: Full distribution — KS test p>0.01.
    """

    EXACT = "exact"
    ANALYTICAL = "analytical"
    MONTE_CARLO = "monte_carlo"
    DISTRIBUTIONAL = "distributional"


@dataclass(frozen=True)
class ToleranceConfig:
    """Configuration for a specific tolerance tier.

    Attributes:
        tier: The tolerance tier level.
        atol: Absolute tolerance for exact/analytical comparisons.
        rtol: Relative tolerance for exact/analytical comparisons.
        ks_significance: Significance level for KS test (reject if p < this).
        moment_rtol: Relative tolerance for moment comparisons.
        quantile_rtol: Relative tolerance for quantile comparisons.
    """

    tier: ToleranceTier
    atol: float = 0.0
    rtol: float = 0.0
    ks_significance: float = 0.01
    moment_rtol: float = 1e-3
    quantile_rtol: float = 1e-2


TIER_EXACT = ToleranceConfig(tier=ToleranceTier.EXACT, atol=0.0, rtol=0.0)
TIER_ANALYTICAL = ToleranceConfig(tier=ToleranceTier.ANALYTICAL, atol=1e-12)
TIER_MONTE_CARLO = ToleranceConfig(
    tier=ToleranceTier.MONTE_CARLO,
    ks_significance=0.01,
    moment_rtol=1e-3,
)
TIER_DISTRIBUTIONAL = ToleranceConfig(
    tier=ToleranceTier.DISTRIBUTIONAL,
    ks_significance=0.01,
    quantile_rtol=1e-2,
)
