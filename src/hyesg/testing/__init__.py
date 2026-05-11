"""Parity testing framework for hyesg.

Provides tools for comparing Python ESG outputs against C# golden master
reference data, with tiered tolerance levels and statistical comparison
methods.

Example::

    from hyesg.testing import GoldenMaster, parity_report, TIER_MONTE_CARLO

    golden = GoldenMaster.load("reference_v3.2.npz")
    report = parity_report(my_result, golden, tolerance=TIER_MONTE_CARLO)
    print(report.to_markdown())
"""

from __future__ import annotations

from hyesg.testing.comparison import (
    ComparisonResult,
    compare_distributions,
    compare_exact,
    compare_moments,
    compare_quantiles,
)
from hyesg.testing.golden_master import GoldenMaster
from hyesg.testing.report import ParityReport, parity_report
from hyesg.testing.tolerance import (
    TIER_ANALYTICAL,
    TIER_DISTRIBUTIONAL,
    TIER_EXACT,
    TIER_MONTE_CARLO,
    ToleranceConfig,
    ToleranceTier,
)

__all__ = [
    "ComparisonResult",
    "GoldenMaster",
    "ParityReport",
    "TIER_ANALYTICAL",
    "TIER_DISTRIBUTIONAL",
    "TIER_EXACT",
    "TIER_MONTE_CARLO",
    "ToleranceConfig",
    "ToleranceTier",
    "compare_distributions",
    "compare_exact",
    "compare_moments",
    "compare_quantiles",
    "parity_report",
]
