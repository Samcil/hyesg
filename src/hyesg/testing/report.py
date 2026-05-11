"""Parity report generation.

Runs a suite of comparisons between actual simulation output and a
golden master, then formats results as Markdown or plain dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from jax import Array

from hyesg.testing.comparison import (
    ComparisonResult,
    compare_distributions,
    compare_exact,
    compare_moments,
    compare_quantiles,
)
from hyesg.testing.golden_master import GoldenMaster
from hyesg.testing.tolerance import (
    TIER_MONTE_CARLO,
    ToleranceConfig,
    ToleranceTier,
)

if TYPE_CHECKING:
    from hyesg.engine.output import SimulationResult


def _native_types(d: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy/JAX scalars in *d* to Python native types."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.generic | np.ndarray):
            v = v.item()
        elif isinstance(v, Array):
            v = float(v)
        elif isinstance(v, bool):
            pass  # already native
        elif isinstance(v, int | float | str):
            pass
        out[k] = v
    return out


@dataclass
class ParityReport:
    """Comprehensive parity comparison report.

    Attributes:
        name: Report title / identifier.
        comparisons: Individual comparison results.
        overall_passed: Whether all comparisons passed.
        summary: Human-readable summary string.
    """

    name: str
    comparisons: list[ComparisonResult] = field(default_factory=list)
    overall_passed: bool = True
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise report to a plain dict.

        Returns:
            Dict suitable for JSON serialisation.
        """
        return {
            "name": self.name,
            "overall_passed": bool(self.overall_passed),
            "summary": self.summary,
            "n_comparisons": len(self.comparisons),
            "comparisons": [
                {
                    "test_name": c.test_name,
                    "passed": bool(c.passed),
                    "metric": float(c.metric),
                    "threshold": float(c.threshold),
                    "details": _native_types(c.details),
                }
                for c in self.comparisons
            ],
        }

    def to_markdown(self) -> str:
        """Render report as Markdown.

        Returns:
            Markdown string with summary table and details.
        """
        status = "✅ PASSED" if self.overall_passed else "❌ FAILED"
        n_passed = sum(1 for c in self.comparisons if c.passed)
        n_total = len(self.comparisons)

        lines = [
            f"# Parity Report: {self.name}",
            "",
            f"**Status:** {status}",
            f"**Passed:** {n_passed}/{n_total}",
            "",
            "## Results",
            "",
            "| Test | Passed | Metric | Threshold |",
            "|------|--------|--------|-----------|",
        ]

        for c in self.comparisons:
            icon = "✅" if c.passed else "❌"
            lines.append(
                f"| {c.test_name} | {icon} | {c.metric:.6g} | {c.threshold:.6g} |"
            )

        if self.summary:
            lines.extend(["", "## Summary", "", self.summary])

        lines.append("")
        return "\n".join(lines)


def _extract_outputs(
    source: SimulationResult | GoldenMaster,
) -> dict[str, dict[str, Array]]:
    """Extract outputs dict from either source type."""
    return source.outputs


def parity_report(
    actual: SimulationResult,
    expected: GoldenMaster | SimulationResult,
    tolerance: ToleranceConfig = TIER_MONTE_CARLO,
) -> ParityReport:
    """Run full parity comparison and generate a report.

    Iterates over all shared model/field pairs and applies the
    comparison strategy determined by the tolerance tier.

    Args:
        actual: Python simulation result.
        expected: Reference golden master or simulation result.
        tolerance: Tolerance configuration controlling comparison mode.

    Returns:
        ParityReport with individual and overall results.
    """
    actual_outputs = _extract_outputs(actual)
    expected_outputs = _extract_outputs(expected)

    comparisons: list[ComparisonResult] = []

    shared_models = sorted(set(actual_outputs) & set(expected_outputs))

    for model in shared_models:
        actual_fields = actual_outputs[model]
        expected_fields = expected_outputs[model]
        shared_fields = sorted(set(actual_fields) & set(expected_fields))

        for field_name in shared_fields:
            actual_arr = actual_fields[field_name]
            expected_arr = expected_fields[field_name]
            label = f"{model}.{field_name}"

            if tolerance.tier == ToleranceTier.EXACT:
                result = compare_exact(actual_arr, expected_arr, atol=0.0)
                result.test_name = f"exact:{label}"
                comparisons.append(result)

            elif tolerance.tier == ToleranceTier.ANALYTICAL:
                result = compare_exact(
                    actual_arr, expected_arr, atol=tolerance.atol
                )
                result.test_name = f"analytical:{label}"
                comparisons.append(result)

            elif tolerance.tier == ToleranceTier.MONTE_CARLO:
                moment_result = compare_moments(
                    actual_arr, expected_arr, rtol=tolerance.moment_rtol
                )
                moment_result.test_name = f"moments:{label}"
                comparisons.append(moment_result)

                ks_result = compare_distributions(
                    actual_arr,
                    expected_arr,
                    test="ks",
                    significance=tolerance.ks_significance,
                )
                ks_result.test_name = f"ks:{label}"
                comparisons.append(ks_result)

            elif tolerance.tier == ToleranceTier.DISTRIBUTIONAL:
                ks_result = compare_distributions(
                    actual_arr,
                    expected_arr,
                    test="ks",
                    significance=tolerance.ks_significance,
                )
                ks_result.test_name = f"ks:{label}"
                comparisons.append(ks_result)

                q_result = compare_quantiles(
                    actual_arr,
                    expected_arr,
                    rtol=tolerance.quantile_rtol,
                )
                q_result.test_name = f"quantiles:{label}"
                comparisons.append(q_result)

    overall_passed = all(c.passed for c in comparisons)
    n_passed = sum(1 for c in comparisons if c.passed)
    n_total = len(comparisons)
    failed = [c.test_name for c in comparisons if not c.passed]

    summary_parts = [f"{n_passed}/{n_total} tests passed."]
    if failed:
        summary_parts.append(f"Failed: {', '.join(failed)}")

    report_name = "parity_report"
    if isinstance(expected, GoldenMaster):
        report_name = f"parity:{expected.name}"

    return ParityReport(
        name=report_name,
        comparisons=comparisons,
        overall_passed=overall_passed,
        summary=" ".join(summary_parts),
    )
