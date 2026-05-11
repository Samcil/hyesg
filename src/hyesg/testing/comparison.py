"""Statistical comparison tools for parity testing.

Provides functions for comparing simulation outputs at different
levels of strictness — from exact numerical match to distributional
equivalence via Kolmogorov-Smirnov tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of a single comparison.

    Attributes:
        test_name: Descriptive name of the test performed.
        passed: Whether the comparison passed.
        metric: Primary metric value computed by the test.
        threshold: Threshold used for pass/fail decision.
        details: Additional diagnostic information.
    """

    test_name: str
    passed: bool
    metric: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


def compare_moments(
    actual: Array,
    expected: Array,
    rtol: float = 1e-3,
) -> ComparisonResult:
    """Compare mean, variance, skewness, and kurtosis.

    Flattens both arrays and computes the maximum relative error across
    the four central moments.

    Args:
        actual: Array of actual values.
        expected: Array of expected (reference) values.
        rtol: Relative tolerance for all moments.

    Returns:
        ComparisonResult with ``metric`` set to the maximum relative
        error across moments.
    """
    a = np.asarray(actual).ravel().astype(np.float64)
    e = np.asarray(expected).ravel().astype(np.float64)

    moment_names = ["mean", "variance", "skewness", "kurtosis"]
    actual_moments = [
        float(np.mean(a)),
        float(np.var(a)),
        float(stats.skew(a)),
        float(stats.kurtosis(a)),
    ]
    expected_moments = [
        float(np.mean(e)),
        float(np.var(e)),
        float(stats.skew(e)),
        float(stats.kurtosis(e)),
    ]

    rel_errors: dict[str, float] = {}
    zipped = zip(moment_names, actual_moments, expected_moments, strict=False)
    for name, act_m, exp_m in zipped:
        # Use max(|expected|, 1.0) so near-zero moments (e.g. excess
        # kurtosis of a normal) are compared by absolute error instead.
        denom = max(abs(exp_m), 1.0)
        rel_errors[name] = abs(act_m - exp_m) / denom

    max_error = max(rel_errors.values())

    return ComparisonResult(
        test_name="moment_comparison",
        passed=max_error <= rtol,
        metric=max_error,
        threshold=rtol,
        details={
            "actual_moments": dict(zip(moment_names, actual_moments, strict=False)),
            "expected_moments": dict(zip(moment_names, expected_moments, strict=False)),
            "relative_errors": rel_errors,
        },
    )


def compare_distributions(
    actual: Array,
    expected: Array,
    test: str = "ks",
    significance: float = 0.01,
) -> ComparisonResult:
    """Test distributional equivalence via KS or Anderson-Darling.

    Args:
        actual: Array of actual values.
        expected: Array of expected (reference) values.
        test: Statistical test to use — ``"ks"`` or ``"ad"``.
        significance: Significance level; fail if p-value < this.

    Returns:
        ComparisonResult with ``metric`` set to the test p-value.

    Raises:
        ValueError: If ``test`` is not ``"ks"`` or ``"ad"``.
    """
    a = np.asarray(actual).ravel().astype(np.float64)
    e = np.asarray(expected).ravel().astype(np.float64)

    if test == "ks":
        statistic, p_value = stats.ks_2samp(a, e)
        return ComparisonResult(
            test_name="ks_test",
            passed=p_value >= significance,
            metric=float(p_value),
            threshold=significance,
            details={"statistic": float(statistic), "p_value": float(p_value)},
        )

    if test == "ad":
        result = stats.anderson_ksamp([a, e])
        p_value = float(result.pvalue)
        return ComparisonResult(
            test_name="anderson_darling_test",
            passed=p_value >= significance,
            metric=p_value,
            threshold=significance,
            details={
                "statistic": float(result.statistic),
                "p_value": p_value,
            },
        )

    raise ValueError(f"Unknown test '{test}'. Use 'ks' or 'ad'.")


def compare_quantiles(
    actual: Array,
    expected: Array,
    quantiles: list[float] | None = None,
    rtol: float = 1e-2,
) -> ComparisonResult:
    """Compare empirical quantiles between two samples.

    Args:
        actual: Array of actual values.
        expected: Array of expected (reference) values.
        quantiles: Probability levels to compare. Defaults to
            ``[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]``.
        rtol: Relative tolerance for each quantile.

    Returns:
        ComparisonResult with ``metric`` set to the maximum relative
        error across quantiles.
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    a = np.asarray(actual).ravel().astype(np.float64)
    e = np.asarray(expected).ravel().astype(np.float64)

    actual_q = np.quantile(a, quantiles)
    expected_q = np.quantile(e, quantiles)

    rel_errors: dict[str, float] = {}
    for q, aq, eq in zip(quantiles, actual_q, expected_q, strict=False):
        # Use max(|expected|, 1.0) so quantiles near zero are compared
        # by absolute rather than relative error.
        denom = max(abs(float(eq)), 1.0)
        rel_errors[f"q{q:.2f}"] = abs(float(aq) - float(eq)) / denom

    max_error = max(rel_errors.values())

    return ComparisonResult(
        test_name="quantile_comparison",
        passed=max_error <= rtol,
        metric=max_error,
        threshold=rtol,
        details={
            "quantiles": quantiles,
            "actual_quantiles": [float(v) for v in actual_q],
            "expected_quantiles": [float(v) for v in expected_q],
            "relative_errors": rel_errors,
        },
    )


def compare_exact(
    actual: Array,
    expected: Array,
    atol: float = 1e-12,
) -> ComparisonResult:
    """Exact element-wise comparison within tolerance.

    Suitable for analytic results where outputs should be numerically
    identical up to floating-point precision.

    Args:
        actual: Array of actual values.
        expected: Array of expected (reference) values.
        atol: Absolute tolerance for each element.

    Returns:
        ComparisonResult with ``metric`` set to the maximum absolute
        difference.
    """
    diff = jnp.abs(jnp.asarray(actual) - jnp.asarray(expected))
    max_diff = float(jnp.max(diff))

    n_total = int(diff.size)
    n_failing = int(jnp.sum(diff > atol))

    return ComparisonResult(
        test_name="exact_comparison",
        passed=max_diff <= atol,
        metric=max_diff,
        threshold=atol,
        details={
            "max_abs_diff": max_diff,
            "mean_abs_diff": float(jnp.mean(diff)),
            "n_failing_elements": n_failing,
            "n_total_elements": n_total,
        },
    )
