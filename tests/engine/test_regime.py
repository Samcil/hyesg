"""Tests for regime system — proportional trial ordering."""

from __future__ import annotations

from collections import Counter

import pytest

from hyesg.engine.regime import (
    RegimeSpec,
    RegimeTrialMap,
    build_proportional_trial_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

THREE_REGIMES = [
    RegimeSpec(name="Strong", trials=2500, weight=0.50),
    RegimeSpec(name="Moderate", trials=1500, weight=0.30),
    RegimeSpec(name="Weak", trials=1000, weight=0.20),
]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBuildProportionalTrialMap:
    """Tests for build_proportional_trial_map."""

    def test_total_trials_matches_sum(self) -> None:
        """Total trials equals sum of regime trial counts."""
        result = build_proportional_trial_map(THREE_REGIMES)
        assert result.total_trials == 5000

    def test_trial_to_regime_length(self) -> None:
        """trial_to_regime has one entry per trial."""
        result = build_proportional_trial_map(THREE_REGIMES)
        assert len(result.trial_to_regime) == 5000

    def test_each_regime_gets_correct_count(self) -> None:
        """Each regime appears exactly the right number of times."""
        result = build_proportional_trial_map(THREE_REGIMES)
        counts = Counter(result.trial_to_regime)
        assert counts[0] == 2500  # Strong
        assert counts[1] == 1500  # Moderate
        assert counts[2] == 1000  # Weak

    def test_interleaved_not_blocked(self) -> None:
        """Trials are interleaved, not in contiguous blocks."""
        result = build_proportional_trial_map(THREE_REGIMES)
        # The first 10 trials should contain more than one regime
        first_10 = set(result.trial_to_regime[:10])
        assert len(first_10) > 1

    def test_proportional_ordering_three_regimes(self) -> None:
        """Strong regime appears roughly 50% in any window."""
        result = build_proportional_trial_map(THREE_REGIMES)
        # Check first 100 trials: regime 0 should appear ~50 times
        window = result.trial_to_regime[:100]
        count_strong = window.count(0)
        assert 40 <= count_strong <= 60

    def test_cumulative_counts(self) -> None:
        """Cumulative counts are correct."""
        result = build_proportional_trial_map(THREE_REGIMES)
        assert result.regime_trial_counts == (2500, 4000, 5000)


class TestSingleRegime:
    """Tests for single-regime edge case."""

    def test_single_regime_all_zero(self) -> None:
        """Single regime → all trials get regime 0."""
        regimes = [RegimeSpec(name="Only", trials=100, weight=1.0)]
        result = build_proportional_trial_map(regimes)
        assert all(r == 0 for r in result.trial_to_regime)
        assert result.total_trials == 100

    def test_single_regime_one_trial(self) -> None:
        """Edge case: 1 regime, 1 trial."""
        regimes = [RegimeSpec(name="Solo", trials=1, weight=1.0)]
        result = build_proportional_trial_map(regimes)
        assert result.trial_to_regime == (0,)
        assert result.total_trials == 1


class TestEqualRegimes:
    """Tests for equal-weight regimes."""

    def test_two_equal_alternating(self) -> None:
        """Two equal regimes → alternating pattern."""
        regimes = [
            RegimeSpec(name="A", trials=50, weight=0.5),
            RegimeSpec(name="B", trials=50, weight=0.5),
        ]
        result = build_proportional_trial_map(regimes)
        counts = Counter(result.trial_to_regime)
        assert counts[0] == 50
        assert counts[1] == 50
        # First two should be different regimes (alternating)
        assert result.trial_to_regime[0] != result.trial_to_regime[1]

    def test_three_equal_regimes(self) -> None:
        """Three equal regimes all get the same count."""
        regimes = [
            RegimeSpec(name="X", trials=30, weight=1 / 3),
            RegimeSpec(name="Y", trials=30, weight=1 / 3),
            RegimeSpec(name="Z", trials=30, weight=1 / 3),
        ]
        result = build_proportional_trial_map(regimes)
        counts = Counter(result.trial_to_regime)
        assert counts[0] == 30
        assert counts[1] == 30
        assert counts[2] == 30


class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_empty_raises(self) -> None:
        """Empty regimes list raises ValueError."""
        with pytest.raises(ValueError, match="At least one regime"):
            build_proportional_trial_map([])

    def test_zero_trials_raises(self) -> None:
        """Zero trial count raises ValueError."""
        regimes = [RegimeSpec(name="Bad", trials=0, weight=1.0)]
        with pytest.raises(ValueError, match="positive"):
            build_proportional_trial_map(regimes)

    def test_returns_named_tuple(self) -> None:
        """Result is a RegimeTrialMap NamedTuple."""
        regimes = [RegimeSpec(name="A", trials=10, weight=1.0)]
        result = build_proportional_trial_map(regimes)
        assert isinstance(result, RegimeTrialMap)
