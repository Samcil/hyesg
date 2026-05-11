"""Tests for tolerance tiers and configuration."""

from __future__ import annotations

import pytest

from hyesg.testing.tolerance import (
    TIER_ANALYTICAL,
    TIER_DISTRIBUTIONAL,
    TIER_EXACT,
    TIER_MONTE_CARLO,
    ToleranceConfig,
    ToleranceTier,
)


class TestToleranceTierEnum:
    """Tests for ToleranceTier enum."""

    def test_exact_value(self):
        assert ToleranceTier.EXACT.value == "exact"

    def test_analytical_value(self):
        assert ToleranceTier.ANALYTICAL.value == "analytical"

    def test_monte_carlo_value(self):
        assert ToleranceTier.MONTE_CARLO.value == "monte_carlo"

    def test_distributional_value(self):
        assert ToleranceTier.DISTRIBUTIONAL.value == "distributional"

    def test_four_members(self):
        assert len(ToleranceTier) == 4


class TestToleranceConfigDefaults:
    """Tests for ToleranceConfig default values."""

    def test_default_atol_zero(self):
        cfg = ToleranceConfig(tier=ToleranceTier.EXACT)
        assert cfg.atol == 0.0

    def test_default_rtol_zero(self):
        cfg = ToleranceConfig(tier=ToleranceTier.EXACT)
        assert cfg.rtol == 0.0

    def test_default_ks_significance(self):
        cfg = ToleranceConfig(tier=ToleranceTier.EXACT)
        assert cfg.ks_significance == 0.01

    def test_default_moment_rtol(self):
        cfg = ToleranceConfig(tier=ToleranceTier.EXACT)
        assert cfg.moment_rtol == 1e-3

    def test_default_quantile_rtol(self):
        cfg = ToleranceConfig(tier=ToleranceTier.EXACT)
        assert cfg.quantile_rtol == 1e-2

    def test_frozen(self):
        """ToleranceConfig is immutable."""
        cfg = ToleranceConfig(tier=ToleranceTier.EXACT)
        with pytest.raises(AttributeError):
            cfg.atol = 1.0  # type: ignore[misc]


class TestPreConfiguredTiers:
    """Tests for the four pre-configured tier constants."""

    def test_exact_tier(self):
        assert TIER_EXACT.tier == ToleranceTier.EXACT
        assert TIER_EXACT.atol == 0.0
        assert TIER_EXACT.rtol == 0.0

    def test_analytical_tier(self):
        assert TIER_ANALYTICAL.tier == ToleranceTier.ANALYTICAL
        assert TIER_ANALYTICAL.atol == 1e-12

    def test_monte_carlo_tier(self):
        assert TIER_MONTE_CARLO.tier == ToleranceTier.MONTE_CARLO
        assert TIER_MONTE_CARLO.ks_significance == 0.01
        assert TIER_MONTE_CARLO.moment_rtol == 1e-3

    def test_distributional_tier(self):
        assert TIER_DISTRIBUTIONAL.tier == ToleranceTier.DISTRIBUTIONAL
        assert TIER_DISTRIBUTIONAL.ks_significance == 0.01
        assert TIER_DISTRIBUTIONAL.quantile_rtol == 1e-2

    def test_custom_override(self):
        """Custom configs can override any default."""
        cfg = ToleranceConfig(
            tier=ToleranceTier.MONTE_CARLO,
            ks_significance=0.05,
            moment_rtol=0.01,
        )
        assert cfg.ks_significance == 0.05
        assert cfg.moment_rtol == 0.01
