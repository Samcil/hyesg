"""Tests for the factor equity derivation system.

Tests cover:
- PolynomialBlendingCurve golden-master values at multiple strengths.
- FactorEquityOverrides validation and immutability.
- derive_factor_equity_curves clone-override semantics.
- All 7 UK/US factor types produce correct results.
"""

from __future__ import annotations

import pytest

from hyesg.config.factor_equity import (
    UK_FACTOR_OVERRIDES,
    US_FACTOR_OVERRIDES,
    EquityCurveSet,
    FactorEquityOverrides,
    FactorType,
    create_factor_equity_blending_curve,
    derive_factor_equity_curves,
)
from hyesg.math.curves import (
    BlendedCurve,
    ConstantCurve,
    PolynomialBlendingCurve,
)


# ── PolynomialBlendingCurve Tests ───────────────────────────────────


class TestPolynomialBlendingCurve:
    """Tests for the standalone blending weight curve."""

    def test_boundary_left(self) -> None:
        """Weight is 1.0 at and below start_point."""
        c = PolynomialBlendingCurve(5.0, 7.0, 2.0)
        assert c.evaluate(0.0) == 1.0
        assert c.evaluate(5.0) == 1.0
        assert c.evaluate(4.99) == 1.0

    def test_boundary_right(self) -> None:
        """Weight is 0.0 at and above end_point."""
        c = PolynomialBlendingCurve(5.0, 7.0, 2.0)
        assert c.evaluate(7.0) == 0.0
        assert c.evaluate(7.01) == 0.0
        assert c.evaluate(100.0) == 0.0

    def test_midpoint_strength_1(self) -> None:
        """At midpoint with strength=1, Hermite basis gives 0.5."""
        c = PolynomialBlendingCurve(0.0, 1.0, 1.0)
        assert c.evaluate(0.5) == pytest.approx(0.5)

    def test_midpoint_strength_2(self) -> None:
        """At midpoint with strength=2, weight = 0.5^2 = 0.25."""
        c = PolynomialBlendingCurve(0.0, 1.0, 2.0)
        assert c.evaluate(0.5) == pytest.approx(0.25)

    def test_golden_values_strength_1(self) -> None:
        """Golden-master values for S(t) at strength=1."""
        c = PolynomialBlendingCurve(0.0, 1.0, 1.0)
        # S(t) = 1 - 3t² + 2t³
        assert c.evaluate(0.0) == pytest.approx(1.0)
        assert c.evaluate(0.25) == pytest.approx(0.84375)
        assert c.evaluate(0.5) == pytest.approx(0.5)
        assert c.evaluate(0.75) == pytest.approx(0.15625)
        assert c.evaluate(1.0) == pytest.approx(0.0)

    def test_golden_values_strength_2(self) -> None:
        """Golden-master values for S(t)^2 at strength=2."""
        c = PolynomialBlendingCurve(0.0, 1.0, 2.0)
        assert c.evaluate(0.0) == pytest.approx(1.0)
        assert c.evaluate(0.25) == pytest.approx(0.84375**2)
        assert c.evaluate(0.5) == pytest.approx(0.25)
        assert c.evaluate(0.75) == pytest.approx(0.15625**2)
        assert c.evaluate(1.0) == pytest.approx(0.0)

    def test_golden_values_factor_equity_curve(self) -> None:
        """Golden values for the standard C# factor equity curve (5, 7, 2)."""
        c = PolynomialBlendingCurve(5.0, 7.0, 2.0)
        # t = (x - 5) / 2; spline = 1 - 3t² + 2t³; weight = spline^2
        assert c.evaluate(5.0) == pytest.approx(1.0)
        assert c.evaluate(5.5) == pytest.approx(0.84375**2)  # t=0.25
        assert c.evaluate(6.0) == pytest.approx(0.25)  # t=0.5
        assert c.evaluate(6.5) == pytest.approx(0.15625**2)  # t=0.75
        assert c.evaluate(7.0) == pytest.approx(0.0)

    def test_monotonically_decreasing(self) -> None:
        """Weight is monotonically non-increasing over the blend region."""
        c = PolynomialBlendingCurve(5.0, 7.0, 2.0)
        xs = [5.0 + i * 0.1 for i in range(21)]
        values = [c.evaluate(x) for x in xs]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_callable_syntax(self) -> None:
        """Curve can be called directly via __call__."""
        c = PolynomialBlendingCurve(0.0, 1.0, 1.0)
        assert c(0.5) == pytest.approx(0.5)

    def test_properties(self) -> None:
        """Properties expose constructor parameters."""
        c = PolynomialBlendingCurve(3.0, 8.0, 4.0)
        assert c.start_point == 3.0
        assert c.end_point == 8.0
        assert c.strength == 4.0

    def test_invalid_range(self) -> None:
        """Raises ValueError if end <= start."""
        with pytest.raises(ValueError, match="end_point must be greater"):
            PolynomialBlendingCurve(5.0, 5.0, 1.0)
        with pytest.raises(ValueError, match="end_point must be greater"):
            PolynomialBlendingCurve(7.0, 5.0, 1.0)

    def test_invalid_strength(self) -> None:
        """Raises ValueError if strength <= 0."""
        with pytest.raises(ValueError, match="strength must be positive"):
            PolynomialBlendingCurve(0.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="strength must be positive"):
            PolynomialBlendingCurve(0.0, 1.0, -1.0)

    def test_strength_4_faster_decay(self) -> None:
        """Higher strength means faster decay (lower midpoint weight)."""
        c1 = PolynomialBlendingCurve(0.0, 1.0, 1.0)
        c2 = PolynomialBlendingCurve(0.0, 1.0, 2.0)
        c4 = PolynomialBlendingCurve(0.0, 1.0, 4.0)
        # At midpoint, higher strength → lower weight
        assert c1.evaluate(0.5) > c2.evaluate(0.5) > c4.evaluate(0.5)


# ── FactorType Tests ────────────────────────────────────────────────


class TestFactorType:
    """Tests for the FactorType enum."""

    def test_all_seven_types(self) -> None:
        """All 7 factor types are defined."""
        assert len(FactorType) == 7

    def test_string_values(self) -> None:
        """Factor types have snake_case string values."""
        assert FactorType.SIZE == "size"
        assert FactorType.LOW_VOLATILITY == "low_volatility"
        assert FactorType.SIZE_MID == "size_mid"

    def test_uk_overrides_cover_all_types(self) -> None:
        """UK overrides dict has an entry for every factor type."""
        for ft in FactorType:
            assert ft in UK_FACTOR_OVERRIDES

    def test_us_overrides_cover_all_types(self) -> None:
        """US overrides dict has an entry for every factor type."""
        for ft in FactorType:
            assert ft in US_FACTOR_OVERRIDES


# ── FactorEquityOverrides Tests ─────────────────────────────────────


class TestFactorEquityOverrides:
    """Tests for the FactorEquityOverrides Pydantic model."""

    def test_default_is_pure_clone(self) -> None:
        """Default overrides produce a pure clone (no adjustments)."""
        o = FactorEquityOverrides()
        assert o.dy_mu is None
        assert o.vol_multiplier is None
        assert o.mpr_multiplier is None
        assert o.blend_start == 5.0
        assert o.blend_end == 7.0
        assert o.blend_strength == 2.0

    def test_frozen(self) -> None:
        """Overrides are immutable."""
        o = FactorEquityOverrides(mpr_multiplier=1.1)
        with pytest.raises(Exception):
            o.mpr_multiplier = 2.0  # type: ignore[misc]

    def test_uk_income_overrides(self) -> None:
        """UK Income has custom dy_mu override."""
        o = UK_FACTOR_OVERRIDES[FactorType.INCOME]
        assert o.dy_mu == pytest.approx(0.0611431)
        assert o.vol_multiplier is None
        assert o.mpr_multiplier is None

    def test_uk_low_vol_overrides(self) -> None:
        """UK LowVolatility has vol and MPR overrides."""
        o = UK_FACTOR_OVERRIDES[FactorType.LOW_VOLATILITY]
        assert o.vol_multiplier == pytest.approx(0.8)
        assert o.mpr_multiplier == pytest.approx(1.15613)
        assert o.dy_mu is None

    def test_us_low_vol_differs_from_uk(self) -> None:
        """US LowVolatility uses different multipliers than UK."""
        us = US_FACTOR_OVERRIDES[FactorType.LOW_VOLATILITY]
        uk = UK_FACTOR_OVERRIDES[FactorType.LOW_VOLATILITY]
        assert us.vol_multiplier == pytest.approx(0.7)
        assert us.mpr_multiplier == pytest.approx(1.27583)
        assert us.vol_multiplier != uk.vol_multiplier

    def test_invalid_vol_multiplier(self) -> None:
        """vol_multiplier must be positive."""
        with pytest.raises(Exception):
            FactorEquityOverrides(vol_multiplier=0.0)
        with pytest.raises(Exception):
            FactorEquityOverrides(vol_multiplier=-0.5)


# ── derive_factor_equity_curves Tests ───────────────────────────────


@pytest.fixture()
def benchmark() -> EquityCurveSet:
    """A representative benchmark equity curve set."""
    return EquityCurveSet(
        dy_mu=ConstantCurve(0.038),
        vol_mu=ConstantCurve(0.18),
        mpr=ConstantCurve(0.30),
        vol_x0=0.18,
    )


class TestDeriveFactorEquityCurves:
    """Tests for the factory function."""

    def test_pure_clone(self, benchmark: EquityCurveSet) -> None:
        """No overrides → identical curves returned."""
        result = derive_factor_equity_curves(
            benchmark, FactorEquityOverrides(),
        )
        # Should be the same curve objects (identity, not copies)
        assert result.dy_mu is benchmark.dy_mu
        assert result.vol_mu is benchmark.vol_mu
        assert result.mpr is benchmark.mpr
        assert result.vol_x0 == benchmark.vol_x0

    def test_dy_override_short_horizon(self, benchmark: EquityCurveSet) -> None:
        """DY override dominates at short horizons (before blend region)."""
        overrides = FactorEquityOverrides(dy_mu=0.06)
        result = derive_factor_equity_curves(benchmark, overrides)
        # At t=0, weight=1 → uses adjusted DY (0.06)
        assert result.dy_mu.evaluate(0.0) == pytest.approx(0.06)
        assert result.dy_mu.evaluate(4.0) == pytest.approx(0.06)

    def test_dy_override_long_horizon(self, benchmark: EquityCurveSet) -> None:
        """DY override reverts to benchmark at long horizons."""
        overrides = FactorEquityOverrides(dy_mu=0.06)
        result = derive_factor_equity_curves(benchmark, overrides)
        # At t=10, weight=0 → uses benchmark DY (0.038)
        assert result.dy_mu.evaluate(10.0) == pytest.approx(0.038)

    def test_dy_override_blended_at_midpoint(
        self, benchmark: EquityCurveSet,
    ) -> None:
        """DY at blend midpoint (t=6) is a weighted average."""
        overrides = FactorEquityOverrides(dy_mu=0.06)
        result = derive_factor_equity_curves(benchmark, overrides)
        # t=6 → blend_t = 0.5 → weight = 0.5^2 = 0.25
        # blended = 0.25 * 0.06 + 0.75 * 0.038 = 0.015 + 0.0285 = 0.0435
        assert result.dy_mu.evaluate(6.0) == pytest.approx(0.0435)

    def test_vol_multiplier(self, benchmark: EquityCurveSet) -> None:
        """Vol multiplier scales Mu and X0."""
        overrides = FactorEquityOverrides(vol_multiplier=0.8)
        result = derive_factor_equity_curves(benchmark, overrides)
        # X0 is directly scaled
        assert result.vol_x0 == pytest.approx(0.18 * 0.8)
        # At t=0 (weight=1): uses adjusted vol (0.8 * 0.18 = 0.144)
        assert result.vol_mu.evaluate(0.0) == pytest.approx(0.144)
        # At t=10 (weight=0): reverts to benchmark (0.18)
        assert result.vol_mu.evaluate(10.0) == pytest.approx(0.18)

    def test_mpr_multiplier(self, benchmark: EquityCurveSet) -> None:
        """MPR multiplier scales the curve and blends back."""
        overrides = FactorEquityOverrides(mpr_multiplier=1.1)
        result = derive_factor_equity_curves(benchmark, overrides)
        # At t=0 (weight=1): 1.1 * 0.30 = 0.33
        assert result.mpr.evaluate(0.0) == pytest.approx(0.33)
        # At t=10 (weight=0): reverts to 0.30
        assert result.mpr.evaluate(10.0) == pytest.approx(0.30)

    def test_combined_overrides(self, benchmark: EquityCurveSet) -> None:
        """Multiple overrides are applied independently."""
        overrides = FactorEquityOverrides(
            vol_multiplier=0.8, mpr_multiplier=1.15,
        )
        result = derive_factor_equity_curves(benchmark, overrides)
        # Both should be blended curves (not the benchmark originals)
        assert result.vol_mu is not benchmark.vol_mu
        assert result.mpr is not benchmark.mpr
        # DY should remain unchanged
        assert result.dy_mu is benchmark.dy_mu

    def test_all_uk_factors_produce_valid_curves(
        self, benchmark: EquityCurveSet,
    ) -> None:
        """All 7 UK factor types produce evaluable curve sets."""
        for factor_type, overrides in UK_FACTOR_OVERRIDES.items():
            result = derive_factor_equity_curves(benchmark, overrides)
            # All curves should be evaluable at arbitrary horizons
            for t in [0.0, 3.0, 6.0, 10.0, 50.0]:
                assert isinstance(result.dy_mu.evaluate(t), float)
                assert isinstance(result.vol_mu.evaluate(t), float)
                assert isinstance(result.mpr.evaluate(t), float)

    def test_all_us_factors_produce_valid_curves(
        self, benchmark: EquityCurveSet,
    ) -> None:
        """All 7 US factor types produce evaluable curve sets."""
        for factor_type, overrides in US_FACTOR_OVERRIDES.items():
            result = derive_factor_equity_curves(benchmark, overrides)
            for t in [0.0, 3.0, 6.0, 10.0, 50.0]:
                assert isinstance(result.dy_mu.evaluate(t), float)
                assert isinstance(result.vol_mu.evaluate(t), float)
                assert isinstance(result.mpr.evaluate(t), float)

    def test_custom_blend_region(self, benchmark: EquityCurveSet) -> None:
        """Custom blend_start/blend_end are respected."""
        overrides = FactorEquityOverrides(
            mpr_multiplier=1.5,
            blend_start=10.0,
            blend_end=15.0,
            blend_strength=1.0,
        )
        result = derive_factor_equity_curves(benchmark, overrides)
        # Factor dominates until year 10
        assert result.mpr.evaluate(9.0) == pytest.approx(0.45)
        # Benchmark restored after year 15
        assert result.mpr.evaluate(16.0) == pytest.approx(0.30)


# ── create_factor_equity_blending_curve Tests ───────────────────────


class TestCreateFactorEquityBlendingCurve:
    """Tests for the blending curve factory helper."""

    def test_default_parameters(self) -> None:
        """Default produces (5, 7, 2) matching C#."""
        c = create_factor_equity_blending_curve()
        assert c.start_point == 5.0
        assert c.end_point == 7.0
        assert c.strength == 2.0

    def test_custom_parameters(self) -> None:
        """Custom parameters are forwarded correctly."""
        c = create_factor_equity_blending_curve(3.0, 10.0, 4.0)
        assert c.start_point == 3.0
        assert c.end_point == 10.0
        assert c.strength == 4.0
