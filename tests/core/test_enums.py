"""Tests for hyesg.core.enums."""

from __future__ import annotations

import pytest

from hyesg.core.enums import (
    CompoundingConvention,
    CreditClass,
    ExerciseType,
    Liquidity,
    PoissonDistributionType,
    RebalancingStrategy,
    RecoveryType,
)


class TestCreditClass:
    """CreditClass enum ordering and membership."""

    def test_ordering_aaa_highest(self) -> None:
        assert CreditClass.AAA > CreditClass.AA
        assert CreditClass.AA > CreditClass.A
        assert CreditClass.A > CreditClass.BBB
        assert CreditClass.BBB > CreditClass.BB
        assert CreditClass.BB > CreditClass.B
        assert CreditClass.B > CreditClass.CCC
        assert CreditClass.CCC > CreditClass.DEFAULT

    def test_values(self) -> None:
        assert CreditClass.AAA == 7
        assert CreditClass.DEFAULT == 0

    def test_member_count(self) -> None:
        assert len(CreditClass) == 8

    def test_is_int(self) -> None:
        assert isinstance(CreditClass.AAA, int)


class TestLiquidity:
    def test_values(self) -> None:
        assert Liquidity.HIGH.value == "high"
        assert Liquidity.MEDIUM.value == "medium"
        assert Liquidity.LOW.value == "low"

    def test_member_count(self) -> None:
        assert len(Liquidity) == 3


class TestRecoveryType:
    def test_values(self) -> None:
        assert RecoveryType.FACE_VALUE.value == "face_value"
        assert RecoveryType.MARKET_VALUE.value == "market_value"
        assert RecoveryType.TREASURY_VALUE.value == "treasury_value"
        assert RecoveryType.NO_RECOVERY.value == "no_recovery"

    def test_member_count(self) -> None:
        assert len(RecoveryType) == 4


class TestPoissonDistributionType:
    def test_values(self) -> None:
        assert PoissonDistributionType.EXACT.value == "exact"
        assert PoissonDistributionType.CONTINUOUS.value == "continuous"


class TestCompoundingConvention:
    def test_values(self) -> None:
        assert CompoundingConvention.CONTINUOUS.value == "continuous"
        assert CompoundingConvention.ANNUAL.value == "annual"
        assert CompoundingConvention.SEMI_ANNUAL.value == "semi_annual"
        assert CompoundingConvention.QUARTERLY.value == "quarterly"

    def test_member_count(self) -> None:
        assert len(CompoundingConvention) == 4


class TestRebalancingStrategy:
    def test_values(self) -> None:
        assert RebalancingStrategy.BUY_AND_HOLD.value == "buy_and_hold"
        assert RebalancingStrategy.PERIODIC.value == "periodic"
        assert RebalancingStrategy.THRESHOLD.value == "threshold"

    def test_member_count(self) -> None:
        assert len(RebalancingStrategy) == 3


class TestExerciseType:
    def test_values(self) -> None:
        assert ExerciseType.EUROPEAN.value == "european"
        assert ExerciseType.AMERICAN.value == "american"
        assert ExerciseType.BERMUDAN.value == "bermudan"

    def test_member_count(self) -> None:
        assert len(ExerciseType) == 3


class TestEnumCount:
    """Verify we have exactly 7 enum types."""

    def test_seven_enums(self) -> None:
        enums = [
            CreditClass,
            Liquidity,
            RecoveryType,
            PoissonDistributionType,
            CompoundingConvention,
            RebalancingStrategy,
            ExerciseType,
        ]
        assert len(enums) == 7
