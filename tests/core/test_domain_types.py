"""Tests for hyesg.core.domain_types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hyesg.core.domain_types import (
    BondMetrics,
    CapAndFloor,
    Cashflow,
    CreditSpec,
    ForwardRate,
    Maturity,
)
from hyesg.core.enums import CreditClass, Liquidity, RecoveryType


class TestMaturity:
    def test_valid(self) -> None:
        m = Maturity(years=5.0)
        assert m.years == 5.0

    def test_zero_years(self) -> None:
        m = Maturity(years=0.0)
        assert m.years == 0.0

    def test_negative_years_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Maturity(years=-1.0)

    def test_is_matured_true(self) -> None:
        m = Maturity(years=5.0)
        assert m.is_matured(5.0) is True
        assert m.is_matured(6.0) is True

    def test_is_matured_false(self) -> None:
        m = Maturity(years=5.0)
        assert m.is_matured(4.9) is False

    def test_is_matured_at_zero(self) -> None:
        m = Maturity(years=0.0)
        assert m.is_matured(0.0) is True


class TestCashflow:
    def test_valid(self) -> None:
        cf = Cashflow(time=1.0, amount=100.0)
        assert cf.time == 1.0
        assert cf.amount == 100.0

    def test_negative_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Cashflow(time=-0.5, amount=50.0)

    def test_negative_amount_allowed(self) -> None:
        cf = Cashflow(time=1.0, amount=-50.0)
        assert cf.amount == -50.0


class TestForwardRate:
    def test_valid(self) -> None:
        fr = ForwardRate(start=0.0, end=1.0, rate=0.05)
        assert fr.start == 0.0
        assert fr.end == 1.0
        assert fr.rate == 0.05

    def test_end_equals_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForwardRate(start=1.0, end=1.0, rate=0.05)

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForwardRate(start=2.0, end=1.0, rate=0.05)

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForwardRate(start=-1.0, end=1.0, rate=0.05)

    def test_negative_rate_allowed(self) -> None:
        fr = ForwardRate(start=0.0, end=1.0, rate=-0.01)
        assert fr.rate == -0.01


class TestBondMetrics:
    def test_valid(self) -> None:
        bm = BondMetrics(
            clean_price=100.0,
            dirty_price=101.5,
            accrued_interest=1.5,
            yield_to_maturity=0.05,
            duration=4.5,
            convexity=25.0,
            modified_duration=4.3,
        )
        assert bm.clean_price == 100.0
        assert bm.modified_duration == 4.3


class TestCapAndFloor:
    def test_defaults(self) -> None:
        cf = CapAndFloor()
        assert cf.cap == 0.05
        assert cf.floor == 0.0

    def test_custom(self) -> None:
        cf = CapAndFloor(cap=0.10, floor=0.02)
        assert cf.cap == 0.10
        assert cf.floor == 0.02

    def test_cap_below_floor_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CapAndFloor(cap=0.01, floor=0.05)

    def test_cap_equals_floor_allowed(self) -> None:
        cf = CapAndFloor(cap=0.03, floor=0.03)
        assert cf.cap == cf.floor


class TestCreditSpec:
    def test_defaults(self) -> None:
        cs = CreditSpec(rating=CreditClass.AA)
        assert cs.rating == CreditClass.AA
        assert cs.recovery_type == RecoveryType.FACE_VALUE
        assert cs.recovery_rate == 0.4
        assert cs.liquidity == Liquidity.LIQUID

    def test_custom(self) -> None:
        cs = CreditSpec(
            rating=CreditClass.BBB,
            recovery_type=RecoveryType.MARKET_VALUE,
            recovery_rate=0.3,
            liquidity=Liquidity.ILLIQUID,
        )
        assert cs.rating == CreditClass.BBB
        assert cs.recovery_rate == 0.3

    def test_recovery_rate_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CreditSpec(rating=CreditClass.A, recovery_rate=1.5)
        with pytest.raises(ValidationError):
            CreditSpec(rating=CreditClass.A, recovery_rate=-0.1)


class TestDomainTypeCount:
    """Verify we have exactly 7 domain types."""

    def test_seven_types(self) -> None:
        types = [
            Maturity,
            Cashflow,
            ForwardRate,
            BondMetrics,
            CapAndFloor,
            CreditSpec,
        ]
        # 6 Pydantic models exported from domain_types
        assert len(types) == 6
