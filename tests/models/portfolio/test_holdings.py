"""Tests for holding types."""

from __future__ import annotations

import pytest

from hyesg.models.portfolio.holdings import (
    BondHolding,
    CashHolding,
    CDSHolding,
    DFRNHolding,
    EquityHolding,
    ForwardHolding,
    FundHolding,
    SwapHolding,
)


class TestEquityHolding:
    """Tests for EquityHolding."""

    def test_construction(self) -> None:
        """Construct with all fields."""
        h = EquityHolding(asset_label="uk_equity", weight=0.6, initial_price=100.0)
        assert h.asset_label == "uk_equity"
        assert h.weight == 0.6
        assert h.initial_price == 100.0

    def test_default_price(self) -> None:
        """Default initial_price is 1.0."""
        h = EquityHolding(asset_label="us_equity", weight=0.4)
        assert h.initial_price == 1.0

    def test_immutability(self) -> None:
        """NamedTuple should not allow mutation."""
        h = EquityHolding(asset_label="eq", weight=0.5)
        with pytest.raises(AttributeError):
            h.weight = 0.6  # type: ignore[misc]


class TestBondHolding:
    """Tests for BondHolding."""

    def test_construction(self) -> None:
        """Construct with all fields."""
        h = BondHolding(
            asset_label="gilt_5y",
            weight=0.3,
            face=100.0,
            coupon=0.05,
            maturity=5.0,
            freq=2,
        )
        assert h.asset_label == "gilt_5y"
        assert h.freq == 2

    def test_default_freq(self) -> None:
        """Default frequency is semi-annual (2)."""
        h = BondHolding(
            asset_label="bond", weight=0.5, face=100.0, coupon=0.04, maturity=10.0
        )
        assert h.freq == 2


class TestCashHolding:
    """Tests for CashHolding."""

    def test_construction(self) -> None:
        """Construct with weight."""
        h = CashHolding(weight=0.1)
        assert h.weight == 0.1


class TestForwardHolding:
    """Tests for ForwardHolding."""

    def test_construction(self) -> None:
        """Construct with all fields."""
        h = ForwardHolding(asset_label="oil_fwd", weight=0.05, delivery_date=1.0)
        assert h.delivery_date == 1.0


class TestSwapHolding:
    """Tests for SwapHolding."""

    def test_construction(self) -> None:
        """Construct with all fields."""
        h = SwapHolding(fixed_rate=0.03, notional=1_000_000.0, maturity=10.0)
        assert h.fixed_rate == 0.03


class TestCDSHolding:
    """Tests for CDSHolding."""

    def test_construction(self) -> None:
        """Construct with all fields."""
        h = CDSHolding(
            reference_label="corp_a", spread=0.01, notional=500_000.0, maturity=5.0
        )
        assert h.reference_label == "corp_a"


class TestDFRNHolding:
    """Tests for DFRNHolding."""

    def test_construction(self) -> None:
        """Construct with all fields."""
        h = DFRNHolding(reference_label="corp_b", spread=0.005, notional=1_000_000.0)
        assert h.spread == 0.005


class TestFundHolding:
    """Tests for FundHolding."""

    def test_construction(self) -> None:
        """Construct with all fields."""
        h = FundHolding(fund_ref="balanced_fund", weight=1.0)
        assert h.fund_ref == "balanced_fund"


class TestAllHoldingTypes:
    """Cross-cutting tests for all holding types."""

    def test_all_are_named_tuples(self) -> None:
        """All holding types are NamedTuples (have _fields)."""
        types = [
            EquityHolding,
            BondHolding,
            CashHolding,
            ForwardHolding,
            SwapHolding,
            CDSHolding,
            DFRNHolding,
            FundHolding,
        ]
        for cls in types:
            assert hasattr(cls, "_fields"), f"{cls.__name__} is not a NamedTuple"
