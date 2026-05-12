"""Tests for fund definition models and FundBuilder."""

from __future__ import annotations

import pytest

from hyesg.config.fund_builder import FundBuilder
from hyesg.config.funds import (
    FundCategory,
    FundDefinition,
    FundRebalanceStrategy,
    HoldingSpec,
    NetOfFeesFund,
)


# ============================================================================
# HoldingSpec
# ============================================================================


class TestHoldingSpec:
    """Tests for HoldingSpec dataclass."""

    def test_create_holding(self) -> None:
        h = HoldingSpec(asset_name="UKEquity", weight=0.5, economy="GBP")
        assert h.asset_name == "UKEquity"
        assert h.weight == 0.5
        assert h.economy == "GBP"

    def test_default_economy(self) -> None:
        h = HoldingSpec(asset_name="Bond", weight=1.0)
        assert h.economy == "GBP"

    def test_frozen(self) -> None:
        h = HoldingSpec(asset_name="Equity", weight=0.3)
        with pytest.raises(Exception):
            h.weight = 0.5  # type: ignore[misc]


# ============================================================================
# FundDefinition
# ============================================================================


class TestFundDefinition:
    """Tests for FundDefinition model."""

    def test_create_simple_fund(self) -> None:
        fund = FundDefinition(
            name="Test Fund",
            category=FundCategory.WORLD_EQUITY,
            holdings=[HoldingSpec(asset_name="Equity", weight=1.0)],
        )
        assert fund.name == "Test Fund"
        assert fund.category == FundCategory.WORLD_EQUITY
        assert len(fund.holdings) == 1
        assert fund.total_weight == pytest.approx(1.0)

    def test_multi_holding_fund(self) -> None:
        fund = FundDefinition(
            name="Multi",
            category=FundCategory.DGF,
            holdings=[
                HoldingSpec(asset_name="Equity", weight=0.6),
                HoldingSpec(asset_name="Bonds", weight=0.4),
            ],
        )
        assert fund.total_weight == pytest.approx(1.0)

    def test_defaults(self) -> None:
        fund = FundDefinition(
            name="Defaults",
            category=FundCategory.ABS,
            holdings=[HoldingSpec(asset_name="ABS", weight=1.0)],
        )
        assert fund.rebalance == FundRebalanceStrategy.ANNUAL
        assert fund.fee_bps == 0.0
        assert fund.currency == "GBP"
        assert fund.hedge_ratio is None

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="weights sum to"):
            FundDefinition(
                name="Bad",
                category=FundCategory.WORLD_EQUITY,
                holdings=[HoldingSpec(asset_name="Equity", weight=0.5)],
            )

    def test_weights_over_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights sum to"):
            FundDefinition(
                name="Over",
                category=FundCategory.WORLD_EQUITY,
                holdings=[
                    HoldingSpec(asset_name="A", weight=0.6),
                    HoldingSpec(asset_name="B", weight=0.6),
                ],
            )

    def test_empty_holdings_rejected(self) -> None:
        with pytest.raises(Exception):
            FundDefinition(
                name="Empty",
                category=FundCategory.WORLD_EQUITY,
                holdings=[],
            )

    def test_hedge_ratio(self) -> None:
        fund = FundDefinition(
            name="Hedged",
            category=FundCategory.HIGH_YIELD,
            holdings=[HoldingSpec(asset_name="HY", weight=1.0)],
            hedge_ratio=1.0,
        )
        assert fund.hedge_ratio == 1.0

    def test_frozen(self) -> None:
        fund = FundDefinition(
            name="Frozen",
            category=FundCategory.ABS,
            holdings=[HoldingSpec(asset_name="ABS", weight=1.0)],
        )
        with pytest.raises(Exception):
            fund.name = "Changed"  # type: ignore[misc]

    def test_fee_bps(self) -> None:
        fund = FundDefinition(
            name="Fee Fund",
            category=FundCategory.PROPERTY,
            holdings=[HoldingSpec(asset_name="Prop", weight=1.0)],
            fee_bps=50.0,
        )
        assert fund.fee_bps == 50.0


# ============================================================================
# FundCategory
# ============================================================================


class TestFundCategory:
    """Tests for FundCategory enum."""

    def test_all_categories_exist(self) -> None:
        expected = {
            "WorldEquity",
            "DistressedDebt",
            "EMD",
            "InfrastructureDebt",
            "HighYield",
            "SeniorLoans",
            "ABS",
            "InsuranceLinked",
            "AbsoluteReturn",
            "DGF",
            "PrivateEquity",
            "DirectLending",
            "Property",
            "Parmenion",
            "NetOfFees",
        }
        actual = {c.value for c in FundCategory}
        assert actual == expected

    def test_category_count(self) -> None:
        assert len(FundCategory) == 15


# ============================================================================
# FundRebalanceStrategy
# ============================================================================


class TestFundRebalanceStrategy:
    """Tests for FundRebalanceStrategy enum."""

    def test_values(self) -> None:
        assert FundRebalanceStrategy.ANNUAL == "annual"
        assert FundRebalanceStrategy.QUARTERLY == "quarterly"
        assert FundRebalanceStrategy.MONTHLY == "monthly"
        assert FundRebalanceStrategy.NONE == "none"


# ============================================================================
# NetOfFeesFund
# ============================================================================


class TestNetOfFeesFund:
    """Tests for NetOfFeesFund model."""

    def test_create(self) -> None:
        nof = NetOfFeesFund(
            gross_fund="Test Gross",
            fee_bps=50.0,
            label="Test Gross Net 50bps",
        )
        assert nof.gross_fund == "Test Gross"
        assert nof.fee_bps == 50.0
        assert nof.label == "Test Gross Net 50bps"

    def test_frozen(self) -> None:
        nof = NetOfFeesFund(
            gross_fund="G", fee_bps=10.0, label="L"
        )
        with pytest.raises(Exception):
            nof.fee_bps = 20.0  # type: ignore[misc]


# ============================================================================
# FundBuilder
# ============================================================================


class TestFundBuilder:
    """Tests for the fluent FundBuilder API."""

    def test_simple_build(self) -> None:
        fund = (
            FundBuilder("Test")
            .category(FundCategory.WORLD_EQUITY)
            .add_holding("Equity", 1.0)
            .build()
        )
        assert fund.name == "Test"
        assert fund.category == FundCategory.WORLD_EQUITY
        assert fund.total_weight == pytest.approx(1.0)

    def test_add_equity(self) -> None:
        fund = (
            FundBuilder("EQ")
            .category(FundCategory.WORLD_EQUITY)
            .add_equity("WorldEq", 1.0)
            .build()
        )
        assert fund.holdings[0].asset_name == "WorldEq"

    def test_add_bond(self) -> None:
        fund = (
            FundBuilder("BD")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("HY", 1.0, "USD")
            .build()
        )
        assert fund.holdings[0].asset_name == "HY"
        assert fund.holdings[0].economy == "USD"

    def test_multi_holding_build(self) -> None:
        fund = (
            FundBuilder("Multi")
            .category(FundCategory.DGF)
            .add_equity("Eq", 0.6)
            .add_bond("Bond", 0.4)
            .build()
        )
        assert len(fund.holdings) == 2
        assert fund.total_weight == pytest.approx(1.0)

    def test_fee_bps(self) -> None:
        fund = (
            FundBuilder("Fee")
            .category(FundCategory.ABS)
            .add_holding("ABS", 1.0)
            .fee_bps(50.0)
            .build()
        )
        assert fund.fee_bps == 50.0

    def test_hedge(self) -> None:
        fund = (
            FundBuilder("H")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("HY", 1.0)
            .hedge(1.0)
            .build()
        )
        assert fund.hedge_ratio == 1.0

    def test_partial_hedge(self) -> None:
        fund = (
            FundBuilder("PH")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("HY", 1.0)
            .hedge(0.5)
            .build()
        )
        assert fund.hedge_ratio == 0.5

    def test_rebalance_quarterly(self) -> None:
        fund = (
            FundBuilder("Q")
            .category(FundCategory.ABS)
            .add_holding("ABS", 1.0)
            .rebalance_quarterly()
            .build()
        )
        assert fund.rebalance == FundRebalanceStrategy.QUARTERLY

    def test_rebalance_annually(self) -> None:
        fund = (
            FundBuilder("A")
            .category(FundCategory.ABS)
            .add_holding("ABS", 1.0)
            .rebalance_annually()
            .build()
        )
        assert fund.rebalance == FundRebalanceStrategy.ANNUAL

    def test_rebalance_monthly(self) -> None:
        fund = (
            FundBuilder("M")
            .category(FundCategory.ABS)
            .add_holding("ABS", 1.0)
            .rebalance_monthly()
            .build()
        )
        assert fund.rebalance == FundRebalanceStrategy.MONTHLY

    def test_no_rebalance(self) -> None:
        fund = (
            FundBuilder("NR")
            .category(FundCategory.PRIVATE_EQUITY)
            .add_holding("PE", 1.0)
            .no_rebalance()
            .build()
        )
        assert fund.rebalance == FundRebalanceStrategy.NONE

    def test_currency(self) -> None:
        fund = (
            FundBuilder("USD")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("HY", 1.0)
            .currency("USD")
            .build()
        )
        assert fund.currency == "USD"

    def test_missing_category_raises(self) -> None:
        with pytest.raises(ValueError, match="must have a category"):
            FundBuilder("NoCategory").add_holding("A", 1.0).build()

    def test_bad_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="weights sum to"):
            (
                FundBuilder("BadWeights")
                .category(FundCategory.WORLD_EQUITY)
                .add_holding("A", 0.5)
                .build()
            )

    def test_chaining_returns_self(self) -> None:
        builder = FundBuilder("Chain")
        result = builder.category(FundCategory.ABS)
        assert result is builder
        result = builder.add_holding("X", 0.5)
        assert result is builder
