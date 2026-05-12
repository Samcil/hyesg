"""Tests for FundCatalogue and the default catalogue builder."""

from __future__ import annotations

import pytest

from hyesg.config.fee_wrappers import FEE_WRAPPERS, get_fee_wrappers
from hyesg.config.fund_catalogue import FundCatalogue, build_default_catalogue
from hyesg.config.funds import FundCategory, FundDefinition, HoldingSpec


# ============================================================================
# build_default_catalogue
# ============================================================================


@pytest.fixture()
def catalogue() -> FundCatalogue:
    """Build the default catalogue once for the test session."""
    return build_default_catalogue()


class TestBuildDefaultCatalogue:
    """Tests for the build_default_catalogue factory."""

    def test_returns_catalogue(self, catalogue: FundCatalogue) -> None:
        assert isinstance(catalogue, FundCatalogue)

    def test_fund_count_at_least_40(self, catalogue: FundCatalogue) -> None:
        assert len(catalogue) >= 40

    def test_no_validation_errors(self, catalogue: FundCatalogue) -> None:
        errors = catalogue.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_all_weights_sum_to_one(self, catalogue: FundCatalogue) -> None:
        for fund in catalogue.all_funds():
            total = sum(h.weight for h in fund.holdings)
            assert total == pytest.approx(
                1.0, abs=1e-6
            ), f"Fund '{fund.name}' weights sum to {total}"

    def test_unique_fund_names(self, catalogue: FundCatalogue) -> None:
        names = [f.name for f in catalogue.all_funds()]
        assert len(names) == len(set(names)), "Duplicate fund names found"


# ============================================================================
# FundCatalogue.get
# ============================================================================


class TestCatalogueGet:
    """Tests for fund lookup by name."""

    def test_get_existing_fund(self, catalogue: FundCatalogue) -> None:
        fund = catalogue.get("Dev World GBP Unhedged")
        assert fund.name == "Dev World GBP Unhedged"
        assert fund.category == FundCategory.WORLD_EQUITY

    def test_get_missing_fund_raises(self, catalogue: FundCatalogue) -> None:
        with pytest.raises(KeyError, match="not found"):
            catalogue.get("NonExistent Fund XYZ")

    def test_contains(self, catalogue: FundCatalogue) -> None:
        assert "Dev World GBP Unhedged" in catalogue
        assert "DOES NOT EXIST" not in catalogue


# ============================================================================
# FundCatalogue.by_category
# ============================================================================


class TestCatalogueByCategory:
    """Tests for fund lookup by category."""

    def test_world_equity_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.WORLD_EQUITY)
        assert len(funds) == 5

    def test_distressed_debt_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.DISTRESSED_DEBT)
        assert len(funds) == 4

    def test_emd_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.EMD)
        assert len(funds) == 1

    def test_infrastructure_debt_count(
        self, catalogue: FundCatalogue
    ) -> None:
        funds = catalogue.by_category(FundCategory.INFRASTRUCTURE_DEBT)
        assert len(funds) == 5

    def test_high_yield_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.HIGH_YIELD)
        assert len(funds) == 6

    def test_senior_loans_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.SENIOR_LOANS)
        assert len(funds) == 2

    def test_abs_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.ABS)
        assert len(funds) == 3

    def test_insurance_linked_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.INSURANCE_LINKED)
        assert len(funds) == 1

    def test_absolute_return_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.ABSOLUTE_RETURN)
        assert len(funds) == 1

    def test_dgf_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.DGF)
        assert len(funds) == 3

    def test_private_equity_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.PRIVATE_EQUITY)
        assert len(funds) == 3

    def test_direct_lending_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.DIRECT_LENDING)
        assert len(funds) == 2

    def test_property_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.PROPERTY)
        assert len(funds) == 4

    def test_parmenion_count(self, catalogue: FundCatalogue) -> None:
        funds = catalogue.by_category(FundCategory.PARMENION)
        assert len(funds) == 5

    def test_all_categories_represented(
        self, catalogue: FundCatalogue
    ) -> None:
        represented = {f.category for f in catalogue.all_funds()}
        # NET_OF_FEES is only used for fee wrappers, not gross funds
        expected = set(FundCategory) - {FundCategory.NET_OF_FEES}
        assert represented == expected


# ============================================================================
# Specific fund properties
# ============================================================================


class TestSpecificFunds:
    """Tests for specific fund properties."""

    def test_dev_world_hedged_has_hedge(
        self, catalogue: FundCatalogue
    ) -> None:
        fund = catalogue.get("Dev World GBP Hedged")
        assert fund.hedge_ratio == 1.0

    def test_dev_world_unhedged_no_hedge(
        self, catalogue: FundCatalogue
    ) -> None:
        fund = catalogue.get("Dev World GBP Unhedged")
        assert fund.hedge_ratio is None

    def test_all_world_has_em_component(
        self, catalogue: FundCatalogue
    ) -> None:
        fund = catalogue.get("All World GBP Unhedged")
        assets = {h.asset_name for h in fund.holdings}
        assert "EMEquity" in assets

    def test_private_equity_no_rebalance(
        self, catalogue: FundCatalogue
    ) -> None:
        fund = catalogue.get("UK Private Equity")
        from hyesg.config.funds import FundRebalanceStrategy

        assert fund.rebalance == FundRebalanceStrategy.NONE

    def test_dgf_tier1_multi_asset(self, catalogue: FundCatalogue) -> None:
        fund = catalogue.get("DGF Tier 1")
        assert len(fund.holdings) >= 4

    def test_parmenion_cautious_conservative(
        self, catalogue: FundCatalogue
    ) -> None:
        fund = catalogue.get("Parmenion Cautious")
        equity_weight = sum(
            h.weight
            for h in fund.holdings
            if "Equity" in h.asset_name or "equity" in h.asset_name
        )
        assert equity_weight <= 0.3  # cautious = low equity

    def test_us_high_yield_usd_economy(
        self, catalogue: FundCatalogue
    ) -> None:
        fund = catalogue.get("US High Yield Unhedged")
        assert fund.holdings[0].economy == "USD"


# ============================================================================
# Fee wrappers
# ============================================================================


class TestFeeWrappers:
    """Tests for fee wrapper definitions."""

    def test_fee_wrappers_count(self) -> None:
        wrappers = get_fee_wrappers()
        assert len(wrappers) >= 20

    def test_fee_wrappers_not_empty(self) -> None:
        assert len(FEE_WRAPPERS) > 0

    def test_all_wrappers_reference_valid_funds(
        self, catalogue: FundCatalogue
    ) -> None:
        errors = catalogue.validate()
        fee_errors = [e for e in errors if "Fee wrapper" in e]
        assert fee_errors == [], f"Invalid fee wrappers: {fee_errors}"

    def test_catalogue_fee_wrappers(self, catalogue: FundCatalogue) -> None:
        assert len(catalogue.fee_wrappers) >= 20

    def test_fee_wrapper_has_positive_fee(self) -> None:
        for fw in FEE_WRAPPERS:
            assert fw.fee_bps > 0, f"Wrapper '{fw.label}' has zero fee"


# ============================================================================
# FundCatalogue validation
# ============================================================================


class TestCatalogueValidation:
    """Tests for the validate method."""

    def test_validate_good_catalogue(self, catalogue: FundCatalogue) -> None:
        assert catalogue.validate() == []

    def test_validate_bad_fee_wrapper(self) -> None:
        from hyesg.config.funds import NetOfFeesFund

        bad_wrapper = NetOfFeesFund(
            gross_fund="DOES NOT EXIST",
            fee_bps=50.0,
            label="Bad Wrapper",
        )
        cat = FundCatalogue(
            funds=[
                FundDefinition(
                    name="OK Fund",
                    category=FundCategory.ABS,
                    holdings=[HoldingSpec(asset_name="ABS", weight=1.0)],
                )
            ],
            fee_wrappers=[bad_wrapper],
        )
        errors = cat.validate()
        assert len(errors) == 1
        assert "DOES NOT EXIST" in errors[0]

    def test_len(self, catalogue: FundCatalogue) -> None:
        assert len(catalogue) == len(catalogue.all_funds())
