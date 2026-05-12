"""Tests for benchmark portfolio registry."""

from __future__ import annotations

import pytest

from hyesg.models.bond_portfolios.benchmarks import (
    BENCHMARK_REGISTRY,
    get_benchmark,
    list_benchmarks,
)
from hyesg.models.bond_portfolios.config import (
    BondType,
)


class TestBenchmarkRegistry:
    """Tests for the benchmark registry population."""

    def test_at_least_44_benchmarks(self) -> None:
        """Registry contains at least 44 benchmarks."""
        assert len(BENCHMARK_REGISTRY) >= 44

    def test_list_benchmarks_sorted(self) -> None:
        """list_benchmarks returns sorted names."""
        names = list_benchmarks()
        assert names == sorted(names)
        assert len(names) >= 44

    def test_all_configs_valid(self) -> None:
        """Every registered benchmark has a valid portfolio config."""
        for name, bench in BENCHMARK_REGISTRY.items():
            assert bench.name == name
            assert len(bench.portfolio.holdings) > 0
            total_weight = sum(h.weight for h in bench.portfolio.holdings)
            assert total_weight > 0.0, f"{name}: weights sum to {total_weight}"

    def test_get_benchmark_returns_config(self) -> None:
        """get_benchmark returns the registered config."""
        bench = get_benchmark("NominalGiltsBasket")
        assert bench.name == "NominalGiltsBasket"

    def test_get_benchmark_unknown_raises(self) -> None:
        """get_benchmark raises KeyError for unknown names."""
        with pytest.raises(KeyError, match="NonExistent"):
            get_benchmark("NonExistent")


class TestSpecificBenchmarks:
    """Spot-checks on specific benchmark definitions."""

    def test_nominal_gilts_basket(self) -> None:
        """NominalGiltsBasket has 6 govt bond holdings."""
        b = get_benchmark("NominalGiltsBasket")
        assert len(b.portfolio.holdings) == 6
        assert all(h.bond_type == BondType.GOVERNMENT for h in b.portfolio.holdings)
        weights = [h.weight for h in b.portfolio.holdings]
        assert abs(sum(weights) - 1.0) < 1e-12

    def test_nominal_swaps_basket(self) -> None:
        """NominalSwapsBasket has swap-curve bonds."""
        b = get_benchmark("NominalSwapsBasket")
        assert len(b.portfolio.holdings) == 6
        assert all(h.bond_type == BondType.SWAP_CURVE for h in b.portfolio.holdings)

    def test_real_rpi_basket(self) -> None:
        """RealRPIBasket has index-linked bonds."""
        b = get_benchmark("RealRPIBasket")
        assert len(b.portfolio.holdings) == 6
        assert all(h.bond_type == BondType.INDEX_LINKED for h in b.portfolio.holdings)

    def test_sterling_corporate_index(self) -> None:
        """Sterling Corporate Index GBP has 3 corporate holdings."""
        b = get_benchmark("Sterling Corporate Index GBP")
        assert len(b.portfolio.holdings) == 3
        assert all(h.bond_type == BondType.CORPORATE for h in b.portfolio.holdings)
        assert b.portfolio.n_issues_per_tranche == 2

    def test_us_leveraged_loan(self) -> None:
        """US Leveraged Loan GBP has 4 holdings in USD."""
        b = get_benchmark("US Leveraged Loan GBP")
        assert len(b.portfolio.holdings) == 4
        assert all(h.economy == "USD" for h in b.portfolio.holdings)

    def test_hedged_variant_uses_gbp(self) -> None:
        """GBP Hedged variants use economy='GBP'."""
        b = get_benchmark("US Leveraged Loan GBP Hedged")
        assert all(h.economy == "GBP" for h in b.portfolio.holdings)

    def test_illiquid_variant(self) -> None:
        """Illiquid variants use low liquidity."""
        from hyesg.core.enums import Liquidity

        b = get_benchmark("US Leveraged Loan Illiquid GBP")
        assert all(h.liquidity == Liquidity.LOW for h in b.portfolio.holdings)

    def test_rolling_govt_bond(self) -> None:
        """Rolling1yrGovtBond has rolling maturity type."""
        from hyesg.models.bond_portfolios.config import MaturityType

        b = get_benchmark("Rolling1yrGovtBond")
        assert len(b.portfolio.holdings) == 1
        assert b.portfolio.holdings[0].maturity_type == MaturityType.ROLLING
        assert b.portfolio.holdings[0].maturity == 1.0

    def test_global_distressed_debt_multi_economy(self) -> None:
        """Global Distressed Debt GBP has multi-economy holdings."""
        b = get_benchmark("Global Distressed Debt GBP")
        economies = {h.economy for h in b.portfolio.holdings}
        assert "EUR" in economies
        assert "USD" in economies

    def test_corp_ultrashort_trw(self) -> None:
        """TRW ultra-short corporate benchmarks exist."""
        b_med = get_benchmark("CorpUltraShort_B_Medium")
        b_low = get_benchmark("CorpUltraShort_B_Low")
        from hyesg.core.enums import CreditClass, Liquidity

        assert b_med.portfolio.holdings[0].credit_class == CreditClass.B
        assert b_med.portfolio.holdings[0].liquidity == Liquidity.MEDIUM
        assert b_low.portfolio.holdings[0].liquidity == Liquidity.LOW
        assert b_med.portfolio.n_issues_per_tranche == 6

    def test_em_local_currency_is_govt(self) -> None:
        """EM local currency debt is a government bond."""
        b = get_benchmark("Emerging Market Debt (local currency) GBP")
        assert b.portfolio.holdings[0].bond_type == BondType.GOVERNMENT
        assert b.portfolio.holdings[0].economy == "EM"

    def test_global_corporate_fi_multi_economy(self) -> None:
        """Global Corporate FI GBP has holdings across 3 economies."""
        b = get_benchmark("Global Corporate FI GBP")
        economies = {h.economy for h in b.portfolio.holdings}
        assert len(economies) == 3
        assert "GBP" in economies
        assert "EUR" in economies
        assert "USD" in economies


class TestAllBenchmarkNames:
    """Ensure all 44 expected benchmark names are present."""

    EXPECTED_NAMES: list[str] = [
        "Sterling Corporate Index GBP",
        "US Leveraged Loan GBP",
        "US Leveraged Loan Illiquid GBP",
        "US Leveraged Loan GBP Hedged",
        "US Leveraged Loan Illiquid GBP Hedged",
        "EU Leveraged Loan GBP",
        "EU Leveraged Loan Illiquid GBP",
        "EU Leveraged Loan GBP Hedged",
        "EU Leveraged Loan Illiquid GBP Hedged",
        "US Distressed Debt GBP",
        "US Distressed Debt GBP Hedged",
        "Global Distressed Debt GBP",
        "Global Distressed Debt GBP Hedged",
        "Emerging Market Debt (hard currency) GBP",
        "Emerging Market Debt (hard currency) GBP Hedged",
        "Emerging Market Debt (local currency) GBP",
        "US High Yield Debt GBP",
        "US High Yield Debt GBP Hedged",
        "EU High Yield Debt GBP",
        "EU High Yield Debt GBP Hedged",
        "Global High Yield Debt GBP",
        "Global High Yield Debt GBP Hedged",
        "UK Infrastructure Debt GBP",
        "US Infrastructure Debt GBP",
        "US Infrastructure Debt GBP Hedged",
        "EU Infrastructure Debt GBP",
        "EU Infrastructure Debt GBP Hedged",
        "Global Corporate FI GBP",
        "Global Corporate FI GBP Hedged",
        "Global Aggregate FI GBP",
        "Global Aggregate FI GBP Hedged",
        "NominalGiltsBasket",
        "NominalSwapsBasket",
        "RealRPIBasket",
        "RealCPIBasket",
        "CorpBasket",
        "17yrCorpBasket",
        "SterlingInvestmentGradeIndex",
        "GlobalHighYieldIndex",
        "Rolling1yrGovtBond",
        "4yrGiltsBond",
        "8yrGiltsBond",
        "CorpUltraShort_B_Medium",
        "CorpUltraShort_B_Low",
    ]

    @pytest.mark.parametrize("name", EXPECTED_NAMES)
    def test_benchmark_exists(self, name: str) -> None:
        """Each expected benchmark is registered."""
        assert name in BENCHMARK_REGISTRY, f"Missing benchmark: {name}"
