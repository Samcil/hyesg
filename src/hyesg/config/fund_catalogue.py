"""Fund composition catalogue — ~50 funds matching the C# ESG.

Provides the ``FundCatalogue`` container and the
``build_default_catalogue`` factory that assembles all fund
definitions using the fluent ``FundBuilder`` API.
"""

from __future__ import annotations

from hyesg.config.fee_wrappers import FEE_WRAPPERS
from hyesg.config.fund_builder import FundBuilder
from hyesg.config.funds import (
    FundCategory,
    FundDefinition,
    NetOfFeesFund,
)


class FundCatalogue:
    """Container for a collection of ``FundDefinition`` objects.

    Attributes:
        _funds: Mapping of fund name → definition.
        _fee_wrappers: List of net-of-fees wrappers.
    """

    def __init__(
        self,
        funds: list[FundDefinition],
        fee_wrappers: list[NetOfFeesFund] | None = None,
    ) -> None:
        self._funds: dict[str, FundDefinition] = {f.name: f for f in funds}
        self._fee_wrappers = list(fee_wrappers) if fee_wrappers else []

    def get(self, name: str) -> FundDefinition:
        """Look up a fund by exact name.

        Raises:
            KeyError: If the fund name is not found.
        """
        if name not in self._funds:
            msg = (
                f"Fund '{name}' not found. "
                f"Available: {sorted(self._funds.keys())}"
            )
            raise KeyError(msg)
        return self._funds[name]

    def by_category(self, cat: FundCategory) -> list[FundDefinition]:
        """Return all funds matching a category."""
        return [f for f in self._funds.values() if f.category == cat]

    def all_funds(self) -> list[FundDefinition]:
        """Return all fund definitions."""
        return list(self._funds.values())

    @property
    def fee_wrappers(self) -> list[NetOfFeesFund]:
        """Return all net-of-fees wrappers."""
        return list(self._fee_wrappers)

    def validate(self) -> list[str]:
        """Validate all funds and fee wrappers.

        Returns:
            List of validation error messages (empty = all OK).
        """
        errors: list[str] = []
        for fund in self._funds.values():
            total = sum(h.weight for h in fund.holdings)
            if abs(total - 1.0) > 1e-6:
                errors.append(
                    f"Fund '{fund.name}': weights sum to {total:.6f}, "
                    f"expected 1.0"
                )
        for fw in self._fee_wrappers:
            if fw.gross_fund not in self._funds:
                errors.append(
                    f"Fee wrapper '{fw.label}' references unknown fund "
                    f"'{fw.gross_fund}'"
                )
        return errors

    def __len__(self) -> int:
        return len(self._funds)

    def __contains__(self, name: str) -> bool:
        return name in self._funds


# ============================================================================
# Fund Definitions — grouped by category
# ============================================================================


def _world_equity_funds() -> list[FundDefinition]:
    """World Equity funds (5)."""
    return [
        (
            FundBuilder("Dev World GBP Unhedged")
            .category(FundCategory.WORLD_EQUITY)
            .add_equity("DevWorldEquity", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("Dev World GBP Hedged")
            .category(FundCategory.WORLD_EQUITY)
            .add_equity("DevWorldEquity", 1.0, "GBP")
            .hedge(1.0)
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("All World GBP Unhedged")
            .category(FundCategory.WORLD_EQUITY)
            .add_equity("DevWorldEquity", 0.88, "GBP")
            .add_equity("EMEquity", 0.12, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("All World GBP Hedged")
            .category(FundCategory.WORLD_EQUITY)
            .add_equity("DevWorldEquity", 0.88, "GBP")
            .add_equity("EMEquity", 0.12, "GBP")
            .hedge(1.0)
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("Dev World 50:50 GBP")
            .category(FundCategory.WORLD_EQUITY)
            .add_equity("UKEquity", 0.50, "GBP")
            .add_equity("DevWorldEquity", 0.50, "GBP")
            .rebalance_annually()
            .build()
        ),
    ]


def _distressed_debt_funds() -> list[FundDefinition]:
    """Distressed Debt funds (4)."""
    return [
        (
            FundBuilder("US Distressed Debt Unhedged")
            .category(FundCategory.DISTRESSED_DEBT)
            .add_bond("USDistressedDebt", 1.0, "USD")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("US Distressed Debt Hedged")
            .category(FundCategory.DISTRESSED_DEBT)
            .add_bond("USDistressedDebt", 1.0, "USD")
            .hedge(1.0)
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Global Distressed Debt Unhedged")
            .category(FundCategory.DISTRESSED_DEBT)
            .add_bond("USDistressedDebt", 0.60, "USD")
            .add_bond("EURDistressedDebt", 0.40, "EUR")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Global Distressed Debt Hedged")
            .category(FundCategory.DISTRESSED_DEBT)
            .add_bond("USDistressedDebt", 0.60, "USD")
            .add_bond("EURDistressedDebt", 0.40, "EUR")
            .hedge(1.0)
            .rebalance_quarterly()
            .build()
        ),
    ]


def _emd_funds() -> list[FundDefinition]:
    """Emerging Market Debt fund (1)."""
    return [
        (
            FundBuilder("EMD Blend")
            .category(FundCategory.EMD)
            .add_bond("EMBond", 0.70, "GBP")
            .add_equity("EMEquity", 0.30, "GBP")
            .rebalance_quarterly()
            .build()
        ),
    ]


def _infrastructure_debt_funds() -> list[FundDefinition]:
    """Infrastructure Debt funds (5)."""
    return [
        (
            FundBuilder("UK Infrastructure Debt Senior")
            .category(FundCategory.INFRASTRUCTURE_DEBT)
            .add_bond("UKInfraDebtSenior", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("UK Infrastructure Debt Subordinated")
            .category(FundCategory.INFRASTRUCTURE_DEBT)
            .add_bond("UKInfraDebtSubordinated", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("Global Infrastructure Debt Hedged")
            .category(FundCategory.INFRASTRUCTURE_DEBT)
            .add_bond("UKInfraDebtSenior", 0.40, "GBP")
            .add_bond("USInfraDebt", 0.35, "USD")
            .add_bond("EURInfraDebt", 0.25, "EUR")
            .hedge(1.0)
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("UK Infrastructure Debt Long Dated")
            .category(FundCategory.INFRASTRUCTURE_DEBT)
            .add_bond("UKInfraDebtLongDated", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("UK Infrastructure Debt Short Dated")
            .category(FundCategory.INFRASTRUCTURE_DEBT)
            .add_bond("UKInfraDebtShortDated", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
    ]


def _high_yield_funds() -> list[FundDefinition]:
    """High Yield funds (6)."""
    return [
        (
            FundBuilder("US High Yield Unhedged")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("USHighYield", 1.0, "USD")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("US High Yield Hedged")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("USHighYield", 1.0, "USD")
            .hedge(1.0)
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("EUR High Yield Unhedged")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("EURHighYield", 1.0, "EUR")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("EUR High Yield Hedged")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("EURHighYield", 1.0, "EUR")
            .hedge(1.0)
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Global High Yield Unhedged")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("USHighYield", 0.55, "USD")
            .add_bond("EURHighYield", 0.45, "EUR")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Global High Yield Hedged")
            .category(FundCategory.HIGH_YIELD)
            .add_bond("USHighYield", 0.55, "USD")
            .add_bond("EURHighYield", 0.45, "EUR")
            .hedge(1.0)
            .rebalance_quarterly()
            .build()
        ),
    ]


def _senior_loans_funds() -> list[FundDefinition]:
    """Senior Loans and Multi-Asset Credit funds (2)."""
    return [
        (
            FundBuilder("Senior Secured Loans")
            .category(FundCategory.SENIOR_LOANS)
            .add_bond("SeniorSecuredLoans", 1.0, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Multi-Asset Credit")
            .category(FundCategory.SENIOR_LOANS)
            .add_bond("USHighYield", 0.35, "USD")
            .add_bond("SeniorSecuredLoans", 0.35, "GBP")
            .add_bond("EMBond", 0.30, "GBP")
            .rebalance_quarterly()
            .build()
        ),
    ]


def _abs_funds() -> list[FundDefinition]:
    """Asset-Backed Securities funds (3)."""
    return [
        (
            FundBuilder("UK ABS")
            .category(FundCategory.ABS)
            .add_bond("UKABS", 1.0, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("US ABS")
            .category(FundCategory.ABS)
            .add_bond("USABS", 1.0, "USD")
            .hedge(1.0)
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Global ABS")
            .category(FundCategory.ABS)
            .add_bond("UKABS", 0.40, "GBP")
            .add_bond("USABS", 0.35, "USD")
            .add_bond("EURABS", 0.25, "EUR")
            .hedge(1.0)
            .rebalance_quarterly()
            .build()
        ),
    ]


def _insurance_linked_funds() -> list[FundDefinition]:
    """Insurance Linked Securities fund (1)."""
    return [
        (
            FundBuilder("Insurance Linked Securities")
            .category(FundCategory.INSURANCE_LINKED)
            .add_bond("InsuranceLinked", 1.0, "USD")
            .hedge(1.0)
            .rebalance_annually()
            .build()
        ),
    ]


def _absolute_return_funds() -> list[FundDefinition]:
    """Absolute Return fund (1)."""
    return [
        (
            FundBuilder("Absolute Return")
            .category(FundCategory.ABSOLUTE_RETURN)
            .add_holding("AbsoluteReturn", 1.0, "GBP")
            .rebalance_quarterly()
            .build()
        ),
    ]


def _dgf_funds() -> list[FundDefinition]:
    """Diversified Growth Funds — 3 tiers."""
    return [
        (
            FundBuilder("DGF Tier 1")
            .category(FundCategory.DGF)
            .add_equity("DevWorldEquity", 0.40, "GBP")
            .add_bond("UKCorpBond", 0.20, "GBP")
            .add_holding("AbsoluteReturn", 0.15, "GBP")
            .add_bond("UKGilt", 0.10, "GBP")
            .add_holding("UKProperty", 0.10, "GBP")
            .add_bond("EMBond", 0.05, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("DGF Tier 2")
            .category(FundCategory.DGF)
            .add_equity("DevWorldEquity", 0.30, "GBP")
            .add_bond("UKCorpBond", 0.25, "GBP")
            .add_holding("AbsoluteReturn", 0.15, "GBP")
            .add_bond("UKGilt", 0.15, "GBP")
            .add_holding("UKProperty", 0.10, "GBP")
            .add_bond("EMBond", 0.05, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("DGF Tier 3")
            .category(FundCategory.DGF)
            .add_equity("DevWorldEquity", 0.20, "GBP")
            .add_bond("UKCorpBond", 0.30, "GBP")
            .add_holding("AbsoluteReturn", 0.15, "GBP")
            .add_bond("UKGilt", 0.20, "GBP")
            .add_holding("UKProperty", 0.10, "GBP")
            .add_bond("EMBond", 0.05, "GBP")
            .rebalance_quarterly()
            .build()
        ),
    ]


def _private_equity_funds() -> list[FundDefinition]:
    """Private Equity funds (3)."""
    return [
        (
            FundBuilder("UK Private Equity")
            .category(FundCategory.PRIVATE_EQUITY)
            .add_equity("UKPrivateEquity", 1.0, "GBP")
            .no_rebalance()
            .build()
        ),
        (
            FundBuilder("US Private Equity")
            .category(FundCategory.PRIVATE_EQUITY)
            .add_equity("USPrivateEquity", 1.0, "USD")
            .no_rebalance()
            .build()
        ),
        (
            FundBuilder("Global Private Equity")
            .category(FundCategory.PRIVATE_EQUITY)
            .add_equity("UKPrivateEquity", 0.30, "GBP")
            .add_equity("USPrivateEquity", 0.40, "USD")
            .add_equity("EURPrivateEquity", 0.30, "EUR")
            .no_rebalance()
            .build()
        ),
    ]


def _direct_lending_funds() -> list[FundDefinition]:
    """Direct Lending funds (2)."""
    return [
        (
            FundBuilder("UK Direct Lending")
            .category(FundCategory.DIRECT_LENDING)
            .add_bond("UKDirectLending", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("US Direct Lending")
            .category(FundCategory.DIRECT_LENDING)
            .add_bond("USDirectLending", 1.0, "USD")
            .hedge(1.0)
            .rebalance_annually()
            .build()
        ),
    ]


def _property_funds() -> list[FundDefinition]:
    """Property funds (4)."""
    return [
        (
            FundBuilder("UK Direct Property")
            .category(FundCategory.PROPERTY)
            .add_holding("UKProperty", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("UK Property Unit Trust")
            .category(FundCategory.PROPERTY)
            .add_holding("UKProperty", 0.90, "GBP")
            .add_bond("UKGilt", 0.10, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("UK Secondary Property")
            .category(FundCategory.PROPERTY)
            .add_holding("UKSecondaryProperty", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
        (
            FundBuilder("UK Long Lease Property")
            .category(FundCategory.PROPERTY)
            .add_holding("UKLongLeaseProperty", 1.0, "GBP")
            .rebalance_annually()
            .build()
        ),
    ]


def _parmenion_funds() -> list[FundDefinition]:
    """Parmenion legacy funds (5)."""
    return [
        (
            FundBuilder("Parmenion Cautious")
            .category(FundCategory.PARMENION)
            .add_equity("DevWorldEquity", 0.20, "GBP")
            .add_bond("UKCorpBond", 0.35, "GBP")
            .add_bond("UKGilt", 0.35, "GBP")
            .add_holding("UKProperty", 0.10, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Parmenion Balanced")
            .category(FundCategory.PARMENION)
            .add_equity("DevWorldEquity", 0.40, "GBP")
            .add_bond("UKCorpBond", 0.25, "GBP")
            .add_bond("UKGilt", 0.20, "GBP")
            .add_holding("UKProperty", 0.15, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Parmenion Growth")
            .category(FundCategory.PARMENION)
            .add_equity("DevWorldEquity", 0.60, "GBP")
            .add_bond("UKCorpBond", 0.15, "GBP")
            .add_bond("UKGilt", 0.10, "GBP")
            .add_holding("UKProperty", 0.15, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Parmenion Adventurous")
            .category(FundCategory.PARMENION)
            .add_equity("DevWorldEquity", 0.75, "GBP")
            .add_bond("UKCorpBond", 0.10, "GBP")
            .add_bond("UKGilt", 0.05, "GBP")
            .add_holding("UKProperty", 0.10, "GBP")
            .rebalance_quarterly()
            .build()
        ),
        (
            FundBuilder("Parmenion Equity")
            .category(FundCategory.PARMENION)
            .add_equity("DevWorldEquity", 0.85, "GBP")
            .add_equity("EMEquity", 0.15, "GBP")
            .rebalance_quarterly()
            .build()
        ),
    ]


def _build_all_gross_funds() -> list[FundDefinition]:
    """Assemble all gross fund definitions."""
    funds: list[FundDefinition] = []
    funds.extend(_world_equity_funds())
    funds.extend(_distressed_debt_funds())
    funds.extend(_emd_funds())
    funds.extend(_infrastructure_debt_funds())
    funds.extend(_high_yield_funds())
    funds.extend(_senior_loans_funds())
    funds.extend(_abs_funds())
    funds.extend(_insurance_linked_funds())
    funds.extend(_absolute_return_funds())
    funds.extend(_dgf_funds())
    funds.extend(_private_equity_funds())
    funds.extend(_direct_lending_funds())
    funds.extend(_property_funds())
    funds.extend(_parmenion_funds())
    return funds


def build_default_catalogue() -> FundCatalogue:
    """Build the full ~50 fund catalogue matching the C# ESG.

    Returns:
        ``FundCatalogue`` containing all gross funds and fee wrappers.
    """
    return FundCatalogue(
        funds=_build_all_gross_funds(),
        fee_wrappers=list(FEE_WRAPPERS),
    )


__all__ = [
    "FundCatalogue",
    "build_default_catalogue",
]
