"""Fluent builder API for constructing fund definitions.

Provides ``FundBuilder`` — a chainable builder that makes it easy
to construct ``FundDefinition`` instances with readable, declarative
syntax.
"""

from __future__ import annotations

from hyesg.config.funds import (
    FundCategory,
    FundDefinition,
    FundRebalanceStrategy,
    HoldingSpec,
)


class FundBuilder:
    """Fluent builder for ``FundDefinition``.

    Example::

        fund = (
            FundBuilder("Dev World GBP Unhedged")
            .category(FundCategory.WORLD_EQUITY)
            .add_equity("DevWorldEquity", 1.0, "GBP")
            .rebalance_annually()
            .build()
        )
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._category: FundCategory | None = None
        self._holdings: list[HoldingSpec] = []
        self._rebalance = FundRebalanceStrategy.ANNUAL
        self._fee_bps: float = 0.0
        self._currency: str = "GBP"
        self._hedge_ratio: float | None = None

    def category(self, cat: FundCategory) -> FundBuilder:
        """Set the fund category."""
        self._category = cat
        return self

    def add_holding(
        self, asset: str, weight: float, economy: str = "GBP"
    ) -> FundBuilder:
        """Add a generic holding."""
        self._holdings.append(
            HoldingSpec(asset_name=asset, weight=weight, economy=economy)
        )
        return self

    def add_equity(
        self, name: str, weight: float, economy: str = "GBP"
    ) -> FundBuilder:
        """Add an equity holding (alias for ``add_holding``)."""
        return self.add_holding(name, weight, economy)

    def add_bond(
        self, name: str, weight: float, economy: str = "GBP"
    ) -> FundBuilder:
        """Add a bond holding (alias for ``add_holding``)."""
        return self.add_holding(name, weight, economy)

    def fee_bps(self, bps: float) -> FundBuilder:
        """Set fee in basis points."""
        self._fee_bps = bps
        return self

    def hedge(self, ratio: float = 1.0) -> FundBuilder:
        """Set the FX hedge ratio."""
        self._hedge_ratio = ratio
        return self

    def rebalance_quarterly(self) -> FundBuilder:
        """Set quarterly rebalancing."""
        self._rebalance = FundRebalanceStrategy.QUARTERLY
        return self

    def rebalance_annually(self) -> FundBuilder:
        """Set annual rebalancing."""
        self._rebalance = FundRebalanceStrategy.ANNUAL
        return self

    def rebalance_monthly(self) -> FundBuilder:
        """Set monthly rebalancing."""
        self._rebalance = FundRebalanceStrategy.MONTHLY
        return self

    def no_rebalance(self) -> FundBuilder:
        """Disable rebalancing."""
        self._rebalance = FundRebalanceStrategy.NONE
        return self

    def currency(self, ccy: str) -> FundBuilder:
        """Set the fund base currency."""
        self._currency = ccy
        return self

    def build(self) -> FundDefinition:
        """Build and validate the ``FundDefinition``."""
        if self._category is None:
            msg = f"Fund '{self._name}' must have a category set"
            raise ValueError(msg)
        return FundDefinition(
            name=self._name,
            category=self._category,
            holdings=self._holdings,
            rebalance=self._rebalance,
            fee_bps=self._fee_bps,
            currency=self._currency,
            hedge_ratio=self._hedge_ratio,
        )


__all__ = ["FundBuilder"]
