"""Fund composition definitions for the ESG fund catalogue.

Defines the core data models used to represent investment fund
compositions — holdings, rebalance strategies, categories, and
the ``FundDefinition`` that ties them together.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FundCategory(StrEnum):
    """Classification of fund types in the ESG catalogue."""

    WORLD_EQUITY = "WorldEquity"
    DISTRESSED_DEBT = "DistressedDebt"
    EMD = "EMD"
    INFRASTRUCTURE_DEBT = "InfrastructureDebt"
    HIGH_YIELD = "HighYield"
    SENIOR_LOANS = "SeniorLoans"
    ABS = "ABS"
    INSURANCE_LINKED = "InsuranceLinked"
    ABSOLUTE_RETURN = "AbsoluteReturn"
    DGF = "DGF"
    PRIVATE_EQUITY = "PrivateEquity"
    DIRECT_LENDING = "DirectLending"
    PROPERTY = "Property"
    PARMENION = "Parmenion"
    NET_OF_FEES = "NetOfFees"


class FundRebalanceStrategy(StrEnum):
    """Fund-level rebalancing frequency."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    NONE = "none"


class HoldingSpec(BaseModel):
    """A single asset holding within a fund.

    Attributes:
        asset_name: Identifier of the underlying asset/index.
        weight: Portfolio weight (fraction of 1.0).
        economy: Economy/currency zone the asset belongs to.
    """

    model_config = ConfigDict(frozen=True)

    asset_name: str
    weight: float
    economy: str = "GBP"


class FundDefinition(BaseModel):
    """Complete specification of an investment fund.

    Attributes:
        name: Unique fund identifier.
        category: Fund classification category.
        holdings: Constituent asset holdings with weights.
        rebalance: Rebalancing frequency.
        fee_bps: Annual fee in basis points.
        currency: Base currency of the fund.
        hedge_ratio: FX hedge ratio (None = unhedged, 1.0 = fully hedged).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    category: FundCategory
    holdings: list[HoldingSpec] = Field(min_length=1)
    rebalance: FundRebalanceStrategy = FundRebalanceStrategy.ANNUAL
    fee_bps: float = 0.0
    currency: str = "GBP"
    hedge_ratio: float | None = None

    @model_validator(mode="after")
    def _validate_weights(self) -> FundDefinition:
        total = sum(h.weight for h in self.holdings)
        if abs(total - 1.0) > 1e-6:
            msg = (
                f"Fund '{self.name}' holdings weights sum to {total:.6f}, "
                f"expected 1.0"
            )
            raise ValueError(msg)
        return self

    @property
    def total_weight(self) -> float:
        """Sum of all holding weights."""
        return sum(h.weight for h in self.holdings)


class NetOfFeesFund(BaseModel):
    """A fee-adjusted wrapper around an existing gross fund.

    Attributes:
        gross_fund: Name of the underlying gross fund.
        fee_bps: Annual fee deduction in basis points.
        label: Display label for the net-of-fees variant.
    """

    model_config = ConfigDict(frozen=True)

    gross_fund: str
    fee_bps: float
    label: str


__all__ = [
    "FundCategory",
    "FundDefinition",
    "FundRebalanceStrategy",
    "HoldingSpec",
    "NetOfFeesFund",
]
