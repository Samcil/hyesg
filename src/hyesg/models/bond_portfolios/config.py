"""Pydantic configuration models for bond portfolios.

Defines the user-facing configuration hierarchy: individual bond holdings,
rebalancing strategies, complete portfolio specifications, and named
benchmark portfolio configs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class BondType(Enum):
    """Bond instrument type."""

    GOVERNMENT = "government"
    CORPORATE = "corporate"
    INDEX_LINKED = "index_linked"
    SWAP_CURVE = "swap_curve"


class MaturityType(Enum):
    """Bond maturity behaviour."""

    FIXED = "fixed"
    ROLLING = "rolling"


class BondHoldingConfig(BaseModel, frozen=True):
    """Single bond position within a portfolio.

    Attributes:
        maturity: Time to maturity in years.
        maturity_type: Whether the bond has fixed or rolling maturity.
        weight: Portfolio weight (fraction of total).
        coupon_rate: Annual coupon rate (e.g. 0.04 for 4%).
        coupon_frequency: Coupon payments per year (0 = zero-coupon bond).
        is_at_par: If True the bond is coupon-bearing at par.
        bond_type: Instrument classification.
        credit_class: Credit rating (None for government bonds).
        liquidity: Liquidity tier for credit spread modelling.
        economy: Currency / economy identifier.
    """

    maturity: float
    maturity_type: MaturityType = MaturityType.FIXED
    weight: float
    coupon_rate: float = 0.0
    coupon_frequency: int = 0
    is_at_par: bool = False
    bond_type: BondType = BondType.GOVERNMENT
    credit_class: CreditClass | None = None
    liquidity: Liquidity = Field(default_factory=lambda: _default_liquidity())
    economy: str = "GBP"

    @model_validator(mode="after")
    def _validate_corporate_has_credit(self) -> BondHoldingConfig:
        """Corporate bonds must have a credit class."""
        if self.bond_type == BondType.CORPORATE and self.credit_class is None:
            msg = "Corporate bonds require a credit_class"
            raise ValueError(msg)
        return self


class RebalancingConfig(BaseModel, frozen=True):
    """Rebalancing strategy configuration.

    Attributes:
        strategy: Rebalancing strategy name.
        frequency: Rebalancing frequency in months.
        rebalance_to_initial_maturity: Whether to reset maturity on rebalance.
    """

    strategy: str = "maturity_and_allocation"
    frequency: int = 12
    rebalance_to_initial_maturity: bool = True


class BondPortfolioConfig(BaseModel, frozen=True):
    """Complete bond portfolio configuration.

    Attributes:
        name: Human-readable portfolio name.
        holdings: List of individual bond positions.
        rebalancing: Rebalancing strategy settings.
        nominal_economy: Currency for the nominal yield curve.
        n_issues_per_tranche: Number of bond issues per tranche (corporate).
    """

    name: str
    holdings: list[BondHoldingConfig]
    rebalancing: RebalancingConfig = Field(default_factory=RebalancingConfig)
    nominal_economy: str = "GBP"
    n_issues_per_tranche: int = 1

    @model_validator(mode="after")
    def _validate_weights_positive(self) -> BondPortfolioConfig:
        """All holding weights must be positive."""
        for h in self.holdings:
            if h.weight < 0.0:
                msg = f"Holding weight must be >= 0, got {h.weight}"
                raise ValueError(msg)
        return self


class BenchmarkPortfolioConfig(BaseModel, frozen=True):
    """Named benchmark portfolio (e.g. 'Sterling Corporate Index GBP').

    Attributes:
        name: Canonical benchmark name.
        description: Optional human-readable description.
        benchmark_code: Industry code (e.g. 'UR00' for ICE BofA Sterling Corp).
        portfolio: The underlying portfolio configuration.
    """

    name: str
    description: str = ""
    benchmark_code: str = ""
    portfolio: BondPortfolioConfig


# ── Deferred imports to avoid circular dependency with core.enums ──


def _default_liquidity() -> Liquidity:
    """Return the default Liquidity value (deferred import)."""
    from hyesg.core.enums import Liquidity

    return Liquidity.HIGH


# Re-export core enums used in type annotations so callers can import
# from this module if convenient.
from hyesg.core.enums import CreditClass, Liquidity  # noqa: E402

__all__ = [
    "BenchmarkPortfolioConfig",
    "BondHoldingConfig",
    "BondPortfolioConfig",
    "BondType",
    "CreditClass",
    "Liquidity",
    "MaturityType",
    "RebalancingConfig",
]
