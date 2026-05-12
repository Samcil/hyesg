"""Bond grid construction utilities.

Builds systematic grids of ``BondHoldingConfig`` objects by combining
standard tenor and coupon dimensions. Each grid function returns a flat
list of equally-weighted holdings.
"""

from __future__ import annotations

from hyesg.core.enums import CreditClass, Liquidity
from hyesg.models.bond_portfolios.config import (
    BondHoldingConfig,
    BondType,
    MaturityType,
)

STANDARD_TENORS: list[float] = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
STANDARD_COUPONS: list[float] = [0.0, 0.02, 0.04, 0.06, 0.08]


def build_bond_grid(
    tenors: list[float] | None = None,
    coupons: list[float] | None = None,
    bond_type: BondType = BondType.GOVERNMENT,
    maturity_type: MaturityType = MaturityType.FIXED,
    economy: str = "GBP",
    credit_class: CreditClass | None = None,
    liquidity: Liquidity = Liquidity.HIGH,
) -> list[BondHoldingConfig]:
    """Build a grid of bond holding configs from tenor × coupon combinations.

    Each combination receives equal weight (1 / n_combinations).

    Args:
        tenors: Maturities in years.  Defaults to ``STANDARD_TENORS``.
        coupons: Coupon rates.  Defaults to ``STANDARD_COUPONS``.
        bond_type: Instrument classification.
        maturity_type: Fixed or rolling maturity.
        economy: Currency / economy identifier.
        credit_class: Credit rating (required for corporate bonds).
        liquidity: Liquidity tier.

    Returns:
        List of ``BondHoldingConfig`` forming the tenor × coupon grid.
    """
    tenors = tenors if tenors is not None else list(STANDARD_TENORS)
    coupons = coupons if coupons is not None else list(STANDARD_COUPONS)

    n_total = len(tenors) * len(coupons)
    if n_total == 0:
        return []
    weight = 1.0 / n_total

    holdings: list[BondHoldingConfig] = []
    for tenor in tenors:
        for coupon in coupons:
            is_zcb = coupon == 0.0
            holdings.append(
                BondHoldingConfig(
                    maturity=tenor,
                    maturity_type=maturity_type,
                    weight=weight,
                    coupon_rate=coupon,
                    coupon_frequency=0 if is_zcb else 2,
                    is_at_par=not is_zcb,
                    bond_type=bond_type,
                    credit_class=credit_class,
                    liquidity=liquidity,
                    economy=economy,
                )
            )
    return holdings


def build_government_grid(economy: str = "GBP") -> list[BondHoldingConfig]:
    """Government bond grid for an economy.

    Args:
        economy: Currency / economy identifier.

    Returns:
        Grid of government bond holding configs.
    """
    return build_bond_grid(
        bond_type=BondType.GOVERNMENT,
        economy=economy,
    )


def build_corporate_grid(
    economy: str = "GBP",
    credit_class: CreditClass = CreditClass.A,
    liquidity: Liquidity = Liquidity.HIGH,
) -> list[BondHoldingConfig]:
    """Corporate bond grid for an economy / credit class.

    Args:
        economy: Currency / economy identifier.
        credit_class: Credit rating.
        liquidity: Liquidity tier.

    Returns:
        Grid of corporate bond holding configs.
    """
    return build_bond_grid(
        bond_type=BondType.CORPORATE,
        credit_class=credit_class,
        liquidity=liquidity,
        economy=economy,
    )


def build_index_linked_grid(economy: str = "GBP") -> list[BondHoldingConfig]:
    """Index-linked government bond grid.

    Args:
        economy: Currency / economy identifier.

    Returns:
        Grid of index-linked bond holding configs.
    """
    return build_bond_grid(
        bond_type=BondType.INDEX_LINKED,
        economy=economy,
    )


__all__ = [
    "STANDARD_COUPONS",
    "STANDARD_TENORS",
    "build_bond_grid",
    "build_corporate_grid",
    "build_government_grid",
    "build_index_linked_grid",
]
