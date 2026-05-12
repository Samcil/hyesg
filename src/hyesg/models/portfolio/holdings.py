"""Holding types for portfolio construction.

Each holding is an immutable ``NamedTuple`` describing a single position
within a portfolio.
"""

from __future__ import annotations

from typing import NamedTuple


class EquityHolding(NamedTuple):
    """A weighted equity position.

    Attributes:
        asset_label: Model name / identifier for the equity asset.
        weight: Portfolio weight (fraction of total).
        initial_price: Starting price, defaults to 1.0.
    """

    asset_label: str
    weight: float
    initial_price: float = 1.0


class BondHolding(NamedTuple):
    """A weighted fixed-income position.

    Attributes:
        asset_label: Model name / identifier for the bond.
        weight: Portfolio weight.
        face: Face / par value.
        coupon: Annual coupon rate (e.g. 0.05 for 5%).
        maturity: Time to maturity in years.
        freq: Coupon payments per year (default 2 = semi-annual).
    """

    asset_label: str
    weight: float
    face: float
    coupon: float
    maturity: float
    freq: int = 2


class CashHolding(NamedTuple):
    """A cash / money-market position.

    Attributes:
        weight: Portfolio weight.
    """

    weight: float


class ForwardHolding(NamedTuple):
    """A forward contract position.

    Attributes:
        asset_label: Underlying asset model name.
        weight: Portfolio weight.
        delivery_date: Delivery date in years.
    """

    asset_label: str
    weight: float
    delivery_date: float


class SwapHolding(NamedTuple):
    """An interest-rate swap position.

    Attributes:
        fixed_rate: Fixed leg rate.
        notional: Swap notional.
        maturity: Swap maturity in years.
    """

    fixed_rate: float
    notional: float
    maturity: float


class CDSHolding(NamedTuple):
    """A credit default swap position.

    Attributes:
        reference_label: Reference entity model name.
        spread: CDS spread.
        notional: CDS notional.
        maturity: CDS maturity in years.
    """

    reference_label: str
    spread: float
    notional: float
    maturity: float


class DFRNHolding(NamedTuple):
    """A defaultable floating rate note position.

    Attributes:
        reference_label: Reference entity model name.
        spread: Spread over reference rate.
        notional: Note notional.
    """

    reference_label: str
    spread: float
    notional: float


class FundHolding(NamedTuple):
    """A fund-of-funds position referencing another portfolio.

    Attributes:
        fund_ref: Identifier for the referenced fund / portfolio.
        weight: Portfolio weight.
    """

    fund_ref: str
    weight: float
