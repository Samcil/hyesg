"""Fee-adjusted fund wrapper definitions.

Defines net-of-fees variants for gross funds in the catalogue,
matching the C# ESG fee deduction structure.
"""

from __future__ import annotations

from hyesg.config.funds import NetOfFeesFund

# ---------------------------------------------------------------------------
# Net-of-fees wrappers (20+ fee-adjusted variants)
# ---------------------------------------------------------------------------

FEE_WRAPPERS: list[NetOfFeesFund] = [
    # World Equity net-of-fees
    NetOfFeesFund(
        gross_fund="Dev World GBP Unhedged",
        fee_bps=30.0,
        label="Dev World GBP Unhedged Net 30bps",
    ),
    NetOfFeesFund(
        gross_fund="Dev World GBP Hedged",
        fee_bps=30.0,
        label="Dev World GBP Hedged Net 30bps",
    ),
    NetOfFeesFund(
        gross_fund="All World GBP Unhedged",
        fee_bps=35.0,
        label="All World GBP Unhedged Net 35bps",
    ),
    NetOfFeesFund(
        gross_fund="All World GBP Hedged",
        fee_bps=35.0,
        label="All World GBP Hedged Net 35bps",
    ),
    # High Yield net-of-fees
    NetOfFeesFund(
        gross_fund="US High Yield Unhedged",
        fee_bps=50.0,
        label="US High Yield Unhedged Net 50bps",
    ),
    NetOfFeesFund(
        gross_fund="US High Yield Hedged",
        fee_bps=50.0,
        label="US High Yield Hedged Net 50bps",
    ),
    NetOfFeesFund(
        gross_fund="EUR High Yield Unhedged",
        fee_bps=50.0,
        label="EUR High Yield Unhedged Net 50bps",
    ),
    NetOfFeesFund(
        gross_fund="EUR High Yield Hedged",
        fee_bps=50.0,
        label="EUR High Yield Hedged Net 50bps",
    ),
    NetOfFeesFund(
        gross_fund="Global High Yield Unhedged",
        fee_bps=55.0,
        label="Global High Yield Unhedged Net 55bps",
    ),
    NetOfFeesFund(
        gross_fund="Global High Yield Hedged",
        fee_bps=55.0,
        label="Global High Yield Hedged Net 55bps",
    ),
    # Distressed Debt net-of-fees
    NetOfFeesFund(
        gross_fund="US Distressed Debt Unhedged",
        fee_bps=75.0,
        label="US Distressed Debt Unhedged Net 75bps",
    ),
    NetOfFeesFund(
        gross_fund="Global Distressed Debt Hedged",
        fee_bps=75.0,
        label="Global Distressed Debt Hedged Net 75bps",
    ),
    # Private Equity net-of-fees
    NetOfFeesFund(
        gross_fund="UK Private Equity",
        fee_bps=200.0,
        label="UK Private Equity Net 200bps",
    ),
    NetOfFeesFund(
        gross_fund="US Private Equity",
        fee_bps=200.0,
        label="US Private Equity Net 200bps",
    ),
    NetOfFeesFund(
        gross_fund="Global Private Equity",
        fee_bps=200.0,
        label="Global Private Equity Net 200bps",
    ),
    # Direct Lending net-of-fees
    NetOfFeesFund(
        gross_fund="UK Direct Lending",
        fee_bps=100.0,
        label="UK Direct Lending Net 100bps",
    ),
    NetOfFeesFund(
        gross_fund="US Direct Lending",
        fee_bps=100.0,
        label="US Direct Lending Net 100bps",
    ),
    # Property net-of-fees
    NetOfFeesFund(
        gross_fund="UK Direct Property",
        fee_bps=50.0,
        label="UK Direct Property Net 50bps",
    ),
    NetOfFeesFund(
        gross_fund="UK Property Unit Trust",
        fee_bps=60.0,
        label="UK Property Unit Trust Net 60bps",
    ),
    # DGF net-of-fees
    NetOfFeesFund(
        gross_fund="DGF Tier 1",
        fee_bps=75.0,
        label="DGF Tier 1 Net 75bps",
    ),
    NetOfFeesFund(
        gross_fund="DGF Tier 2",
        fee_bps=60.0,
        label="DGF Tier 2 Net 60bps",
    ),
    NetOfFeesFund(
        gross_fund="DGF Tier 3",
        fee_bps=45.0,
        label="DGF Tier 3 Net 45bps",
    ),
    # Absolute Return net-of-fees
    NetOfFeesFund(
        gross_fund="Absolute Return",
        fee_bps=100.0,
        label="Absolute Return Net 100bps",
    ),
    # EMD net-of-fees
    NetOfFeesFund(
        gross_fund="EMD Blend",
        fee_bps=60.0,
        label="EMD Blend Net 60bps",
    ),
]


def get_fee_wrappers() -> list[NetOfFeesFund]:
    """Return a copy of all defined net-of-fees wrappers."""
    return list(FEE_WRAPPERS)


__all__ = [
    "FEE_WRAPPERS",
    "get_fee_wrappers",
]
