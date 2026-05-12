"""Portfolio aggregation module for post-processing simulation results."""

from __future__ import annotations

from hyesg.models.portfolio.analytics import PortfolioAnalytics
from hyesg.models.portfolio.holdings import (
    BondHolding,
    CashHolding,
    CDSHolding,
    DFRNHolding,
    EquityHolding,
    ForwardHolding,
    FundHolding,
    SwapHolding,
)
from hyesg.models.portfolio.portfolio import Portfolio
from hyesg.models.portfolio.rebalancer import AllocationRebalancer
from hyesg.models.portfolio.result import PortfolioConfig, PortfolioResult
from hyesg.models.portfolio.types import (
    BondPortfolio,
    CurrencyHedgePortfolio,
    DerivativePortfolio,
    EquityPortfolio,
    FundPortfolio,
    PortfolioOfPortfolios,
    PortfolioProtocol,
)

__all__ = [
    "AllocationRebalancer",
    "BondHolding",
    "BondPortfolio",
    "CashHolding",
    "CDSHolding",
    "CurrencyHedgePortfolio",
    "DerivativePortfolio",
    "DFRNHolding",
    "EquityHolding",
    "EquityPortfolio",
    "ForwardHolding",
    "FundHolding",
    "FundPortfolio",
    "Portfolio",
    "PortfolioAnalytics",
    "PortfolioConfig",
    "PortfolioOfPortfolios",
    "PortfolioProtocol",
    "PortfolioResult",
    "SwapHolding",
]
