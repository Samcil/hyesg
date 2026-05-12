"""Market data framework for hyesg.

Provides protocols, file-based providers, snapshots, and
transform utilities for loading and preparing market data
for ESG simulations.
"""

from __future__ import annotations

from hyesg.market_data.file_provider import FileMarketData, MarketDataError
from hyesg.market_data.protocols import MarketDataProvider
from hyesg.market_data.readers import read_correlation_csv, read_yield_curve_csv
from hyesg.market_data.snapshot import MarketDataSnapshot
from hyesg.market_data.transforms import to_simulation_curves

__all__ = [
    "FileMarketData",
    "MarketDataError",
    "MarketDataProvider",
    "MarketDataSnapshot",
    "read_correlation_csv",
    "read_yield_curve_csv",
    "to_simulation_curves",
]