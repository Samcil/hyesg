"""Market data framework for hyesg.

Provides protocols, file-based providers, snapshots, and
transform utilities for loading and preparing market data
for ESG simulations.  Also includes calibration data readers
for loading minor calibration parameters from Excel, JSON,
or CSV sources.
"""

from __future__ import annotations

from hyesg.market_data.calibration_reader import (
    CalibrationDataReader,
    CsvDirectoryCalibrationReader,
    ExcelCalibrationReader,
    JsonCalibrationReader,
)
from hyesg.market_data.constants import (
    AGGREGATE_EQUITY,
    ALTERNATIVES_MPR,
    PROPERTY_MPR,
    SALARY_WEDGE,
    AggregateEquityParams,
    AlternativesMarketPricesOfRisk,
    PropertyMarketPricesOfRisk,
    SalaryWedgeConstants,
    get_market_price_of_risk,
)
from hyesg.market_data.file_provider import FileMarketData, MarketDataError
from hyesg.market_data.minor_calibration import (
    NAMED_RANGES,
    MinorCalibrationData,
    load_minor_calibration,
)
from hyesg.market_data.protocols import MarketDataProvider
from hyesg.market_data.readers import read_correlation_csv, read_yield_curve_csv
from hyesg.market_data.snapshot import MarketDataSnapshot
from hyesg.market_data.transforms import to_simulation_curves

__all__ = [
    # Calibration reader protocol + implementations
    "CalibrationDataReader",
    "CsvDirectoryCalibrationReader",
    "ExcelCalibrationReader",
    "JsonCalibrationReader",
    # Constants
    "AGGREGATE_EQUITY",
    "ALTERNATIVES_MPR",
    "AggregateEquityParams",
    "AlternativesMarketPricesOfRisk",
    "NAMED_RANGES",
    "PROPERTY_MPR",
    "PropertyMarketPricesOfRisk",
    "SALARY_WEDGE",
    "SalaryWedgeConstants",
    "get_market_price_of_risk",
    # Minor calibration
    "MinorCalibrationData",
    "load_minor_calibration",
    # Market data
    "FileMarketData",
    "MarketDataError",
    "MarketDataProvider",
    "MarketDataSnapshot",
    "read_correlation_csv",
    "read_yield_curve_csv",
    "to_simulation_curves",
]