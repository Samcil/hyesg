"""Minor calibration parameter loading.

Maps the ~40 named ranges read by the C# ``ParameterReader`` into
validated Pydantic schemas. This is the Python equivalent of
``SimulationSetups.Shared.ParameterReader.Read()``.

Named ranges follow the C# convention::

    {category}_{SubCategory}_Keys   → string array of asset names
    {category}_{SubCategory}_Values → float array of values
    {Currency}_Maturities           → float array of curve tenors
    {Currency}_Rates                → float array of spot rates
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from hyesg.market_data.calibration_reader import CalibrationDataReader

logger = logging.getLogger(__name__)


# ── Registry of all known named ranges ───────────────────────────

NAMED_RANGES: dict[str, str] = {
    # Simulation date
    "simulationDate": "Simulation date (DateTime cell)",
    # Equity market data (key-value pairs)
    "equityInitialDividendYields_Keys": "Equity asset names",
    "equityInitialDividendYields_Values": "Equity initial dividend yields",
    "equityUnadjustedInitialVolatilities_Keys": "Equity vol asset names",
    "equityUnadjustedInitialVolatilities_Values": "Equity unadjusted vols",
    # Exchange rates
    "exchangeRatesInitialGbpPerUnitForeignRates_Keys": "FX pair names",
    "exchangeRatesInitialGbpPerUnitForeignRates_Values": "FX spot rates",
    "exchangeRatesUnadjustedInitialVolatilities_Keys": "FX vol pair names",
    "exchangeRatesUnadjustedInitialVolatilities_Values": "FX unadjusted vols",
    # Property
    "propertyInitialDividendYields_Keys": "Property type names",
    "propertyInitialDividendYields_Values": "Property dividend yields",
    "propertyUnadjustedInitialVolatilities_Keys": "Property vol names",
    "propertyUnadjustedInitialVolatilities_Values": "Property unadjusted vols",
    # Other equity (REITs, infrastructure, etc.)
    "otherEquityInitialDividendYields_Keys": "Other equity names",
    "otherEquityInitialDividendYields_Values": "Other equity dividend yields",
    "otherEquityUnadjustedInitialVolatilities_Keys": "Other equity vol names",
    "otherEquityUnadjustedInitialVolatilities_Values": "Other equity vols",
    # Foreign economy yield curves
    "USD_Maturities": "USD curve maturities",
    "USD_Rates": "USD spot rates",
    "EUR_Maturities": "EUR curve maturities",
    "EUR_Rates": "EUR spot rates",
    "JPY_Maturities": "JPY curve maturities",
    "JPY_Rates": "JPY spot rates",
    # RPI reform
    "expectedCtsFwdRpiSwapsAtReform": "Expected cts fwd RPI swaps at reform",
    "expectedCtsFwdRpiGiltsAtReform": "Expected cts fwd RPI gilts at reform",
    # Market implied RPI-CPIH wedges
    "marketImpliedSwapsRpiCpihWedge": "Market implied swaps RPI-CPIH wedge",
    "marketImpliedGiltsRpiCpihWedge": "Market implied gilts RPI-CPIH wedge",
    # Credit parameters
    "creditParameters": "Credit model parameters (7 values)",
}
"""Complete registry of named ranges used by the C# ESG minor calibration."""

# Currencies with independent yield curve data in the calibration spreadsheet
FOREIGN_CURVE_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "JPY")


# ── Validated parameter schemas ──────────────────────────────────


class KeyValuePairs(BaseModel):
    """A validated set of key-value pairs from named ranges.

    Attributes:
        keys: Asset/category names.
        values: Corresponding numeric values.
    """

    model_config = ConfigDict(frozen=True)

    keys: tuple[str, ...]
    values: tuple[float, ...]

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v: object) -> tuple[float, ...]:
        if isinstance(v, np.ndarray):
            return tuple(float(x) for x in v)
        if isinstance(v, (list, tuple)):
            return tuple(float(x) for x in v)
        raise ValueError(f"Expected array-like, got {type(v)}")

    @field_validator("keys", mode="before")
    @classmethod
    def _coerce_keys(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, np.ndarray):
            return tuple(str(x) for x in v)
        if isinstance(v, (list, tuple)):
            return tuple(str(x) for x in v)
        raise ValueError(f"Expected array-like, got {type(v)}")

    def to_dict(self) -> dict[str, float]:
        """Convert to a standard dictionary.

        Returns:
            Mapping of key → value.

        Raises:
            ValueError: If keys and values have different lengths.
        """
        if len(self.keys) != len(self.values):
            raise ValueError(
                f"Length mismatch: {len(self.keys)} keys, "
                f"{len(self.values)} values"
            )
        return dict(zip(self.keys, self.values, strict=True))


class ForeignCurveData(BaseModel):
    """Yield curve data for a foreign economy.

    Attributes:
        maturities: Tenor points in years.
        rates: Continuously compounded spot rates.
    """

    model_config = ConfigDict(frozen=True)

    maturities: tuple[float, ...]
    rates: tuple[float, ...]

    @field_validator("maturities", "rates", mode="before")
    @classmethod
    def _coerce_array(cls, v: object) -> tuple[float, ...]:
        if isinstance(v, np.ndarray):
            return tuple(float(x) for x in v)
        if isinstance(v, (list, tuple)):
            return tuple(float(x) for x in v)
        raise ValueError(f"Expected array-like, got {type(v)}")


class MinorCalibrationData(BaseModel):
    """Complete minor calibration data loaded from named ranges.

    This is the Python equivalent of C#
    ``MinorCalibrationParameters``. All fields are validated on
    construction.

    Attributes:
        simulation_date: Valuation date for the calibration.
        equity_dividend_yields: Equity name → initial dividend yield.
        equity_volatilities: Equity name → unadjusted initial vol.
        fx_rates: FX pair → GBP per unit foreign rate.
        fx_volatilities: FX pair → unadjusted initial vol.
        property_dividend_yields: Property type → initial dividend yield.
        property_volatilities: Property type → unadjusted initial vol.
        other_equity_dividend_yields: Other equity → dividend yield.
        other_equity_volatilities: Other equity → unadjusted vol.
        foreign_curves: Currency → (maturities, rates) for nominal curves.
        expected_cts_fwds_at_reform: {Swaps, Gilts} → expected cts fwd.
        market_implied_rpi_cpih_wedges: {Swaps, Gilts} → wedge values.
        credit_parameters: Credit model parameters (7-element vector).
    """

    model_config = ConfigDict(frozen=True)

    simulation_date: datetime
    equity_dividend_yields: dict[str, float] = Field(default_factory=dict)
    equity_volatilities: dict[str, float] = Field(default_factory=dict)
    fx_rates: dict[str, float] = Field(default_factory=dict)
    fx_volatilities: dict[str, float] = Field(default_factory=dict)
    property_dividend_yields: dict[str, float] = Field(default_factory=dict)
    property_volatilities: dict[str, float] = Field(default_factory=dict)
    other_equity_dividend_yields: dict[str, float] = Field(
        default_factory=dict
    )
    other_equity_volatilities: dict[str, float] = Field(default_factory=dict)
    foreign_curves: dict[str, ForeignCurveData] = Field(default_factory=dict)
    expected_cts_fwds_at_reform: dict[str, float] = Field(
        default_factory=dict
    )
    market_implied_rpi_cpih_wedges: dict[str, float] = Field(
        default_factory=dict
    )
    credit_parameters: tuple[float, ...] = ()

    @field_validator("credit_parameters", mode="before")
    @classmethod
    def _coerce_credit(cls, v: object) -> tuple[float, ...]:
        if isinstance(v, np.ndarray):
            return tuple(float(x) for x in v)
        if isinstance(v, (list, tuple)):
            return tuple(float(x) for x in v)
        raise ValueError(f"Expected array-like, got {type(v)}")


# ── Loading function ─────────────────────────────────────────────


def _read_key_value_pairs(
    reader: CalibrationDataReader,
    keys_range: str,
    values_range: str,
) -> dict[str, float]:
    """Read a Keys/Values named range pair into a dictionary.

    Args:
        reader: Calibration data reader.
        keys_range: Named range for string keys.
        values_range: Named range for float values.

    Returns:
        Dictionary mapping keys to values.
    """
    keys = reader.read_named_range(keys_range)
    values = reader.read_named_range(values_range)
    kv = KeyValuePairs(keys=keys, values=values)
    return kv.to_dict()


def load_minor_calibration(
    reader: CalibrationDataReader,
) -> MinorCalibrationData:
    """Load the full set of minor calibration parameters.

    Reads all ~40 named ranges from the reader and validates them
    into a :class:`MinorCalibrationData` schema.

    Args:
        reader: Any implementation of :class:`CalibrationDataReader`
            (Excel, JSON, or CSV directory).

    Returns:
        Validated minor calibration data.

    Raises:
        KeyError: If required named ranges are missing.
        ValueError: If loaded data fails validation.
    """
    logger.info("Loading minor calibration parameters...")

    # Simulation date
    sim_date_arr = reader.read_named_range("simulationDate")
    if sim_date_arr.dtype.kind in ("U", "S", "O"):
        sim_date = datetime.fromisoformat(str(sim_date_arr.flat[0]))
    else:
        # Excel serial date — openpyxl may return datetime directly
        val = sim_date_arr.flat[0]
        if isinstance(val, datetime):
            sim_date = val
        else:
            sim_date = datetime.fromisoformat(str(val))

    # Key-value pair named ranges
    equity_dy = _read_key_value_pairs(
        reader,
        "equityInitialDividendYields_Keys",
        "equityInitialDividendYields_Values",
    )
    equity_vol = _read_key_value_pairs(
        reader,
        "equityUnadjustedInitialVolatilities_Keys",
        "equityUnadjustedInitialVolatilities_Values",
    )
    fx_rates = _read_key_value_pairs(
        reader,
        "exchangeRatesInitialGbpPerUnitForeignRates_Keys",
        "exchangeRatesInitialGbpPerUnitForeignRates_Values",
    )
    fx_vol = _read_key_value_pairs(
        reader,
        "exchangeRatesUnadjustedInitialVolatilities_Keys",
        "exchangeRatesUnadjustedInitialVolatilities_Values",
    )
    prop_dy = _read_key_value_pairs(
        reader,
        "propertyInitialDividendYields_Keys",
        "propertyInitialDividendYields_Values",
    )
    prop_vol = _read_key_value_pairs(
        reader,
        "propertyUnadjustedInitialVolatilities_Keys",
        "propertyUnadjustedInitialVolatilities_Values",
    )
    other_eq_dy = _read_key_value_pairs(
        reader,
        "otherEquityInitialDividendYields_Keys",
        "otherEquityInitialDividendYields_Values",
    )
    other_eq_vol = _read_key_value_pairs(
        reader,
        "otherEquityUnadjustedInitialVolatilities_Keys",
        "otherEquityUnadjustedInitialVolatilities_Values",
    )

    # Foreign economy curves
    foreign_curves: dict[str, ForeignCurveData] = {}
    for ccy in FOREIGN_CURVE_CURRENCIES:
        mats = reader.read_named_range(f"{ccy}_Maturities")
        rates = reader.read_named_range(f"{ccy}_Rates")
        foreign_curves[ccy] = ForeignCurveData(
            maturities=mats, rates=rates
        )

    # Scalar named ranges for RPI reform
    expected_cts_fwds = {
        "Swaps": reader.read_scalar("expectedCtsFwdRpiSwapsAtReform"),
        "Gilts": reader.read_scalar("expectedCtsFwdRpiGiltsAtReform"),
    }

    # Market implied RPI-CPIH wedges
    rpi_cpih_wedges = {
        "Swaps": reader.read_scalar("marketImpliedSwapsRpiCpihWedge"),
        "Gilts": reader.read_scalar("marketImpliedGiltsRpiCpihWedge"),
    }

    # Credit parameters
    credit_params = reader.read_named_range("creditParameters")

    result = MinorCalibrationData(
        simulation_date=sim_date,
        equity_dividend_yields=equity_dy,
        equity_volatilities=equity_vol,
        fx_rates=fx_rates,
        fx_volatilities=fx_vol,
        property_dividend_yields=prop_dy,
        property_volatilities=prop_vol,
        other_equity_dividend_yields=other_eq_dy,
        other_equity_volatilities=other_eq_vol,
        foreign_curves=foreign_curves,
        expected_cts_fwds_at_reform=expected_cts_fwds,
        market_implied_rpi_cpih_wedges=rpi_cpih_wedges,
        credit_parameters=credit_params,
    )

    logger.info(
        "Loaded minor calibration: %d equities, %d FX pairs, "
        "%d property types, %d foreign curves",
        len(equity_dy),
        len(fx_rates),
        len(prop_dy),
        len(foreign_curves),
    )
    return result
