"""Tests for minor calibration data loading."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from hyesg.market_data.calibration_reader import JsonCalibrationReader
from hyesg.market_data.minor_calibration import (
    ForeignCurveData,
    KeyValuePairs,
    MinorCalibrationData,
    load_minor_calibration,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Fixtures ─────────────────────────────────────────────────────


def _make_full_json(tmp_path: Path) -> Path:
    """Create a complete calibration JSON for integration testing."""
    data = {
        "simulationDate": "2025-01-09",
        "equityInitialDividendYields_Keys": [
            "UK Equity",
            "World ex UK Equity",
        ],
        "equityInitialDividendYields_Values": [0.0365, 0.018],
        "equityUnadjustedInitialVolatilities_Keys": [
            "UK Equity",
            "World ex UK Equity",
        ],
        "equityUnadjustedInitialVolatilities_Values": [0.155, 0.14],
        "exchangeRatesInitialGbpPerUnitForeignRates_Keys": ["USD", "EUR"],
        "exchangeRatesInitialGbpPerUnitForeignRates_Values": [0.79, 0.84],
        "exchangeRatesUnadjustedInitialVolatilities_Keys": ["USD", "EUR"],
        "exchangeRatesUnadjustedInitialVolatilities_Values": [0.08, 0.07],
        "propertyInitialDividendYields_Keys": ["Commercial", "PRS"],
        "propertyInitialDividendYields_Values": [0.055, 0.045],
        "propertyUnadjustedInitialVolatilities_Keys": [
            "Commercial",
            "PRS",
        ],
        "propertyUnadjustedInitialVolatilities_Values": [0.12, 0.10],
        "otherEquityInitialDividendYields_Keys": [
            "Private Equity",
            "Commodities",
        ],
        "otherEquityInitialDividendYields_Values": [0.0, 0.0],
        "otherEquityUnadjustedInitialVolatilities_Keys": [
            "Private Equity",
            "Commodities",
        ],
        "otherEquityUnadjustedInitialVolatilities_Values": [0.22, 0.16],
        "USD_Maturities": [1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
        "USD_Rates": [0.05, 0.048, 0.04, 0.038, 0.036, 0.035],
        "EUR_Maturities": [1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
        "EUR_Rates": [0.035, 0.034, 0.032, 0.030, 0.028, 0.027],
        "JPY_Maturities": [1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
        "JPY_Rates": [0.002, 0.003, 0.007, 0.010, 0.013, 0.015],
        "expectedCtsFwdRpiSwapsAtReform": 0.035,
        "expectedCtsFwdRpiGiltsAtReform": 0.032,
        "marketImpliedSwapsRpiCpihWedge": 0.01,
        "marketImpliedGiltsRpiCpihWedge": 0.008,
        "creditParameters": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
    }
    json_path = tmp_path / "full_calibration.json"
    json_path.write_text(json.dumps(data, indent=2))
    return json_path


@pytest.fixture()
def full_json_path(tmp_path: Path) -> Path:
    """Fixture returning path to a complete calibration JSON."""
    return _make_full_json(tmp_path)


# ── Unit tests ───────────────────────────────────────────────────


class TestKeyValuePairs:
    """Test KeyValuePairs validation model."""

    def test_basic(self) -> None:
        kv = KeyValuePairs(keys=("A", "B"), values=(1.0, 2.0))
        assert kv.to_dict() == {"A": 1.0, "B": 2.0}

    def test_from_numpy(self) -> None:
        kv = KeyValuePairs(
            keys=np.array(["X", "Y"]),
            values=np.array([3.14, 2.72]),
        )
        d = kv.to_dict()
        assert d["X"] == pytest.approx(3.14)
        assert d["Y"] == pytest.approx(2.72)

    def test_frozen(self) -> None:
        kv = KeyValuePairs(keys=("A",), values=(1.0,))
        with pytest.raises(ValidationError):
            kv.keys = ("B",)  # type: ignore[misc]


class TestForeignCurveData:
    """Test ForeignCurveData validation."""

    def test_basic(self) -> None:
        fc = ForeignCurveData(
            maturities=(1.0, 5.0, 10.0), rates=(0.01, 0.02, 0.03)
        )
        assert len(fc.maturities) == 3
        assert fc.rates[2] == pytest.approx(0.03)

    def test_from_numpy(self) -> None:
        fc = ForeignCurveData(
            maturities=np.array([1.0, 2.0]),
            rates=np.array([0.05, 0.04]),
        )
        assert len(fc.maturities) == 2


class TestMinorCalibrationData:
    """Test MinorCalibrationData validation."""

    def test_minimal(self) -> None:
        data = MinorCalibrationData(
            simulation_date=datetime(2025, 1, 9),
            credit_parameters=(0.01, 0.02, 0.03),
        )
        assert data.simulation_date.year == 2025
        assert len(data.credit_parameters) == 3

    def test_credit_from_numpy(self) -> None:
        data = MinorCalibrationData(
            simulation_date=datetime(2025, 1, 9),
            credit_parameters=np.array([0.1, 0.2, 0.3]),
        )
        assert data.credit_parameters == (0.1, 0.2, 0.3)


# ── Integration test ─────────────────────────────────────────────


class TestLoadMinorCalibration:
    """Integration test: load full calibration from JSON."""

    def test_load_complete(self, full_json_path: Path) -> None:
        """Load a complete calibration JSON and verify all fields."""
        reader = JsonCalibrationReader(full_json_path)
        result = load_minor_calibration(reader)

        # Simulation date
        assert result.simulation_date == datetime(2025, 1, 9)

        # Equity
        assert len(result.equity_dividend_yields) == 2
        assert result.equity_dividend_yields["UK Equity"] == pytest.approx(
            0.0365
        )
        assert len(result.equity_volatilities) == 2
        assert result.equity_volatilities["UK Equity"] == pytest.approx(0.155)

        # FX
        assert len(result.fx_rates) == 2
        assert result.fx_rates["USD"] == pytest.approx(0.79)
        assert len(result.fx_volatilities) == 2

        # Property
        assert len(result.property_dividend_yields) == 2
        assert result.property_dividend_yields["Commercial"] == pytest.approx(
            0.055
        )

        # Other equity
        assert len(result.other_equity_dividend_yields) == 2
        assert len(result.other_equity_volatilities) == 2

        # Foreign curves
        assert len(result.foreign_curves) == 3
        assert "USD" in result.foreign_curves
        assert len(result.foreign_curves["USD"].maturities) == 6
        assert len(result.foreign_curves["USD"].rates) == 6

        # RPI reform
        assert result.expected_cts_fwds_at_reform["Swaps"] == pytest.approx(
            0.035
        )
        assert result.expected_cts_fwds_at_reform["Gilts"] == pytest.approx(
            0.032
        )

        # RPI-CPIH wedges
        assert result.market_implied_rpi_cpih_wedges[
            "Swaps"
        ] == pytest.approx(0.01)

        # Credit
        assert len(result.credit_parameters) == 7
        assert result.credit_parameters[0] == pytest.approx(0.01)

    def test_missing_range_raises(self, tmp_path: Path) -> None:
        """Missing named range raises KeyError."""
        # Create minimal JSON missing most ranges
        data = {"simulationDate": "2025-01-09"}
        path = tmp_path / "partial.json"
        path.write_text(json.dumps(data))
        reader = JsonCalibrationReader(path)
        with pytest.raises(KeyError):
            load_minor_calibration(reader)
