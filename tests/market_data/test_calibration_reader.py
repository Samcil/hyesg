"""Tests for calibration data reader implementations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from hyesg.market_data.calibration_reader import (
    CalibrationDataReader,
    CsvDirectoryCalibrationReader,
    JsonCalibrationReader,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def sample_calibration_json(tmp_path: Path) -> Path:
    """Create a sample calibration JSON file."""
    data = {
        "simulationDate": "2025-01-09",
        "equityInitialDividendYields_Keys": [
            "UK Equity",
            "World ex UK Equity",
            "Emerging Market Equity",
            "UK Property",
            "Global REITs",
            "Listed Infra",
            "Unlisted Infra",
            "Private Equity",
        ],
        "equityInitialDividendYields_Values": [
            0.0365,
            0.018,
            0.026,
            0.048,
            0.035,
            0.032,
            0.045,
            0.0,
        ],
        "equityUnadjustedInitialVolatilities_Keys": [
            "UK Equity",
            "World ex UK Equity",
            "Emerging Market Equity",
            "APAC ex Japan Equity",
            "Private Equity",
            "Commodities",
        ],
        "equityUnadjustedInitialVolatilities_Values": [
            0.155,
            0.14,
            0.20,
            0.18,
            0.22,
            0.16,
        ],
        "exchangeRatesInitialGbpPerUnitForeignRates_Keys": [
            "USD",
            "EUR",
            "JPY",
            "EM",
            "APAC",
        ],
        "exchangeRatesInitialGbpPerUnitForeignRates_Values": [
            0.79,
            0.84,
            0.0053,
            0.70,
            0.72,
        ],
        "exchangeRatesUnadjustedInitialVolatilities_Keys": [
            "USD",
            "EUR",
            "JPY",
            "EM",
            "APAC",
        ],
        "exchangeRatesUnadjustedInitialVolatilities_Values": [
            0.08,
            0.07,
            0.10,
            0.12,
            0.09,
        ],
        "propertyInitialDividendYields_Keys": [
            "Commercial",
            "PRS",
            "Long Lease",
            "Social",
            "REITs",
            "Listed Infra",
        ],
        "propertyInitialDividendYields_Values": [
            0.055,
            0.045,
            0.040,
            0.048,
            0.035,
            0.032,
        ],
        "propertyUnadjustedInitialVolatilities_Keys": [
            "Commercial",
            "PRS",
            "Long Lease",
            "Social",
            "REITs",
            "Listed Infra",
        ],
        "propertyUnadjustedInitialVolatilities_Values": [
            0.12,
            0.10,
            0.08,
            0.09,
            0.14,
            0.11,
        ],
        "otherEquityInitialDividendYields_Keys": [
            "Private Equity",
            "Commodities",
            "Unlisted Infra",
            "Hedge Funds",
        ],
        "otherEquityInitialDividendYields_Values": [
            0.0,
            0.0,
            0.045,
            0.01,
        ],
        "otherEquityUnadjustedInitialVolatilities_Keys": [
            "Private Equity",
            "Commodities",
            "Unlisted Infra",
            "Hedge Funds",
        ],
        "otherEquityUnadjustedInitialVolatilities_Values": [
            0.22,
            0.16,
            0.15,
            0.06,
        ],
        "USD_Maturities": [
            0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0,
        ],
        "USD_Rates": [
            0.052, 0.050, 0.048, 0.045, 0.043, 0.040, 0.039, 0.038,
            0.037, 0.036, 0.035, 0.034,
        ],
        "EUR_Maturities": list(range(1, 34)),
        "EUR_Rates": [0.035 - 0.0003 * i for i in range(33)],
        "JPY_Maturities": [
            0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0,
            25.0, 30.0, 40.0, 50.0, 60.0,
        ],
        "JPY_Rates": [
            0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.009, 0.010,
            0.012, 0.013, 0.014, 0.015, 0.016, 0.016, 0.016,
        ],
        "expectedCtsFwdRpiSwapsAtReform": 0.035,
        "expectedCtsFwdRpiGiltsAtReform": 0.032,
        "marketImpliedSwapsRpiCpihWedge": 0.01,
        "marketImpliedGiltsRpiCpihWedge": 0.008,
        "creditParameters": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
    }
    json_path = tmp_path / "calibration.json"
    json_path.write_text(json.dumps(data, indent=2))
    return json_path


@pytest.fixture()
def sample_csv_dir(tmp_path: Path) -> Path:
    """Create a sample calibration CSV directory."""
    csv_dir = tmp_path / "csv_calib"
    csv_dir.mkdir()

    # Single-value CSV
    (csv_dir / "expectedCtsFwdRpiSwapsAtReform.csv").write_text("0.035\n")
    (csv_dir / "expectedCtsFwdRpiGiltsAtReform.csv").write_text("0.032\n")
    (csv_dir / "marketImpliedSwapsRpiCpihWedge.csv").write_text("0.01\n")
    (csv_dir / "marketImpliedGiltsRpiCpihWedge.csv").write_text("0.008\n")

    # Multi-value CSV
    (csv_dir / "creditParameters.csv").write_text(
        "0.01,0.02,0.03,0.04,0.05,0.06,0.07\n"
    )

    # Key-value pair CSVs
    (csv_dir / "equityInitialDividendYields_Keys.csv").write_text(
        "UK Equity,World ex UK\n"
    )
    (csv_dir / "equityInitialDividendYields_Values.csv").write_text(
        "0.0365,0.018\n"
    )

    # Simulation date
    (csv_dir / "simulationDate.csv").write_text("2025-01-09\n")

    return csv_dir


# ── JSON reader tests ────────────────────────────────────────────


class TestJsonCalibrationReader:
    """Tests for JsonCalibrationReader."""

    def test_protocol_compliance(self, sample_calibration_json: Path) -> None:
        """Reader satisfies CalibrationDataReader protocol."""
        reader = JsonCalibrationReader(sample_calibration_json)
        assert isinstance(reader, CalibrationDataReader)

    def test_read_scalar(self, sample_calibration_json: Path) -> None:
        """Read a scalar value."""
        reader = JsonCalibrationReader(sample_calibration_json)
        val = reader.read_scalar("expectedCtsFwdRpiSwapsAtReform")
        assert val == pytest.approx(0.035)

    def test_read_1d_array(self, sample_calibration_json: Path) -> None:
        """Read a 1D numeric array."""
        reader = JsonCalibrationReader(sample_calibration_json)
        arr = reader.read_named_range("creditParameters")
        assert arr.shape == (7,)
        assert arr[0] == pytest.approx(0.01)
        assert arr[6] == pytest.approx(0.07)

    def test_read_string_array(self, sample_calibration_json: Path) -> None:
        """Read a string array."""
        reader = JsonCalibrationReader(sample_calibration_json)
        arr = reader.read_named_range("equityInitialDividendYields_Keys")
        assert len(arr) == 8
        assert arr[0] == "UK Equity"

    def test_read_named_ranges_batch(
        self, sample_calibration_json: Path
    ) -> None:
        """Read multiple named ranges in one call."""
        reader = JsonCalibrationReader(sample_calibration_json)
        result = reader.read_named_ranges(
            ["creditParameters", "USD_Maturities"]
        )
        assert len(result) == 2
        assert result["creditParameters"].shape == (7,)
        assert result["USD_Maturities"].shape == (12,)

    def test_missing_key_raises(self, sample_calibration_json: Path) -> None:
        """KeyError for unknown named range."""
        reader = JsonCalibrationReader(sample_calibration_json)
        with pytest.raises(KeyError, match="nonexistent"):
            reader.read_named_range("nonexistent")

    def test_file_not_found(self) -> None:
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            JsonCalibrationReader("/nonexistent/path.json")

    def test_scalar_on_array_raises(
        self, sample_calibration_json: Path
    ) -> None:
        """ValueError when reading array as scalar."""
        reader = JsonCalibrationReader(sample_calibration_json)
        with pytest.raises(ValueError, match="expected 1"):
            reader.read_scalar("creditParameters")


# ── CSV directory reader tests ───────────────────────────────────


class TestCsvDirectoryCalibrationReader:
    """Tests for CsvDirectoryCalibrationReader."""

    def test_protocol_compliance(self, sample_csv_dir: Path) -> None:
        """Reader satisfies CalibrationDataReader protocol."""
        reader = CsvDirectoryCalibrationReader(sample_csv_dir)
        assert isinstance(reader, CalibrationDataReader)

    def test_read_scalar(self, sample_csv_dir: Path) -> None:
        """Read a single-value CSV."""
        reader = CsvDirectoryCalibrationReader(sample_csv_dir)
        val = reader.read_scalar("expectedCtsFwdRpiSwapsAtReform")
        assert val == pytest.approx(0.035)

    def test_read_1d_array(self, sample_csv_dir: Path) -> None:
        """Read a multi-value CSV."""
        reader = CsvDirectoryCalibrationReader(sample_csv_dir)
        arr = reader.read_named_range("creditParameters")
        assert arr.shape == (7,)
        assert arr[0] == pytest.approx(0.01)

    def test_read_string_array(self, sample_csv_dir: Path) -> None:
        """Read string values from CSV."""
        reader = CsvDirectoryCalibrationReader(sample_csv_dir)
        arr = reader.read_named_range("equityInitialDividendYields_Keys")
        assert len(arr) == 2
        assert "UK Equity" in arr

    def test_missing_range_raises(self, sample_csv_dir: Path) -> None:
        """KeyError for missing CSV file."""
        reader = CsvDirectoryCalibrationReader(sample_csv_dir)
        with pytest.raises(KeyError, match="nonexistent"):
            reader.read_named_range("nonexistent")

    def test_caching(self, sample_csv_dir: Path) -> None:
        """Second read returns cached result."""
        reader = CsvDirectoryCalibrationReader(sample_csv_dir)
        arr1 = reader.read_named_range("creditParameters")
        arr2 = reader.read_named_range("creditParameters")
        assert arr1 is arr2  # Same object from cache

    def test_directory_not_found(self) -> None:
        """FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            CsvDirectoryCalibrationReader("/nonexistent/dir")
