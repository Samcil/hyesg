"""Calibration data reader protocol and implementations.

Provides a ``CalibrationDataReader`` protocol with Excel and JSON/CSV
backends for loading named ranges from calibration data sources.
The C# ESG reads ~40 named ranges from Excel files (ClosedXML);
this module replicates that capability using ``openpyxl``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────────────


@runtime_checkable
class CalibrationDataReader(Protocol):
    """Protocol for reading named ranges from a calibration data source.

    Mirrors the C# ``ParameterReader`` pattern where named ranges
    are resolved by name to 1D or 2D numpy arrays.
    """

    def read_named_range(self, name: str) -> np.ndarray:
        """Read a single named range.

        Args:
            name: The defined name / named range identifier.

        Returns:
            1D or 2D numpy array of the range values.

        Raises:
            KeyError: If the named range does not exist.
            ValueError: If the range data is invalid.
        """
        ...

    def read_named_ranges(self, names: list[str]) -> dict[str, np.ndarray]:
        """Read multiple named ranges in one call.

        Args:
            names: List of named range identifiers.

        Returns:
            Mapping of name → numpy array.

        Raises:
            KeyError: If any named range does not exist.
        """
        ...

    def read_scalar(self, name: str) -> float:
        """Read a single scalar value from a named range.

        Args:
            name: The named range identifier (must resolve to 1 cell).

        Returns:
            The scalar float value.

        Raises:
            KeyError: If the named range does not exist.
            ValueError: If the range is not a single cell.
        """
        ...


# ── Excel implementation ─────────────────────────────────────────


class ExcelCalibrationReader:
    """Read named ranges from an Excel workbook using openpyxl.

    Caches the workbook once opened so repeated reads of different
    named ranges from the same file do not re-parse the XML.

    Args:
        path: Path to the ``.xlsx`` calibration workbook.

    Raises:
        FileNotFoundError: If the path does not exist.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(
                f"Calibration workbook not found: {self._path}"
            )
        self._workbook: object | None = None  # lazy-loaded openpyxl Workbook

    def _get_workbook(self) -> object:
        """Lazy-load the openpyxl workbook (data-only for cached values)."""
        if self._workbook is None:
            try:
                import openpyxl
            except ImportError as exc:
                raise ImportError(
                    "openpyxl is required for Excel calibration reading. "
                    "Install with: uv add openpyxl"
                ) from exc
            self._workbook = openpyxl.load_workbook(
                str(self._path), data_only=True, read_only=True
            )
            logger.info("Loaded calibration workbook: %s", self._path.name)
        return self._workbook

    def read_named_range(self, name: str) -> np.ndarray:
        """Read a named range from the Excel workbook.

        Handles both single-cell and multi-cell (1D/2D) ranges.

        Args:
            name: Defined name in the workbook.

        Returns:
            Numpy array (1D for vectors, 2D for matrices).

        Raises:
            KeyError: If the named range does not exist.
            ValueError: If the range references are invalid.
        """
        wb = self._get_workbook()
        if name not in wb.defined_names:
            raise KeyError(f"Named range '{name}' not found in workbook")

        defn = wb.defined_names[name]
        destinations = list(defn.destinations)
        if not destinations:
            raise ValueError(
                f"Named range '{name}' has no cell destinations"
            )

        all_values: list[list[float]] = []
        for sheet_title, cell_range in destinations:
            ws = wb[sheet_title]
            cells = ws[cell_range]

            # Single cell
            if not isinstance(cells, tuple):
                val = _to_float(cells.value, name)
                return np.array([val])

            # Single row tuple of cells
            if not isinstance(cells[0], tuple):
                row_vals = [_to_float(c.value, name) for c in cells]
                return np.array(row_vals)

            # 2D range (tuple of tuples)
            for row in cells:
                row_vals = [_to_float(c.value, name) for c in row]
                all_values.append(row_vals)

        result = np.array(all_values)
        # Flatten to 1D if only one row or one column
        if result.ndim == 2 and (result.shape[0] == 1 or result.shape[1] == 1):
            result = result.flatten()
        return result

    def read_named_ranges(self, names: list[str]) -> dict[str, np.ndarray]:
        """Read multiple named ranges.

        Args:
            names: List of named range identifiers.

        Returns:
            Mapping of name → numpy array.
        """
        return {name: self.read_named_range(name) for name in names}

    def read_scalar(self, name: str) -> float:
        """Read a single scalar from a named range.

        Args:
            name: Named range that resolves to a single cell.

        Returns:
            The scalar float value.

        Raises:
            ValueError: If the named range contains more than one cell.
        """
        arr = self.read_named_range(name)
        if arr.size != 1:
            raise ValueError(
                f"Named range '{name}' has {arr.size} values, expected 1"
            )
        return float(arr.flat[0])

    def close(self) -> None:
        """Close the underlying workbook."""
        if self._workbook is not None:
            self._workbook.close()
            self._workbook = None

    def __enter__(self) -> ExcelCalibrationReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ── JSON/CSV fallback implementation ─────────────────────────────


class JsonCalibrationReader:
    """Read named ranges from a JSON file.

    The JSON file is a flat mapping of named-range names to either
    a scalar, a list (1D array), or a list-of-lists (2D array).

    Example JSON::

        {
            "simulationDate": "2025-01-09",
            "equityInitialDividendYields_Keys": ["UK", "US"],
            "equityInitialDividendYields_Values": [0.035, 0.014],
            "creditParameters": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        }

    Args:
        path: Path to the JSON file.

    Raises:
        FileNotFoundError: If the path does not exist.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(
                f"Calibration JSON not found: {self._path}"
            )
        self._data: dict[str, object] | None = None

    def _get_data(self) -> dict[str, object]:
        """Lazy-load and cache the JSON data."""
        if self._data is None:
            with self._path.open() as fh:
                self._data = json.load(fh)
            logger.info("Loaded calibration JSON: %s", self._path.name)
        return self._data

    def read_named_range(self, name: str) -> np.ndarray:
        """Read a named range from the JSON data.

        Args:
            name: Key in the JSON object.

        Returns:
            Numpy array (scalar wrapped as 1D, list as 1D,
            list-of-lists as 2D).

        Raises:
            KeyError: If the key does not exist.
        """
        data = self._get_data()
        if name not in data:
            raise KeyError(
                f"Named range '{name}' not found in JSON. "
                f"Available: {sorted(data.keys())}"
            )

        value = data[name]
        if isinstance(value, (int, float)):
            return np.array([float(value)])
        if isinstance(value, str):
            return np.array([value])
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return np.array(value, dtype=float)
            # Check if all elements are strings
            if value and isinstance(value[0], str):
                return np.array(value)
            return np.array(value, dtype=float)
        raise ValueError(
            f"Named range '{name}' has unsupported type: {type(value)}"
        )

    def read_named_ranges(self, names: list[str]) -> dict[str, np.ndarray]:
        """Read multiple named ranges.

        Args:
            names: List of key names.

        Returns:
            Mapping of name → numpy array.
        """
        return {name: self.read_named_range(name) for name in names}

    def read_scalar(self, name: str) -> float:
        """Read a single scalar value.

        Args:
            name: Key in the JSON object.

        Returns:
            Float value.

        Raises:
            ValueError: If the value is not a scalar.
        """
        data = self._get_data()
        if name not in data:
            raise KeyError(f"Named range '{name}' not found in JSON")
        value = data[name]
        if isinstance(value, (int, float)):
            return float(value)
        arr = self.read_named_range(name)
        if arr.size != 1:
            raise ValueError(
                f"Named range '{name}' has {arr.size} values, expected 1"
            )
        return float(arr.flat[0])


class CsvDirectoryCalibrationReader:
    """Read named ranges from a directory of CSV files.

    Each named range corresponds to a CSV file named ``{range_name}.csv``
    in the directory. The CSV should have no header for raw arrays, or
    a single column of values.

    Args:
        directory: Path to the directory containing CSV files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        if not self._directory.is_dir():
            raise FileNotFoundError(
                f"Calibration CSV directory not found: {self._directory}"
            )
        self._cache: dict[str, np.ndarray] = {}

    def read_named_range(self, name: str) -> np.ndarray:
        """Read a named range from a CSV file.

        Args:
            name: The file stem (e.g. ``"creditParameters"``
                maps to ``creditParameters.csv``).

        Returns:
            Numpy array from the CSV contents.

        Raises:
            KeyError: If no matching CSV file exists.
        """
        if name in self._cache:
            return self._cache[name]

        path = self._directory / f"{name}.csv"
        if not path.is_file():
            raise KeyError(
                f"Named range '{name}' not found: "
                f"expected file {path}"
            )

        rows: list[list[str]] = []
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if row:
                    rows.append(row)

        if not rows:
            raise ValueError(f"Empty CSV for named range '{name}': {path}")

        # Try numeric first, fall back to string
        try:
            arr = np.array(
                [[float(v) for v in row] for row in rows], dtype=float
            )
        except ValueError:
            arr = np.array(rows)

        # Flatten to 1D if single row or single column
        if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            arr = arr.flatten()

        self._cache[name] = arr
        return arr

    def read_named_ranges(self, names: list[str]) -> dict[str, np.ndarray]:
        """Read multiple named ranges.

        Args:
            names: List of CSV file stems.

        Returns:
            Mapping of name → numpy array.
        """
        return {name: self.read_named_range(name) for name in names}

    def read_scalar(self, name: str) -> float:
        """Read a single scalar from a CSV file.

        Args:
            name: CSV file stem.

        Returns:
            Float value.

        Raises:
            ValueError: If the CSV has more than one value.
        """
        arr = self.read_named_range(name)
        if arr.size != 1:
            raise ValueError(
                f"Named range '{name}' has {arr.size} values, expected 1"
            )
        return float(arr.flat[0])


# ── Helpers ──────────────────────────────────────────────────────


def _to_float(value: object, range_name: str) -> float:
    """Convert a cell value to float with error context.

    Args:
        value: Raw cell value from openpyxl.
        range_name: Named range for error messages.

    Returns:
        Float value.

    Raises:
        ValueError: If conversion fails.
    """
    if value is None:
        raise ValueError(
            f"Null cell value in named range '{range_name}'"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert '{value}' to float in "
            f"named range '{range_name}'"
        ) from exc
