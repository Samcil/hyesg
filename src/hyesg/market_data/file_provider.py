"""File-based market data provider.

Reads market data from a directory of CSV files and builds
interpolation curves using :class:`CubicSpline`.
"""

from __future__ import annotations

import csv
import functools
import math
from pathlib import Path

from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.curves.splines import CubicSpline


class MarketDataError(Exception):
    """Raised when market data cannot be loaded or is invalid."""


class FileMarketData:
    """File-based market data provider.

    Reads CSV files from a structured directory and constructs
    :class:`CubicSpline` curves for term-structure data.

    Directory structure::

        market_data/
            GBP/
                zero_curve.csv       # tenor,rate columns
                inflation_curve.csv
            USD/
                zero_curve.csv
            credit/
                AAA/
                    GBP/spread_curve.csv
            fx/
                GBPUSD.csv          # date,rate columns
            equity/
                FTSE100.csv         # date,level columns

    Args:
        directory: Root directory containing market data files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        if not self._directory.is_dir():
            raise FileNotFoundError(
                f"Market data directory not found: {self._directory}"
            )

    def __hash__(self) -> int:
        return hash(self._directory.resolve())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileMarketData):
            return NotImplemented
        return self._directory.resolve() == other._directory.resolve()

    @functools.lru_cache(maxsize=128)
    def get_zero_curve(
        self,
        currency: str,
        date: str | None = None,
    ) -> ParametricCurve:
        """Return the nominal zero-coupon yield curve.

        Args:
            currency: ISO currency code (e.g. ``"GBP"``).
            date: Optional valuation date (currently unused).

        Returns:
            Cubic-spline curve mapping tenor to zero rate.

        Raises:
            MarketDataError: If the CSV file is missing or invalid.
        """
        path = self._directory / currency / "zero_curve.csv"
        return self._load_curve(path, f"{currency} zero curve")

    @functools.lru_cache(maxsize=128)
    def get_inflation_curve(
        self,
        currency: str,
        date: str | None = None,
    ) -> ParametricCurve:
        """Return the real (inflation-linked) yield curve.

        Args:
            currency: ISO currency code.
            date: Optional valuation date (currently unused).

        Returns:
            Cubic-spline curve mapping tenor to real rate.

        Raises:
            MarketDataError: If the CSV file is missing or invalid.
        """
        path = self._directory / currency / "inflation_curve.csv"
        return self._load_curve(path, f"{currency} inflation curve")

    @functools.lru_cache(maxsize=128)
    def get_credit_curve(
        self,
        rating: str,
        currency: str,
        date: str | None = None,
    ) -> ParametricCurve:
        """Return the credit spread curve for a given rating.

        Args:
            rating: Credit rating label (e.g. ``"AAA"``).
            currency: ISO currency code.
            date: Optional valuation date (currently unused).

        Returns:
            Cubic-spline curve mapping tenor to credit spread.

        Raises:
            MarketDataError: If the CSV file is missing or invalid.
        """
        path = self._directory / "credit" / rating / currency / "spread_curve.csv"
        return self._load_curve(path, f"{rating}/{currency} credit spread curve")

    @functools.lru_cache(maxsize=128)
    def get_fx_spot(
        self,
        domestic: str,
        foreign: str,
        date: str | None = None,
    ) -> float:
        """Return the FX spot rate.

        Args:
            domestic: Domestic currency code.
            foreign: Foreign currency code.
            date: Optional valuation date (currently unused).

        Returns:
            Spot exchange rate.

        Raises:
            MarketDataError: If the CSV file is missing or invalid.
        """
        pair = f"{domestic}{foreign}"
        path = self._directory / "fx" / f"{pair}.csv"
        return self._load_scalar(path, "rate", f"{pair} FX spot")

    @functools.lru_cache(maxsize=128)
    def get_equity_index(
        self,
        name: str,
        date: str | None = None,
    ) -> float:
        """Return the level of an equity index.

        Args:
            name: Index identifier (e.g. ``"FTSE100"``).
            date: Optional valuation date (currently unused).

        Returns:
            Index level.

        Raises:
            MarketDataError: If the CSV file is missing or invalid.
        """
        path = self._directory / "equity" / f"{name}.csv"
        return self._load_scalar(path, "level", f"{name} equity index")

    # ── Private helpers ──────────────────────────────────────────

    def _load_curve(self, path: Path, description: str) -> CubicSpline:
        """Read a ``tenor,rate`` CSV and build a :class:`CubicSpline`.

        Args:
            path: Path to the CSV file.
            description: Human-readable label for error messages.

        Returns:
            A CubicSpline interpolating the data.

        Raises:
            MarketDataError: On file-not-found or validation failure.
        """
        if not path.is_file():
            raise MarketDataError(f"{description}: file not found at {path}")

        tenors: list[float] = []
        rates: list[float] = []

        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {
                "tenor",
                "rate",
            }.issubset(reader.fieldnames):
                raise MarketDataError(
                    f"{description}: CSV must have 'tenor' and 'rate' columns, "
                    f"got {reader.fieldnames}"
                )

            for line_no, row in enumerate(reader, start=2):
                try:
                    tenor = float(row["tenor"])
                    rate = float(row["rate"])
                except (ValueError, KeyError) as exc:
                    raise MarketDataError(
                        f"{description}: invalid data on line {line_no}: {row}"
                    ) from exc

                if tenor < 0.0:
                    raise MarketDataError(
                        f"{description}: negative tenor {tenor} on line {line_no}"
                    )
                if not math.isfinite(rate):
                    raise MarketDataError(
                        f"{description}: non-finite rate {rate} on line {line_no}"
                    )

                tenors.append(tenor)
                rates.append(rate)

        if len(tenors) < 2:
            raise MarketDataError(
                f"{description}: need at least 2 data points, got {len(tenors)}"
            )

        # Sort by tenor for spline construction
        sorted_pairs = sorted(zip(tenors, rates))
        sorted_tenors = [t for t, _ in sorted_pairs]
        sorted_rates = [r for _, r in sorted_pairs]

        # Check for duplicate tenors
        for i in range(1, len(sorted_tenors)):
            if sorted_tenors[i] == sorted_tenors[i - 1]:
                raise MarketDataError(
                    f"{description}: duplicate tenor {sorted_tenors[i]}"
                )

        return CubicSpline(sorted_tenors, sorted_rates)

    def _load_scalar(
        self,
        path: Path,
        value_column: str,
        description: str,
    ) -> float:
        """Read the first data row of a CSV and return a scalar value.

        Args:
            path: Path to the CSV file.
            value_column: Name of the column to extract.
            description: Human-readable label for error messages.

        Returns:
            The scalar value.

        Raises:
            MarketDataError: On file-not-found or validation failure.
        """
        if not path.is_file():
            raise MarketDataError(f"{description}: file not found at {path}")

        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or value_column not in reader.fieldnames:
                raise MarketDataError(
                    f"{description}: CSV must have a '{value_column}' column, "
                    f"got {reader.fieldnames}"
                )

            row = next(reader, None)
            if row is None:
                raise MarketDataError(f"{description}: CSV has no data rows")

            try:
                value = float(row[value_column])
            except (ValueError, KeyError) as exc:
                raise MarketDataError(
                    f"{description}: invalid {value_column} value: {row}"
                ) from exc

            if not math.isfinite(value):
                raise MarketDataError(
                    f"{description}: non-finite {value_column}: {value}"
                )

        return value
