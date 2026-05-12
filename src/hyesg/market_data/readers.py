"""CSV readers for market data files.

Provides functions for loading yield curves and correlation matrices
from CSV files into calibration parameter schemas.
"""

from __future__ import annotations

import csv
from pathlib import Path

from hyesg.config.calibration_params import CorrelationSpec, YieldCurveSpec


def read_yield_curve_csv(path: Path) -> YieldCurveSpec:
    """Read spot rates from a CSV file (maturity, rate columns).

    Expects a CSV with header row containing 'tenor'/'maturity' and
    'rate'/'spot_rate' columns (case-insensitive).

    Args:
        path: Path to the CSV file.

    Returns:
        A YieldCurveSpec with knots and spot_rates from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the CSV format is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Yield curve file not found: {path}")

    knots: list[float] = []
    rates: list[float] = []

    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {path}")

        # Normalise headers to lowercase
        headers = {h.strip().lower(): h for h in reader.fieldnames}

        tenor_col = _find_column(headers, ("tenor", "maturity", "term"))
        rate_col = _find_column(headers, ("rate", "spot_rate", "spot"))

        if tenor_col is None:
            raise ValueError(
                f"CSV must have a 'tenor' or 'maturity' column, "
                f"found: {list(headers.keys())}"
            )
        if rate_col is None:
            raise ValueError(
                f"CSV must have a 'rate' or 'spot_rate' column, "
                f"found: {list(headers.keys())}"
            )

        for row in reader:
            knots.append(float(row[headers[tenor_col]].strip()))
            rates.append(float(row[headers[rate_col]].strip()))

    if not knots:
        raise ValueError(f"No data rows in CSV: {path}")

    return YieldCurveSpec(
        knots=tuple(knots),
        spot_rates=tuple(rates),
    )


def read_correlation_csv(path: Path) -> CorrelationSpec:
    """Read a correlation matrix from CSV.

    Expects a CSV where the first column is row labels and the header
    row contains column labels. The matrix must be symmetric.

    Args:
        path: Path to the CSV file.

    Returns:
        A CorrelationSpec with labels and matrix from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the CSV format is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Correlation file not found: {path}")

    with path.open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV file: {path}")

        # First cell is usually empty or a label column header
        labels = tuple(h.strip() for h in header[1:])
        if not labels:
            raise ValueError(f"No column labels in header: {path}")

        rows: list[tuple[float, ...]] = []
        for row in reader:
            if not row:
                continue
            values = tuple(float(v.strip()) for v in row[1:])
            rows.append(values)

    if not rows:
        raise ValueError(f"No data rows in CSV: {path}")

    return CorrelationSpec(
        labels=labels,
        matrix=tuple(rows),
    )


def _find_column(
    headers: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    """Find the first matching column name from candidates.

    Args:
        headers: Mapping of lowercase header → original header.
        candidates: Column name candidates to search for.

    Returns:
        The matching lowercase key, or None if not found.
    """
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return None
