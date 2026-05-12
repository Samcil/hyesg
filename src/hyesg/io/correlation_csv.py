"""CSV-based correlation source — loads the 7 correlation block files.

Matches the C# ESG engine's 7-file correlation assembly pattern from
``Calibration.GetCorrelations``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# The 7 canonical CSV file names in assembly order (matching C#)
CORRELATION_CSV_NAMES: tuple[str, ...] = (
    "GbpNominalsAndRealRates",
    "GbpInflations",
    "ForeignNominals",
    "ExchangeRates",
    "EquitiesAndGrowthAssets",
    "EquityJumps",
    "LpiLiquidityGaps",
)


@runtime_checkable
class CorrelationSource(Protocol):
    """Protocol for pluggable correlation data backends."""

    def load_correlation_block(self, name: str) -> tuple[list[str], np.ndarray]:
        """Return (labels, correlation_matrix) for a named block.

        Args:
            name: Block name (e.g. ``"GbpNominalsAndRealRates"``).

        Returns:
            Tuple of (dZ-factor label list, square correlation matrix).
        """
        ...

    def available_blocks(self) -> list[str]:
        """Return names of all available correlation blocks.

        Returns:
            List of block names that can be loaded.
        """
        ...


class CsvCorrelationSource:
    """Load correlation blocks from CSV files on disk.

    Each CSV file has:

    - Row 1: comma-separated dZ-factor labels
    - Remaining rows: correlation matrix values

    This matches the C# ``ReadCorrelationMatrixCsv`` pattern.
    """

    def __init__(self, directory: Path | str) -> None:
        """Initialize with directory containing correlation CSV files.

        Args:
            directory: Path to directory containing the 7 CSV files.

        Raises:
            FileNotFoundError: If directory does not exist.
        """
        self._directory = Path(directory)
        if not self._directory.is_dir():
            raise FileNotFoundError(
                f"Correlation CSV directory not found: {self._directory}"
            )

    def load_correlation_block(self, name: str) -> tuple[list[str], np.ndarray]:
        """Load a single correlation block from its CSV file.

        Args:
            name: Block name (without .csv extension).

        Returns:
            Tuple of (labels, matrix) where labels is a list of
            dZ-factor label strings and matrix is a square numpy array.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV data is malformed.
        """
        csv_path = self._directory / f"{name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Correlation CSV not found: {csv_path}")

        # Use utf-8-sig to strip BOM if present (C# CSVs use UTF-8 BOM)
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            # First row: labels
            labels = next(reader)
            labels = [label.strip() for label in labels]
            n = len(labels)

            # Remaining rows: matrix data
            rows: list[list[float]] = []
            for row in reader:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                values = [float(v.strip()) for v in row]
                if len(values) != n:
                    raise ValueError(
                        f"Row {len(rows) + 1} in {csv_path.name} has "
                        f"{len(values)} values, expected {n}"
                    )
                rows.append(values)

        if len(rows) != n:
            raise ValueError(
                f"{csv_path.name} has {len(rows)} data rows, "
                f"expected {n} (square matrix)"
            )

        matrix = np.array(rows, dtype=np.float64)
        logger.debug("Loaded correlation block %r: %d×%d", name, n, n)
        return labels, matrix

    def available_blocks(self) -> list[str]:
        """Return names of CSV files available in the directory.

        Returns:
            List of block names (without .csv extension).
        """
        return [p.stem for p in sorted(self._directory.glob("*.csv"))]
