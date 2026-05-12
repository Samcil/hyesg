"""IO module for reading and writing SimulationResult objects.

CSV support is always available.  Parquet and HDF5 require optional
dependencies (``pip install hyesg[io]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hyesg.io.correlation_csv import (
    CORRELATION_CSV_NAMES,
    CorrelationSource,
    CsvCorrelationSource,
)
from hyesg.io.csv_io import from_csv, to_csv
from hyesg.io.streaming import StreamingWriter

if TYPE_CHECKING:
    from pathlib import Path

    from hyesg.engine.output import SimulationResult

__all__ = [
    "CORRELATION_CSV_NAMES",
    "CorrelationSource",
    "CsvCorrelationSource",
    "StreamingWriter",
    "from_csv",
    "from_hdf5",
    "from_parquet",
    "to_csv",
    "to_hdf5",
    "to_parquet",
]


# ------------------------------------------------------------------
# Lazy wrappers for optional-dependency formats
# ------------------------------------------------------------------


def to_parquet(result: SimulationResult, path: Path | str) -> None:
    """Write results to Parquet (requires ``pyarrow``)."""
    from hyesg.io.parquet_io import to_parquet as _to_parquet

    _to_parquet(result, path)


def from_parquet(path: Path | str) -> SimulationResult:
    """Read results from Parquet (requires ``pyarrow``)."""
    from hyesg.io.parquet_io import from_parquet as _from_parquet

    return _from_parquet(path)


def to_hdf5(result: SimulationResult, path: Path | str) -> None:
    """Write results to HDF5 (requires ``h5py``)."""
    from hyesg.io.hdf5_io import to_hdf5 as _to_hdf5

    _to_hdf5(result, path)


def from_hdf5(path: Path | str) -> SimulationResult:
    """Read results from HDF5 (requires ``h5py``)."""
    from hyesg.io.hdf5_io import from_hdf5 as _from_hdf5

    return _from_hdf5(path)
