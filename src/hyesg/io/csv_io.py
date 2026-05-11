"""CSV reader/writer for SimulationResult.

Provides round-trip serialisation using only the Python standard library
(``csv`` and ``json``).  Two layouts are supported:

* **wide** — one CSV per model, columns ``time, trial_0, …, trial_N`` per field.
* **tidy** — single ``results.csv`` with columns
  ``time, trial, model, variable, value``.

Metadata is always stored as a companion ``metadata.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hyesg.engine.output import SimulationResult

__all__ = ["to_csv", "from_csv"]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def to_csv(
    result: SimulationResult,
    path: Path | str,
    *,
    fmt: str = "tidy",
) -> None:
    """Write simulation results to CSV files.

    Args:
        result: SimulationResult to serialise.
        path: Directory to write into (created if needed).
        fmt: ``"tidy"`` (default) writes a single ``results.csv``;
            ``"wide"`` writes one CSV per model.

    Raises:
        ValueError: If *fmt* is not ``"tidy"`` or ``"wide"``.
    """
    if fmt not in ("tidy", "wide"):
        raise ValueError(f"fmt must be 'tidy' or 'wide', got {fmt!r}")

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "tidy":
        _write_tidy(result, out_dir)
    else:
        _write_wide(result, out_dir)

    _write_metadata(result, out_dir)


def _write_tidy(result: SimulationResult, out_dir: Path) -> None:
    """Write a single tidy-format CSV."""
    time_grid = np.asarray(result.time_grid)
    csv_path = out_dir / "results.csv"

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "trial", "model", "variable", "value"])

        for model_name in sorted(result.outputs):
            for field_name in sorted(result.outputs[model_name]):
                arr = np.asarray(result.outputs[model_name][field_name])
                n_trials, n_steps = arr.shape
                for trial_idx in range(n_trials):
                    for step_idx in range(n_steps):
                        writer.writerow([
                            f"{time_grid[step_idx]:.15g}",
                            trial_idx,
                            model_name,
                            field_name,
                            f"{arr[trial_idx, step_idx]:.15g}",
                        ])


def _write_wide(result: SimulationResult, out_dir: Path) -> None:
    """Write one CSV per model in wide format."""
    time_grid = np.asarray(result.time_grid)

    for model_name in sorted(result.outputs):
        csv_path = out_dir / f"{model_name}.csv"
        fields = result.outputs[model_name]
        n_trials = result.n_trials
        n_steps = result.n_steps

        # Header: time, field_trial_0, field_trial_1, ...
        header = ["time"]
        sorted_fields = sorted(fields)
        for field_name in sorted_fields:
            for t in range(n_trials):
                header.append(f"{field_name}_trial_{t}")

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for step_idx in range(n_steps):
                row: list[str] = [f"{time_grid[step_idx]:.15g}"]
                for field_name in sorted_fields:
                    arr = np.asarray(fields[field_name])
                    for t in range(n_trials):
                        row.append(f"{arr[t, step_idx]:.15g}")
                writer.writerow(row)


def _write_metadata(result: SimulationResult, out_dir: Path) -> None:
    """Write metadata to JSON, coercing non-serialisable values."""
    meta_path = out_dir / "metadata.json"
    serialisable = _make_serialisable(result.metadata)
    with open(meta_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def from_csv(
    path: Path | str,
    *,
    fmt: str | None = None,
) -> SimulationResult:
    """Read simulation results from CSV files.

    Args:
        path: Directory containing CSV and ``metadata.json``.
        fmt: ``"tidy"`` or ``"wide"``.  If *None* the format is
            auto-detected from the directory contents.

    Returns:
        Reconstructed SimulationResult.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the format cannot be determined.
    """
    in_dir = Path(path)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {in_dir}")

    if fmt is None:
        fmt = _detect_format(in_dir)

    if fmt == "tidy":
        outputs, time_grid = _read_tidy(in_dir)
    else:
        outputs, time_grid = _read_wide(in_dir)

    metadata = _read_metadata(in_dir)

    return SimulationResult(
        outputs=outputs,
        time_grid=jnp.array(time_grid),
        metadata=metadata,
    )


def _detect_format(in_dir: Path) -> str:
    """Auto-detect CSV layout."""
    if (in_dir / "results.csv").exists():
        return "tidy"
    csv_files = list(in_dir.glob("*.csv"))
    if csv_files:
        return "wide"
    raise ValueError(f"No CSV files found in {in_dir}")


def _read_tidy(
    in_dir: Path,
) -> tuple[dict[str, dict[str, Array]], np.ndarray]:
    """Read tidy-format CSV."""
    csv_path = in_dir / "results.csv"
    # Accumulate: (model, variable) -> {trial -> {time -> value}}
    raw: dict[tuple[str, str], dict[int, dict[float, float]]] = {}
    times_set: set[float] = set()

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            time_val = float(row["time"])
            trial = int(row["trial"])
            model = row["model"]
            variable = row["variable"]
            value = float(row["value"])
            times_set.add(time_val)
            key = (model, variable)
            raw.setdefault(key, {}).setdefault(trial, {})[time_val] = value

    sorted_times = sorted(times_set)
    n_steps = len(sorted_times)

    outputs: dict[str, dict[str, Array]] = {}
    for (model, variable), trial_dict in raw.items():
        n_trials = max(trial_dict) + 1
        arr = np.zeros((n_trials, n_steps), dtype=np.float64)
        for trial_idx, time_values in trial_dict.items():
            for step_idx, t in enumerate(sorted_times):
                arr[trial_idx, step_idx] = time_values[t]
        outputs.setdefault(model, {})[variable] = jnp.array(arr)

    time_grid = np.array(sorted_times, dtype=np.float64)
    return outputs, time_grid


def _read_wide(
    in_dir: Path,
) -> tuple[dict[str, dict[str, Array]], np.ndarray]:
    """Read wide-format CSVs (one per model)."""
    outputs: dict[str, dict[str, Array]] = {}
    time_grid: np.ndarray | None = None

    for csv_path in sorted(in_dir.glob("*.csv")):
        model_name = csv_path.stem
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)

        # Parse header to discover fields and trial count
        field_cols: dict[str, list[int]] = {}
        for col_idx, col_name in enumerate(header):
            if col_name == "time":
                continue
            # Expected pattern: fieldname_trial_N
            parts = col_name.rsplit("_trial_", 1)
            if len(parts) == 2:
                field_name = parts[0]
                field_cols.setdefault(field_name, []).append(col_idx)

        # Read data rows
        data_rows: list[list[str]] = []
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            data_rows = list(reader)

        n_steps = len(data_rows)
        times = np.array([float(r[0]) for r in data_rows], dtype=np.float64)
        if time_grid is None:
            time_grid = times

        model_outputs: dict[str, Array] = {}
        for field_name, col_indices in sorted(field_cols.items()):
            n_trials = len(col_indices)
            arr = np.zeros((n_trials, n_steps), dtype=np.float64)
            for trial_idx, col_idx in enumerate(col_indices):
                for step_idx, row in enumerate(data_rows):
                    arr[trial_idx, step_idx] = float(row[col_idx])
            model_outputs[field_name] = jnp.array(arr)

        outputs[model_name] = model_outputs

    if time_grid is None:
        time_grid = np.array([], dtype=np.float64)

    return outputs, time_grid


def _read_metadata(in_dir: Path) -> dict[str, Any]:
    """Read metadata JSON if it exists."""
    meta_path = in_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "__jax_array__") or type(obj).__name__ == "ArrayImpl":
        return np.asarray(obj).tolist()
    return obj
