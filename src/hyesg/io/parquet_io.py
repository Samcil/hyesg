"""Parquet reader/writer for SimulationResult.

Requires ``pyarrow``.  Install via ``pip install hyesg[io]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hyesg.engine.output import SimulationResult

__all__ = ["to_parquet", "from_parquet"]


def _require_pyarrow() -> tuple[Any, Any]:
    """Import pyarrow or raise a helpful error."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet I/O. "
            "Install it with: pip install hyesg[io]"
        ) from None
    return pa, pq


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def to_parquet(result: SimulationResult, path: Path | str) -> None:
    """Write simulation results to Parquet files.

    Creates a directory containing one ``.parquet`` file per model and a
    ``metadata.json`` sidecar.

    Each Parquet file stores columnar data with one row per
    ``(trial, timestep)`` pair and one column per output field.

    Args:
        result: SimulationResult to serialise.
        path: Output directory (created if needed).
    """
    pa, pq = _require_pyarrow()

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    time_grid = np.asarray(result.time_grid)

    for model_name in sorted(result.outputs):
        fields = result.outputs[model_name]
        n_trials = result.n_trials
        n_steps = result.n_steps

        # Build columnar arrays: trial, time, field_1, field_2, …
        trial_col = np.repeat(np.arange(n_trials), n_steps)
        time_col = np.tile(time_grid[:n_steps], n_trials)

        columns: dict[str, np.ndarray] = {
            "trial": trial_col,
            "time": time_col,
        }
        for field_name in sorted(fields):
            arr = np.asarray(fields[field_name])  # (n_trials, n_steps)
            columns[field_name] = arr.ravel()  # row-major: trial-major

        table = pa.table(columns)

        # Embed metadata in the Parquet file metadata
        meta_json = json.dumps(
            _make_serialisable(result.metadata)
        ).encode("utf-8")
        existing_meta = table.schema.metadata or {}
        new_meta = {**existing_meta, b"hyesg_metadata": meta_json}
        table = table.replace_schema_metadata(new_meta)

        pq.write_table(table, out_dir / f"{model_name}.parquet")

    # Also write metadata.json for convenience
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(_make_serialisable(result.metadata), fh, indent=2)

    # Write time_grid separately so round-trip is exact
    tg_table = pa.table({"time": time_grid})
    pq.write_table(tg_table, out_dir / "_time_grid.parquet")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def from_parquet(path: Path | str) -> SimulationResult:
    """Read simulation results from Parquet files.

    Args:
        path: Directory containing ``.parquet`` files.

    Returns:
        Reconstructed SimulationResult.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    pa, pq = _require_pyarrow()

    in_dir = Path(path)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {in_dir}")

    # Read time grid
    tg_path = in_dir / "_time_grid.parquet"
    if tg_path.exists():
        tg_table = pq.read_table(tg_path)
        time_grid = jnp.array(tg_table["time"].to_numpy())
    else:
        time_grid = jnp.array([])

    # Read each model file
    outputs: dict[str, dict[str, Array]] = {}
    metadata: dict[str, Any] = {}

    for pq_path in sorted(in_dir.glob("*.parquet")):
        if pq_path.name.startswith("_"):
            continue
        model_name = pq_path.stem
        table = pq.read_table(pq_path)

        # Extract metadata from first file encountered
        if not metadata:
            schema_meta = table.schema.metadata or {}
            if b"hyesg_metadata" in schema_meta:
                metadata = json.loads(schema_meta[b"hyesg_metadata"])

        trial_col = table["trial"].to_numpy()
        n_trials = int(trial_col.max()) + 1
        n_steps = len(trial_col) // n_trials

        model_outputs: dict[str, Array] = {}
        for col_name in table.column_names:
            if col_name in ("trial", "time"):
                continue
            values = table[col_name].to_numpy()
            arr = values.reshape(n_trials, n_steps)
            model_outputs[col_name] = jnp.array(arr)

        outputs[model_name] = model_outputs

    # Fall back to metadata.json
    if not metadata:
        meta_path = in_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                metadata = json.load(fh)

    return SimulationResult(
        outputs=outputs,
        time_grid=time_grid,
        metadata=metadata,
    )


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
