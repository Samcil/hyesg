"""HDF5 reader/writer for SimulationResult.

Requires ``h5py``.  Install via ``pip install hyesg[io]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hyesg.engine.output import SimulationResult

__all__ = ["to_hdf5", "from_hdf5"]


def _require_h5py() -> Any:
    """Import h5py or raise a helpful error."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 I/O. "
            "Install it with: pip install hyesg[io]"
        ) from None
    return h5py


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def to_hdf5(result: SimulationResult, path: Path | str) -> None:
    """Write simulation results to an HDF5 file.

    HDF5 layout::

        /time_grid                          — dataset  (n_steps+1,)
        /models/<model>/<field>             — dataset  (n_trials, n_steps)
        /metadata                           — group with JSON attrs

    Args:
        result: SimulationResult to serialise.
        path: Output file path (e.g. ``"results.h5"``).
    """
    h5py = _require_h5py()

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "w") as f:
        # Time grid
        f.create_dataset("time_grid", data=np.asarray(result.time_grid))

        # Model outputs
        models_grp = f.create_group("models")
        for model_name in sorted(result.outputs):
            model_grp = models_grp.create_group(model_name)
            for field_name in sorted(result.outputs[model_name]):
                arr = np.asarray(result.outputs[model_name][field_name])
                model_grp.create_dataset(field_name, data=arr)

        # Metadata as JSON attribute
        meta_grp = f.create_group("metadata")
        meta_json = json.dumps(_make_serialisable(result.metadata))
        meta_grp.attrs["json"] = meta_json


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def from_hdf5(path: Path | str) -> SimulationResult:
    """Read simulation results from an HDF5 file.

    Args:
        path: Path to the ``.h5`` file.

    Returns:
        Reconstructed SimulationResult.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    h5py = _require_h5py()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    outputs: dict[str, dict[str, Array]] = {}
    metadata: dict[str, Any] = {}

    with h5py.File(file_path, "r") as f:
        # Time grid
        time_grid = jnp.array(f["time_grid"][:])

        # Model outputs
        models_grp = f["models"]
        for model_name in sorted(models_grp.keys()):
            model_grp = models_grp[model_name]
            model_outputs: dict[str, Array] = {}
            for field_name in sorted(model_grp.keys()):
                arr = model_grp[field_name][:]
                model_outputs[field_name] = jnp.array(arr)
            outputs[model_name] = model_outputs

        # Metadata
        if "metadata" in f:
            meta_grp = f["metadata"]
            if "json" in meta_grp.attrs:
                metadata = json.loads(meta_grp.attrs["json"])

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
