"""Streaming writer for large simulations.

Writes results incrementally so that very large simulations need not be
held entirely in memory.  Only CSV streaming is supported; this module
has **no** optional dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from jax import Array

__all__ = ["StreamingWriter"]


class StreamingWriter:
    """Incrementally write simulation outputs to disk.

    Useful for very large simulations that exceed available memory.
    Results are flushed to CSV after each timestep.

    Args:
        path: Output directory (created if needed).
        fmt: Output format — currently only ``"csv"`` is supported.

    Raises:
        ValueError: If *fmt* is not ``"csv"``.

    Example::

        writer = StreamingWriter("output/")
        for t in time_grid:
            step = run_one_step(...)
            writer.write_timestep(t, step)
        writer.finalize(metadata={"seed": 42})
    """

    def __init__(self, path: Path | str, fmt: str = "csv") -> None:
        if fmt != "csv":
            raise ValueError(
                f"StreamingWriter only supports fmt='csv', got {fmt!r}"
            )
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._fmt = fmt
        self._csv_path = self._path / "results.csv"
        self._header_written = False
        self._file_handle = open(self._csv_path, "w", newline="")  # noqa: SIM115
        self._writer = csv.writer(self._file_handle)
        self._timesteps_written = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_timestep(
        self,
        t: float,
        step_outputs: dict[str, dict[str, Array]],
    ) -> None:
        """Write one timestep of outputs.

        Args:
            t: The time value for this step.
            step_outputs: Mapping ``model_name -> field_name -> Array``.
                Each array should have shape ``(n_trials,)``.
        """
        if not self._header_written:
            self._writer.writerow(
                ["time", "trial", "model", "variable", "value"]
            )
            self._header_written = True

        for model_name in sorted(step_outputs):
            for field_name in sorted(step_outputs[model_name]):
                arr = np.asarray(step_outputs[model_name][field_name])
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                for trial_idx in range(arr.shape[0]):
                    self._writer.writerow([
                        f"{t:.15g}",
                        trial_idx,
                        model_name,
                        field_name,
                        f"{arr[trial_idx]:.15g}",
                    ])

        self._timesteps_written += 1
        self._file_handle.flush()

    def finalize(self, metadata: dict[str, Any] | None = None) -> None:
        """Close the output file and write metadata.

        Args:
            metadata: Optional metadata dict to persist as JSON.
        """
        self._file_handle.close()

        meta = metadata or {}
        meta_path = self._path / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(_make_serialisable(meta), fh, indent=2)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> StreamingWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._file_handle.closed:
            self._file_handle.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def timesteps_written(self) -> int:
        """Number of timesteps written so far."""
        return self._timesteps_written


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
