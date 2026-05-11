"""Golden master storage for C# ESG reference data.

Provides save/load for reference simulation outputs in ``.npz`` format,
enabling reproducible parity comparisons across versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

    from hyesg.engine.output import SimulationResult

_META_KEY = "__metadata__"
_TIME_GRID_KEY = "__time_grid__"
_SEPARATOR = "/"


@dataclass(frozen=True)
class GoldenMaster:
    """Stored reference data from C# ESG.

    Attributes:
        name: Human-readable identifier for this golden master.
        outputs: Model -> field -> Array mapping of reference values.
        time_grid: Simulation time grid.
        metadata: Provenance info (C# version, params, creation date).
    """

    name: str
    outputs: dict[str, dict[str, Array]]
    time_grid: Array
    metadata: dict[str, Any]

    def save(self, path: Path) -> None:
        """Save golden master to a compressed ``.npz`` file.

        Arrays are converted to numpy for storage. Metadata is serialised
        as JSON in a special key.

        Args:
            path: Destination file path (should end in ``.npz``).
        """
        arrays: dict[str, np.ndarray] = {}
        arrays[_TIME_GRID_KEY] = np.asarray(self.time_grid)

        for model_name, fields in self.outputs.items():
            for field_name, arr in fields.items():
                key = f"{model_name}{_SEPARATOR}{field_name}"
                arrays[key] = np.asarray(arr)

        meta = {**self.metadata, "name": self.name}
        arrays[_META_KEY] = np.array(json.dumps(meta))

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> GoldenMaster:
        """Load golden master from a ``.npz`` file.

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            Reconstructed GoldenMaster instance.

        Raises:
            FileNotFoundError: If the path does not exist.
            KeyError: If required keys are missing.
        """
        data = np.load(path, allow_pickle=True)

        meta_raw = str(data[_META_KEY])
        meta = json.loads(meta_raw)
        name = meta.pop("name")

        time_grid = jnp.array(data[_TIME_GRID_KEY])

        outputs: dict[str, dict[str, Any]] = {}
        for key in data.files:
            if key in (_META_KEY, _TIME_GRID_KEY):
                continue
            model_name, field_name = key.split(_SEPARATOR, 1)
            if model_name not in outputs:
                outputs[model_name] = {}
            outputs[model_name][field_name] = jnp.array(data[key])

        return cls(
            name=name,
            outputs=outputs,
            time_grid=time_grid,
            metadata=meta,
        )

    @classmethod
    def from_simulation_result(
        cls,
        result: SimulationResult,
        name: str,
        **metadata: Any,
    ) -> GoldenMaster:
        """Create a golden master from a SimulationResult.

        Args:
            result: Simulation result to snapshot.
            name: Identifier for this golden master.
            **metadata: Additional provenance metadata.

        Returns:
            New GoldenMaster instance.
        """
        return cls(
            name=name,
            outputs=result.outputs,
            time_grid=result.time_grid,
            metadata={**result.metadata, **metadata},
        )
