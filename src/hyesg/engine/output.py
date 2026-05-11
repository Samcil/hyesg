"""Output capture and SimulationResult container.

Provides the ``SimulationResult`` dataclass for holding simulation outputs
and utilities for extracting scan outputs into the result format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.core.types import OutputSpec


@dataclass
class SimulationResult:
    """Container for simulation results.

    Attributes:
        outputs: Dict mapping model_name -> field_name -> Array(n_steps, n_trials).
        time_grid: Array of time points with shape (n_steps+1,).
        metadata: Dict with config, seed, timing info.
    """

    outputs: dict[str, dict[str, Array]]
    time_grid: Array
    metadata: dict[str, Any] = field(default_factory=dict)

    def select(self, model: str, field_name: str) -> Array:
        """Select a specific output variable.

        Args:
            model: Model name.
            field_name: Output field name.

        Returns:
            Array with shape (n_steps, n_trials).

        Raises:
            KeyError: If model or field not found.
        """
        if model not in self.outputs:
            available = sorted(self.outputs.keys())
            raise KeyError(f"Model '{model}' not found. Available: {available}")
        model_outputs = self.outputs[model]
        if field_name not in model_outputs:
            available = sorted(model_outputs.keys())
            raise KeyError(
                f"Field '{field_name}' not found in model '{model}'. "
                f"Available: {available}"
            )
        return model_outputs[field_name]

    def to_dict(self) -> dict[str, Array]:
        """Flatten to {model.field: Array} dict.

        Returns:
            Dict mapping 'model_name.field_name' to Array.
        """
        flat: dict[str, Array] = {}
        for model_name, fields in self.outputs.items():
            for field_name, arr in fields.items():
                flat[f"{model_name}.{field_name}"] = arr
        return flat

    @property
    def n_trials(self) -> int:
        """Number of Monte Carlo trials."""
        for fields in self.outputs.values():
            for arr in fields.values():
                # Shape is (n_trials, n_steps) from vmap over trials
                return int(arr.shape[0])
        return 0

    @property
    def n_steps(self) -> int:
        """Number of simulation timesteps."""
        for fields in self.outputs.values():
            for arr in fields.values():
                return int(arr.shape[1])
        return 0

    @property
    def model_names(self) -> list[str]:
        """Sorted list of model names in the result."""
        return sorted(self.outputs.keys())


def extract_outputs(
    scan_outputs: dict[str, dict[str, Array]],
    output_specs: list[OutputSpec] | None = None,
) -> dict[str, dict[str, Array]]:
    """Extract and reshape scan outputs into SimulationResult format.

    If ``output_specs`` is None, captures all fields from all models.

    Args:
        scan_outputs: Raw scan outputs mapping model_name -> field_name -> Array.
            Arrays have shape (n_steps,) for a single trial.
        output_specs: Optional list of OutputSpec to filter outputs.

    Returns:
        Dict mapping model_name -> field_name -> Array.
    """
    if output_specs is None:
        return scan_outputs

    result: dict[str, dict[str, Array]] = {}
    for spec in output_specs:
        if spec.model_name not in scan_outputs:
            continue
        model_out = scan_outputs[spec.model_name]
        if spec.member_name not in model_out:
            continue
        if spec.model_name not in result:
            result[spec.model_name] = {}
        result[spec.model_name][spec.output_name] = model_out[spec.member_name]

    return result


def combine_regime_results(results: list[SimulationResult]) -> SimulationResult:
    """Combine results from multiple regimes by concatenating trials.

    All results must share the same time grid and model structure.

    Args:
        results: List of SimulationResult from individual regimes.

    Returns:
        Combined SimulationResult with concatenated trials.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Cannot combine empty results list")

    if len(results) == 1:
        return results[0]

    combined_outputs: dict[str, dict[str, Array]] = {}
    first = results[0]

    for model_name in first.outputs:
        combined_outputs[model_name] = {}
        for field_name in first.outputs[model_name]:
            arrays = [r.outputs[model_name][field_name] for r in results]
            combined_outputs[model_name][field_name] = jnp.concatenate(arrays, axis=0)

    metadata = {
        "regimes": [r.metadata for r in results],
        "n_regimes": len(results),
    }

    return SimulationResult(
        outputs=combined_outputs,
        time_grid=first.time_grid,
        metadata=metadata,
    )
