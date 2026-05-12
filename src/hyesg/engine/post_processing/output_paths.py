"""Output path specifications.

``OutputPathSpec`` describes *which* data to pull from simulation results
and how to transform it before consumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from pydantic import BaseModel

if TYPE_CHECKING:
    from hyesg.engine.post_processing.protocol import SimulationResults


class OutputPathSpec(BaseModel):
    """Specification for resolving a single output path.

    Attributes:
        model: Model name in ``SimulationResults.paths``.
        field: Field name (unused when paths are flat dicts — reserved
            for structured results).
        transform: Optional transform to apply: ``"log"``,
            ``"cumulative"``, or ``"annualised"``.
        label: Human-readable label for the resolved output.
    """

    model_config = {"arbitrary_types_allowed": True}

    model: str
    field: str
    transform: str | None = None
    label: str = ""

    def resolve(self, results: SimulationResults) -> Any:
        """Extract and optionally transform data from *results*.

        Args:
            results: Simulation results containing raw paths.

        Returns:
            JAX array (potentially transformed).

        Raises:
            KeyError: If the model is not present in results.
        """
        if self.model not in results.paths:
            available = sorted(results.paths.keys())
            raise KeyError(
                f"Model '{self.model}' not found in results. Available: {available}"
            )

        arr = jnp.asarray(results.paths[self.model])

        if self.transform == "log":
            arr = jnp.log(jnp.maximum(arr, jnp.float64(1e-12)))
        elif self.transform == "cumulative":
            arr = jnp.cumsum(arr, axis=-1)
        elif self.transform == "annualised":
            # Annualise by raising to (1/t) power along the last axis
            n = arr.shape[-1]
            t = jnp.arange(1, n + 1, dtype=jnp.float64)
            arr = arr ** (jnp.float64(1.0) / t)

        return arr
