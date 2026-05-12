"""Assemble full correlation matrix from CSV blocks and credit correlations.

The ``CorrelationAssembler`` loads the 7 canonical correlation CSV files,
builds a dZ-factor label registry, constructs the full block-diagonal
correlation matrix, adds programmatic credit correlations, and performs
Cholesky decomposition with near-PD repair.

Matches the C# ``Calibration.GetCorrelations`` pipeline.

.. todo::
    **F40 Integration Path** — ``CorrelationAssembler`` currently operates
    independently of the :class:`~hyesg.engine.simulator.Simulator`.
    Future integration:

    1. ``SimulationSetup.to_simulation_config()`` should invoke
       ``CorrelationAssembler.build()`` to produce ``CorrelationEntry``
       pairs from CSV blocks instead of hard-coding them.
    2. The assembled Cholesky factor should feed directly into
       ``Simulator._cholesky`` rather than being re-computed.
    3. The ``DzFactorLabelRegistry`` should be reconciled with the
       ``Simulator``'s model-to-shock mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

from hyesg.core.matrix import SymmetricLabelledMatrix
from hyesg.engine.correlation import cholesky_factor, validate_and_repair
from hyesg.io.correlation_csv import (
    CORRELATION_CSV_NAMES,
    CorrelationSource,
)
from hyesg.orchestration.label_registry import DzFactorLabelRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreditCorrelationConfig:
    """Configuration for programmatic credit correlations.

    Matches the C# ``CreditCorrelations`` method constants.

    Attributes:
        intensity_intensity: Correlation between credit intensity
            processes (same or different class). C# value: 0.93.
        intensity_recovery: Correlation between credit intensity
            and default/liquidity processes. C# value: 0.7.
        recovery_recovery: Correlation between default/liquidity
            processes. C# value: 0.8.
        credit_class_labels: Labels for credit class dZ-factors.
        default_and_liquidity_labels: Labels for default and
            liquidity dZ-factors.
    """

    intensity_intensity: float = 0.93
    intensity_recovery: float = 0.7
    recovery_recovery: float = 0.8
    credit_class_labels: tuple[str, ...] = field(default_factory=tuple)
    default_and_liquidity_labels: tuple[str, ...] = field(default_factory=tuple)


class CorrelationAssembler:
    """Assemble full correlation matrix from CSV blocks.

    Pipeline:

    1. Load all 7 CSV correlation blocks via ``CorrelationSource``.
    2. Build unified dZ-factor label registry.
    3. Assemble block-diagonal matrix from CSV data.
    4. Optionally add programmatic credit correlations.
    5. Validate and repair (nearest-PD if needed).
    6. Compute Cholesky factor.
    """

    def __init__(
        self,
        source: CorrelationSource,
        block_names: tuple[str, ...] = CORRELATION_CSV_NAMES,
    ) -> None:
        """Initialize with a correlation data source.

        Args:
            source: Pluggable backend implementing ``CorrelationSource``.
            block_names: Ordered tuple of block names to load.
        """
        self._source = source
        self._block_names = block_names
        self._blocks: dict[str, tuple[list[str], np.ndarray]] = {}
        self._registry: DzFactorLabelRegistry | None = None

    def load_blocks(self) -> DzFactorLabelRegistry:
        """Load all correlation blocks and build the label registry.

        Returns:
            The label registry built from all loaded blocks.

        Raises:
            FileNotFoundError: If any CSV file is missing.
            ValueError: If any CSV data is malformed or labels collide.
        """
        all_labels: list[str] = []
        self._blocks.clear()

        for name in self._block_names:
            labels, matrix = self._source.load_correlation_block(name)
            self._blocks[name] = (labels, matrix)
            all_labels.extend(labels)
            logger.info(
                "Loaded correlation block %r: %d factors",
                name,
                len(labels),
            )

        self._registry = DzFactorLabelRegistry(all_labels)
        logger.info(
            "Built label registry with %d total factors",
            self._registry.size,
        )
        return self._registry

    @property
    def registry(self) -> DzFactorLabelRegistry:
        """Return the dZ-factor label registry.

        Returns:
            The registry built by ``load_blocks()``.

        Raises:
            RuntimeError: If ``load_blocks()`` has not been called.
        """
        if self._registry is None:
            raise RuntimeError(
                "Call load_blocks() before accessing the registry"
            )
        return self._registry

    def factor_labels(self) -> list[str]:
        """Return ordered list of all dZ-factor labels.

        Returns:
            List of labels in block assembly order.
        """
        return self.registry.labels

    def assemble(
        self,
        credit_config: CreditCorrelationConfig | None = None,
    ) -> SymmetricLabelledMatrix:
        """Build the full correlation matrix from loaded blocks.

        Assembles a block-diagonal matrix from the CSV blocks, then
        overlays any programmatic credit correlations.

        Args:
            credit_config: Optional credit correlation configuration.
                If provided, credit correlations are added to the matrix.

        Returns:
            A ``SymmetricLabelledMatrix`` covering all dZ-factors.
        """
        registry = self.registry
        n = registry.size
        matrix = np.eye(n, dtype=np.float64)

        # Fill block-diagonal from CSV data
        for name in self._block_names:
            if name not in self._blocks:
                continue
            labels, block_matrix = self._blocks[name]
            indices = registry.indices_of(labels)
            for i_local, i_global in enumerate(indices):
                for j_local, j_global in enumerate(indices):
                    matrix[i_global, j_global] = block_matrix[
                        i_local, j_local
                    ]

        # Add programmatic credit correlations
        if credit_config is not None:
            matrix = self._add_credit_correlations(
                matrix, registry, credit_config
            )

        # Convert to JAX and validate/repair
        jax_matrix = jnp.asarray(matrix, dtype=jnp.float64)
        result = validate_and_repair(
            jax_matrix,
            labels=registry.labels,
            method="higham",
        )
        assert isinstance(result, SymmetricLabelledMatrix)
        return result

    def assemble_with_cholesky(
        self,
        credit_config: CreditCorrelationConfig | None = None,
    ) -> tuple[SymmetricLabelledMatrix, Array]:
        """Assemble the full correlation matrix and compute Cholesky factor.

        Args:
            credit_config: Optional credit correlation configuration.

        Returns:
            Tuple of (correlation_matrix, cholesky_L) where
            cholesky_L is the lower-triangular Cholesky factor.
        """
        corr_matrix = self.assemble(credit_config)
        cholesky_L = cholesky_factor(corr_matrix.data)
        logger.info(
            "Computed Cholesky factor: %d×%d",
            cholesky_L.shape[0],
            cholesky_L.shape[1],
        )
        return corr_matrix, cholesky_L

    @staticmethod
    def _add_credit_correlations(
        matrix: np.ndarray,
        registry: DzFactorLabelRegistry,
        config: CreditCorrelationConfig,
    ) -> np.ndarray:
        """Add programmatic credit correlations to the matrix.

        Matches the C# ``CreditCorrelations`` method:

        - Intensity-intensity: ``config.intensity_intensity`` (0.93)
        - Intensity-recovery: ``config.intensity_recovery`` (0.7)
        - Recovery-recovery: ``config.recovery_recovery`` (0.8)

        Args:
            matrix: The correlation matrix to modify (in-place).
            registry: The label registry for index lookups.
            config: Credit correlation parameters and labels.

        Returns:
            The modified matrix.
        """
        # Filter to only labels that exist in the registry
        credit_labels = [
            label
            for label in config.credit_class_labels
            if registry.contains(label)
        ]
        default_labels = [
            label
            for label in config.default_and_liquidity_labels
            if registry.contains(label)
        ]

        credit_indices = registry.indices_of(credit_labels)
        default_indices = registry.indices_of(default_labels)

        # Intensity-intensity correlations
        for i_idx, i_global in enumerate(credit_indices):
            for j_idx in range(i_idx):
                j_global = credit_indices[j_idx]
                matrix[i_global, j_global] = config.intensity_intensity
                matrix[j_global, i_global] = config.intensity_intensity

        # Intensity-recovery correlations
        for i_global in credit_indices:
            for j_global in default_indices:
                matrix[i_global, j_global] = config.intensity_recovery
                matrix[j_global, i_global] = config.intensity_recovery

        # Recovery-recovery correlations
        for i_idx, i_global in enumerate(default_indices):
            for j_idx in range(i_idx):
                j_global = default_indices[j_idx]
                matrix[i_global, j_global] = config.recovery_recovery
                matrix[j_global, i_global] = config.recovery_recovery

        return matrix
