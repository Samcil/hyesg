"""Assemble full correlation matrix from economy-level blocks.

The ``CorrelationAssembler`` combines intra-economy blocks (rates,
equities, credit within the same economy) and inter-economy blocks
(cross-economy correlations) into a single ``SymmetricLabelledMatrix``
covering all Brownian factors in the simulation.
"""

from __future__ import annotations

import jax.numpy as jnp

from hyesg.config.economy import Economy
from hyesg.core.matrix import LabelledMatrix, SymmetricLabelledMatrix


class CorrelationAssembler:
    """Assemble full correlation matrix from economy blocks.

    Combines:
    - Intra-economy blocks (rates-equity-credit within same economy).
    - Inter-economy blocks (GBP nominal vs USD nominal, etc.).

    The resulting matrix is symmetric with ones on the diagonal.
    """

    def __init__(self, economies: list[Economy]) -> None:
        """Initialize with economy specifications.

        Args:
            economies: List of economy specifications whose models
                define the Brownian factor labels.
        """
        self._economies = economies
        self._labels = self._build_factor_labels()

    def _build_factor_labels(self) -> list[str]:
        """Build ordered list of all Brownian factor labels.

        Order follows: domestic economy first, then foreign economies
        in the order provided. Within each economy, models follow
        ``Economy.all_models`` order.

        Returns:
            Ordered list of factor labels.
        """
        labels: list[str] = []
        # Domestic first
        for econ in self._economies:
            if econ.is_domestic:
                for model in econ.all_models:
                    labels.append(model.label)
        # Then foreign in order
        for econ in self._economies:
            if not econ.is_domestic:
                for model in econ.all_models:
                    labels.append(model.label)
        return labels

    def factor_labels(self) -> list[str]:
        """Return ordered list of all Brownian factor labels.

        Returns:
            List of labels, domestic economy first.
        """
        return list(self._labels)

    def assemble(
        self,
        intra_blocks: dict[str, SymmetricLabelledMatrix],
        inter_blocks: dict[tuple[str, str], LabelledMatrix],
    ) -> SymmetricLabelledMatrix:
        """Build full correlation matrix from blocks.

        Args:
            intra_blocks: Mapping from economy name to the symmetric
                correlation block for that economy's factors.
            inter_blocks: Mapping from ``(economy_a, economy_b)`` to the
                rectangular correlation block between those economies'
                factors.

        Returns:
            A ``SymmetricLabelledMatrix`` of shape ``(n, n)`` covering
            all Brownian factors across all economies.
        """
        n = len(self._labels)
        label_index = {label: i for i, label in enumerate(self._labels)}
        matrix = jnp.eye(n, dtype=jnp.float64)

        # Fill intra-economy diagonal blocks
        for econ in self._economies:
            if econ.name not in intra_blocks:
                continue
            block = intra_blocks[econ.name]
            for row_label in block.row_labels:
                for col_label in block.col_labels:
                    if row_label in label_index and col_label in label_index:
                        i = label_index[row_label]
                        j = label_index[col_label]
                        matrix = matrix.at[i, j].set(block[row_label, col_label])

        # Fill inter-economy off-diagonal blocks (both triangles)
        for (econ_a, econ_b), block in inter_blocks.items():
            for row_label in block.row_labels:
                for col_label in block.col_labels:
                    if row_label in label_index and col_label in label_index:
                        i = label_index[row_label]
                        j = label_index[col_label]
                        val = block[row_label, col_label]
                        matrix = matrix.at[i, j].set(val)
                        matrix = matrix.at[j, i].set(val)

        return SymmetricLabelledMatrix(matrix, self._labels)
