"""dZ-Factor Label Registry — single source of truth for label→index mapping.

Maps hierarchical dZ-factor labels from the correlation CSV files to
contiguous integer indices used in the full correlation matrix.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DzFactorLabelRegistry:
    """Registry mapping dZ-factor labels to matrix indices.

    Built from the ordered labels extracted from correlation CSV blocks.
    Provides O(1) label→index lookup and validates uniqueness.
    """

    def __init__(self, labels: list[str]) -> None:
        """Initialize with ordered label list.

        Args:
            labels: Ordered list of all dZ-factor labels. Must be unique.

        Raises:
            ValueError: If any labels are duplicated.
        """
        seen: set[str] = set()
        duplicates: list[str] = []
        for label in labels:
            if label in seen:
                duplicates.append(label)
            seen.add(label)

        if duplicates:
            raise ValueError(
                f"Duplicate dZ-factor labels: {sorted(set(duplicates))}"
            )

        self._labels = list(labels)
        self._index: dict[str, int] = {
            label: i for i, label in enumerate(labels)
        }
        logger.debug(
            "Built dZ-factor label registry with %d labels", len(labels)
        )

    @property
    def labels(self) -> list[str]:
        """Return ordered list of all labels.

        Returns:
            Copy of the label list.
        """
        return list(self._labels)

    @property
    def size(self) -> int:
        """Total number of registered labels."""
        return len(self._labels)

    def index_of(self, label: str) -> int:
        """Return the index for a given label.

        Args:
            label: The dZ-factor label string.

        Returns:
            Integer index in the full correlation matrix.

        Raises:
            KeyError: If label is not registered.
        """
        if label not in self._index:
            raise KeyError(f"Unknown dZ-factor label: {label!r}")
        return self._index[label]

    def contains(self, label: str) -> bool:
        """Check whether a label is registered.

        Args:
            label: The dZ-factor label string.

        Returns:
            True if the label exists in the registry.
        """
        return label in self._index

    def indices_of(self, labels: list[str]) -> list[int]:
        """Return indices for multiple labels.

        Args:
            labels: List of dZ-factor label strings.

        Returns:
            List of integer indices.

        Raises:
            KeyError: If any label is not registered.
        """
        return [self.index_of(label) for label in labels]

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        return f"DzFactorLabelRegistry(size={self.size})"
