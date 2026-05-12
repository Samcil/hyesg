"""Tests for CorrelationAssembler (CSV-based interface)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hyesg.core.matrix import SymmetricLabelledMatrix
from hyesg.io.correlation_csv import CsvCorrelationSource
from hyesg.orchestration.correlation_assembler import (
    CorrelationAssembler,
    CreditCorrelationConfig,
)
from hyesg.orchestration.label_registry import DzFactorLabelRegistry


class FakeCorrelationSource:
    """In-memory correlation source for unit testing."""

    def __init__(self, blocks: dict[str, tuple[list[str], np.ndarray]]) -> None:
        self._blocks = blocks

    def load_correlation_block(self, name: str) -> tuple[list[str], np.ndarray]:
        if name not in self._blocks:
            raise FileNotFoundError(f"Block not found: {name}")
        return self._blocks[name]

    def available_blocks(self) -> list[str]:
        return list(self._blocks.keys())


@pytest.fixture()
def simple_source() -> FakeCorrelationSource:
    """Source with two small blocks (2×2 and 3×3)."""
    block_a = (
        ["factor_a1", "factor_a2"],
        np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    block_b = (
        ["factor_b1", "factor_b2", "factor_b3"],
        np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]),
    )
    return FakeCorrelationSource({"BlockA": block_a, "BlockB": block_b})


class TestLoadBlocks:
    """Tests for load_blocks() and registry creation."""

    def test_builds_registry(self, simple_source: FakeCorrelationSource) -> None:
        """load_blocks() returns a registry with all labels."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        registry = assembler.load_blocks()
        assert isinstance(registry, DzFactorLabelRegistry)
        assert registry.size == 5

    def test_label_order_matches_block_order(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Labels appear in block assembly order."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        labels = assembler.factor_labels()
        assert labels == [
            "factor_a1",
            "factor_a2",
            "factor_b1",
            "factor_b2",
            "factor_b3",
        ]

    def test_registry_not_available_before_load(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Accessing registry before load_blocks() raises RuntimeError."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        with pytest.raises(RuntimeError, match="Call load_blocks"):
            _ = assembler.registry

    def test_missing_block_raises(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Missing block name raises FileNotFoundError."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "NonExistent")
        )
        with pytest.raises(FileNotFoundError):
            assembler.load_blocks()


class TestAssembly:
    """Tests for assemble()."""

    def test_result_is_symmetric_labelled_matrix(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Result is a SymmetricLabelledMatrix."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        result = assembler.assemble()
        assert isinstance(result, SymmetricLabelledMatrix)

    def test_correct_dimensions(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Result has correct shape (sum of block dimensions)."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        result = assembler.assemble()
        assert result.shape == (5, 5)

    def test_diagonal_is_ones(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Diagonal entries are all 1.0."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        result = assembler.assemble()
        for label in assembler.factor_labels():
            assert float(result[label, label]) == pytest.approx(1.0)

    def test_intra_block_correlations_preserved(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Correlations within a block are preserved."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        result = assembler.assemble()
        assert float(result["factor_a1", "factor_a2"]) == pytest.approx(0.5)
        assert float(result["factor_b1", "factor_b2"]) == pytest.approx(0.3)

    def test_cross_block_correlations_are_zero(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Correlations between different blocks are zero (block-diagonal)."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        result = assembler.assemble()
        assert float(result["factor_a1", "factor_b1"]) == pytest.approx(0.0)
        assert float(result["factor_a2", "factor_b3"]) == pytest.approx(0.0)

    def test_symmetry(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Result is symmetric."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        result = assembler.assemble()
        assert float(result["factor_a1", "factor_a2"]) == pytest.approx(
            float(result["factor_a2", "factor_a1"])
        )


class TestCreditCorrelations:
    """Tests for programmatic credit correlation overlay."""

    def test_credit_correlations_applied(self) -> None:
        """Credit config overlays intensity-intensity correlations."""
        block = (
            ["credit_a", "credit_b", "default_x"],
            np.eye(3, dtype=np.float64),
        )
        source = FakeCorrelationSource({"Credits": block})
        assembler = CorrelationAssembler(source, block_names=("Credits",))
        assembler.load_blocks()

        config = CreditCorrelationConfig(
            intensity_intensity=0.93,
            intensity_recovery=0.7,
            recovery_recovery=0.8,
            credit_class_labels=("credit_a", "credit_b"),
            default_and_liquidity_labels=("default_x",),
        )
        result = assembler.assemble(credit_config=config)

        # Intensity-intensity
        assert float(result["credit_a", "credit_b"]) == pytest.approx(
            0.93, abs=1e-6
        )
        # Intensity-recovery
        assert float(result["credit_a", "default_x"]) == pytest.approx(
            0.7, abs=1e-6
        )
        assert float(result["credit_b", "default_x"]) == pytest.approx(
            0.7, abs=1e-6
        )


class TestAssembleWithCholesky:
    """Tests for assemble_with_cholesky()."""

    def test_returns_cholesky_factor(
        self, simple_source: FakeCorrelationSource
    ) -> None:
        """Returns both correlation matrix and Cholesky factor."""
        assembler = CorrelationAssembler(
            simple_source, block_names=("BlockA", "BlockB")
        )
        assembler.load_blocks()
        corr_matrix, cholesky_L = assembler.assemble_with_cholesky()
        assert isinstance(corr_matrix, SymmetricLabelledMatrix)
        assert cholesky_L.shape == (5, 5)
