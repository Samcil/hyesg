"""Integration tests: full CSV correlation assembly pipeline.

Loads the 7 reference CSV blocks shipped in ``src/hyesg/data/correlations/``
and verifies the assembled 150×150 correlation matrix properties.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hyesg.core.matrix import SymmetricLabelledMatrix
from hyesg.io.correlation_csv import (
    CORRELATION_CSV_NAMES,
    CsvCorrelationSource,
)
from hyesg.orchestration.correlation_assembler import CorrelationAssembler
from hyesg.orchestration.label_registry import DzFactorLabelRegistry

# Path to the shipped correlation CSV files.
_CSV_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "hyesg"
    / "data"
    / "correlations"
)


@pytest.fixture()
def csv_source() -> CsvCorrelationSource:
    """CsvCorrelationSource pointed at the reference data directory."""
    return CsvCorrelationSource(_CSV_DIR)


@pytest.fixture()
def assembled(csv_source: CsvCorrelationSource) -> SymmetricLabelledMatrix:
    """Fully assembled 150×150 correlation matrix."""
    assembler = CorrelationAssembler(csv_source)
    assembler.load_blocks()
    return assembler.assemble()


class TestCsvCorrelationSource:
    """Tests for loading individual CSV blocks."""

    def test_available_blocks_lists_all_seven(
        self, csv_source: CsvCorrelationSource
    ) -> None:
        """All 7 canonical blocks are available."""
        available = csv_source.available_blocks()
        for name in CORRELATION_CSV_NAMES:
            assert name in available

    @pytest.mark.parametrize(
        ("name", "expected_size"),
        [
            ("GbpNominalsAndRealRates", 15),
            ("GbpInflations", 12),
            ("ForeignNominals", 10),
            ("ExchangeRates", 10),
            ("EquitiesAndGrowthAssets", 80),
            ("EquityJumps", 20),
            ("LpiLiquidityGaps", 3),
        ],
    )
    def test_block_dimensions(
        self, csv_source: CsvCorrelationSource, name: str, expected_size: int
    ) -> None:
        """Each block has the expected dimension."""
        labels, data = csv_source.load_correlation_block(name)
        assert len(labels) == expected_size
        assert data.shape == (expected_size, expected_size)

    def test_block_diagonal_is_one(
        self, csv_source: CsvCorrelationSource
    ) -> None:
        """Diagonal of each block is 1.0."""
        for name in CORRELATION_CSV_NAMES:
            _, data = csv_source.load_correlation_block(name)
            np.testing.assert_allclose(np.diag(data), 1.0, atol=1e-12)

    def test_block_is_symmetric(
        self, csv_source: CsvCorrelationSource
    ) -> None:
        """Each block matrix is symmetric."""
        for name in CORRELATION_CSV_NAMES:
            _, data = csv_source.load_correlation_block(name)
            np.testing.assert_allclose(data, data.T, atol=1e-14)


class TestFullAssembly:
    """Tests for the fully assembled 150×150 matrix."""

    def test_assembled_shape(
        self, assembled: SymmetricLabelledMatrix
    ) -> None:
        """Assembled matrix is 150×150."""
        assert assembled.shape == (150, 150)

    def test_assembled_diagonal_is_one(
        self, assembled: SymmetricLabelledMatrix
    ) -> None:
        """Diagonal entries are all 1.0."""
        diag = np.diag(np.asarray(assembled.data))
        np.testing.assert_allclose(diag, 1.0, atol=1e-12)

    def test_assembled_is_symmetric(
        self, assembled: SymmetricLabelledMatrix
    ) -> None:
        """Full matrix is symmetric."""
        data = np.asarray(assembled.data)
        np.testing.assert_allclose(data, data.T, atol=1e-14)

    def test_label_count_matches_shape(
        self, assembled: SymmetricLabelledMatrix
    ) -> None:
        """Label count matches matrix dimension."""
        assert len(assembled.row_labels) == 150

    def test_no_duplicate_labels(
        self, assembled: SymmetricLabelledMatrix
    ) -> None:
        """All 150 labels are unique."""
        labels = assembled.row_labels
        assert len(labels) == len(set(labels))

    def test_values_in_valid_range(
        self, assembled: SymmetricLabelledMatrix
    ) -> None:
        """All correlation values are in [-1, 1]."""
        data = np.asarray(assembled.data)
        assert np.all(data >= -1.0 - 1e-12)
        assert np.all(data <= 1.0 + 1e-12)


class TestRegistryFromAssembly:
    """Tests for DzFactorLabelRegistry built during assembly."""

    def test_registry_size(self, csv_source: CsvCorrelationSource) -> None:
        """Registry has 150 entries after loading all blocks."""
        assembler = CorrelationAssembler(csv_source)
        registry = assembler.load_blocks()
        assert registry.size == 150

    def test_registry_index_lookup(
        self, csv_source: CsvCorrelationSource
    ) -> None:
        """Registry supports O(1) label → index lookup."""
        assembler = CorrelationAssembler(csv_source)
        registry = assembler.load_blocks()
        # First label is index 0
        first_label = assembler.factor_labels()[0]
        assert registry.index_of(first_label) == 0
        # Last label is index 149
        last_label = assembler.factor_labels()[-1]
        assert registry.index_of(last_label) == 149

    def test_registry_batch_lookup(
        self, csv_source: CsvCorrelationSource
    ) -> None:
        """indices_of returns correct indices for multiple labels."""
        assembler = CorrelationAssembler(csv_source)
        registry = assembler.load_blocks()
        labels = assembler.factor_labels()
        subset = [labels[0], labels[10], labels[50]]
        indices = registry.indices_of(subset)
        assert indices == [0, 10, 50]


class TestCholeskyIntegration:
    """Tests for full Cholesky factorisation of assembled matrix."""

    def test_cholesky_succeeds(
        self, csv_source: CsvCorrelationSource
    ) -> None:
        """Cholesky factorisation succeeds on the assembled matrix."""
        assembler = CorrelationAssembler(csv_source)
        assembler.load_blocks()
        corr, chol_L = assembler.assemble_with_cholesky()
        assert chol_L.shape == (150, 150)

    def test_cholesky_reconstructs_correlation(
        self, csv_source: CsvCorrelationSource
    ) -> None:
        """L @ L.T reconstructs the correlation matrix."""
        assembler = CorrelationAssembler(csv_source)
        assembler.load_blocks()
        corr, chol_L = assembler.assemble_with_cholesky()
        reconstructed = chol_L @ chol_L.T
        np.testing.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(corr.data),
            atol=1e-8,
        )
