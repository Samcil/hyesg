"""Tests for CorrelationAssembler."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hyesg.config.economy import Economy, EconomyModelConfig
from hyesg.core.matrix import LabelledMatrix, SymmetricLabelledMatrix
from hyesg.orchestration.correlation_assembler import CorrelationAssembler


@pytest.fixture()
def two_economy_setup() -> tuple[Economy, Economy]:
    """Two economies: domestic with 2 models, foreign with 2 models."""
    domestic = Economy(
        name="DOM",
        is_domestic=True,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp", label="dom_nominal"
        ),
        equity_models=[
            EconomyModelConfig(model_type="gbm", label="dom_eq1"),
        ],
    )
    foreign = Economy(
        name="FOR",
        is_domestic=False,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp", label="for_nominal"
        ),
        fx_model=EconomyModelConfig(
            model_type="fx_gbm", label="for_fx"
        ),
    )
    return domestic, foreign


class TestFactorLabels:
    """Tests for factor_labels()."""

    def test_domestic_first(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """Domestic economy labels come first."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])
        labels = assembler.factor_labels()
        assert labels[0] == "dom_nominal"
        assert labels[1] == "dom_eq1"

    def test_foreign_after_domestic(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """Foreign economy labels come after domestic."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])
        labels = assembler.factor_labels()
        assert labels[2] == "for_nominal"
        assert labels[3] == "for_fx"

    def test_total_count(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """Total label count matches sum of all models."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])
        labels = assembler.factor_labels()
        total = len(domestic.all_models) + len(foreign.all_models)
        assert len(labels) == total

    def test_labels_are_unique(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """All labels are unique."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])
        labels = assembler.factor_labels()
        assert len(labels) == len(set(labels))


class TestAssembly:
    """Tests for assemble()."""

    def test_identity_with_no_blocks(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """With no blocks, result is identity matrix."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])
        result = assembler.assemble(intra_blocks={}, inter_blocks={})
        n = len(assembler.factor_labels())
        assert result.shape == (n, n)
        # Diagonal should be ones
        for i, label in enumerate(assembler.factor_labels()):
            assert float(result[label, label]) == pytest.approx(1.0)

    def test_symmetric_result(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """Assembly produces a symmetric matrix."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])

        # Create a simple intra-economy block for domestic
        dom_labels = ["dom_nominal", "dom_eq1"]
        dom_data = jnp.array([[1.0, 0.3], [0.3, 1.0]])
        dom_block = SymmetricLabelledMatrix(dom_data, dom_labels)

        result = assembler.assemble(
            intra_blocks={"DOM": dom_block},
            inter_blocks={},
        )
        # Check symmetry
        assert float(result["dom_nominal", "dom_eq1"]) == pytest.approx(0.3)
        assert float(result["dom_eq1", "dom_nominal"]) == pytest.approx(0.3)

    def test_inter_block_fills_both_triangles(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """Inter-economy block fills both upper and lower triangles."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])

        inter_data = jnp.array([[0.5, 0.2], [0.1, 0.4]])
        inter_block = LabelledMatrix(
            inter_data,
            row_labels=["dom_nominal", "dom_eq1"],
            col_labels=["for_nominal", "for_fx"],
        )

        result = assembler.assemble(
            intra_blocks={},
            inter_blocks={("DOM", "FOR"): inter_block},
        )
        # Value in upper triangle
        assert float(result["dom_nominal", "for_nominal"]) == pytest.approx(0.5)
        # Symmetric counterpart in lower triangle
        assert float(result["for_nominal", "dom_nominal"]) == pytest.approx(0.5)

    def test_correct_dimensions(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """Result matrix has correct dimensions."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])
        result = assembler.assemble(intra_blocks={}, inter_blocks={})
        n = len(assembler.factor_labels())
        assert result.shape == (n, n)

    def test_result_is_symmetric_labelled_matrix(
        self, two_economy_setup: tuple[Economy, Economy]
    ) -> None:
        """Result is a SymmetricLabelledMatrix."""
        domestic, foreign = two_economy_setup
        assembler = CorrelationAssembler([domestic, foreign])
        result = assembler.assemble(intra_blocks={}, inter_blocks={})
        assert isinstance(result, SymmetricLabelledMatrix)
