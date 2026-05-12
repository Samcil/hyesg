"""Tests for output path generation."""

from __future__ import annotations

import pytest

from hyesg.models.bond_portfolios.output_paths import (
    OUTPUT_FIELDS,
    all_output_paths,
    portfolio_output_path,
)


class TestPortfolioOutputPath:
    """Tests for single output path generation."""

    def test_format(self) -> None:
        """Output path matches C# convention."""
        path = portfolio_output_path("NominalGiltsBasket", "TotalReturnIndex")
        assert path == "BondPortfolio(NominalGiltsBasket).TotalReturnIndex"

    def test_basket_yield(self) -> None:
        """BasketYield field generates correct path."""
        path = portfolio_output_path("CorpBasket", "BasketYield")
        assert path == "BondPortfolio(CorpBasket).BasketYield"

    def test_invalid_field_raises(self) -> None:
        """Unknown field name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output field"):
            portfolio_output_path("test", "InvalidField")

    def test_all_fields_valid(self) -> None:
        """All OUTPUT_FIELDS produce valid paths."""
        for field in OUTPUT_FIELDS:
            path = portfolio_output_path("X", field)
            assert path.startswith("BondPortfolio(X).")
            assert field in path


class TestAllOutputPaths:
    """Tests for bulk output path generation."""

    def test_returns_all_fields(self) -> None:
        """all_output_paths returns a path for every field."""
        paths = all_output_paths("MyPortfolio")
        assert len(paths) == len(OUTPUT_FIELDS)
        for field in OUTPUT_FIELDS:
            assert field in paths

    def test_path_values(self) -> None:
        """Each path value matches the expected format."""
        paths = all_output_paths("TestPort")
        for field, path in paths.items():
            assert path == f"BondPortfolio(TestPort).{field}"
