"""Output path registration matching C# ESG naming conventions.

The C# engine emits bond portfolio outputs using the pattern::

    BondPortfolio({name}).{field}

This module provides helpers to generate those canonical path strings
so that Python output keys are directly comparable to C# results.
"""

from __future__ import annotations

OUTPUT_FIELDS: list[str] = [
    "TotalReturnIndex",
    "BasketYield",
    "Duration",
    "Value",
    "YieldAnnualised",
]


def portfolio_output_path(portfolio_name: str, field: str) -> str:
    """Generate a C#-compatible output path for a bond portfolio field.

    Args:
        portfolio_name: Canonical portfolio name (e.g. ``'NominalGiltsBasket'``).
        field: Output field name (e.g. ``'TotalReturnIndex'``).

    Returns:
        Formatted output path string.

    Raises:
        ValueError: If *field* is not in ``OUTPUT_FIELDS``.
    """
    if field not in OUTPUT_FIELDS:
        msg = f"Unknown output field '{field}'. Valid fields: {OUTPUT_FIELDS}"
        raise ValueError(msg)
    return f"BondPortfolio({portfolio_name}).{field}"


def all_output_paths(portfolio_name: str) -> dict[str, str]:
    """Generate all output paths for a portfolio.

    Args:
        portfolio_name: Canonical portfolio name.

    Returns:
        Mapping of field name to full output path string.
    """
    return {
        field: portfolio_output_path(portfolio_name, field)
        for field in OUTPUT_FIELDS
    }


__all__ = [
    "OUTPUT_FIELDS",
    "all_output_paths",
    "portfolio_output_path",
]
