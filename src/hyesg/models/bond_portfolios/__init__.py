"""Bond portfolio infrastructure for the hyesg ESG.

Provides Pydantic configuration models, systematic grid construction,
a registry of 44+ named benchmark portfolios, pure-JAX bond pricing
and analytics functions, and C#-compatible output path generation.

Example::

    from hyesg.models.bond_portfolios import (
        get_benchmark,
        list_benchmarks,
        zcb_price,
    )

    bench = get_benchmark("NominalGiltsBasket")
    print(bench.portfolio.holdings)
"""

from __future__ import annotations

from hyesg.models.bond_portfolios.analytics import (
    convexity,
    macaulay_duration,
    modified_duration,
    total_return,
    yield_to_maturity,
    z_spread,
)
from hyesg.models.bond_portfolios.benchmarks import (
    BENCHMARK_REGISTRY,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)
from hyesg.models.bond_portfolios.config import (
    BenchmarkPortfolioConfig,
    BondHoldingConfig,
    BondPortfolioConfig,
    BondType,
    MaturityType,
    RebalancingConfig,
)
from hyesg.models.bond_portfolios.grid import (
    STANDARD_COUPONS,
    STANDARD_TENORS,
    build_bond_grid,
    build_corporate_grid,
    build_government_grid,
    build_index_linked_grid,
)
from hyesg.models.bond_portfolios.output_paths import (
    OUTPUT_FIELDS,
    all_output_paths,
    portfolio_output_path,
)
from hyesg.models.bond_portfolios.pricing import (
    basket_yield,
    coupon_bond_price,
    credit_bond_price,
    index_linked_bond_price,
    zcb_price,
)

__all__ = [
    # Config
    "BenchmarkPortfolioConfig",
    "BondHoldingConfig",
    "BondPortfolioConfig",
    "BondType",
    "MaturityType",
    "RebalancingConfig",
    # Grid
    "STANDARD_COUPONS",
    "STANDARD_TENORS",
    "build_bond_grid",
    "build_corporate_grid",
    "build_government_grid",
    "build_index_linked_grid",
    # Benchmarks
    "BENCHMARK_REGISTRY",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    # Pricing
    "basket_yield",
    "coupon_bond_price",
    "credit_bond_price",
    "index_linked_bond_price",
    "zcb_price",
    # Analytics
    "convexity",
    "macaulay_duration",
    "modified_duration",
    "total_return",
    "yield_to_maturity",
    "z_spread",
    # Output paths
    "OUTPUT_FIELDS",
    "all_output_paths",
    "portfolio_output_path",
]
