"""Registry of named benchmark bond portfolios.

Contains 44+ benchmark portfolios extracted from the C# ESG
``BondPortfolios.cs``. The registry is populated at module import
time via ``_register_all_benchmarks()``.
"""

from __future__ import annotations

from hyesg.core.enums import CreditClass, Liquidity
from hyesg.models.bond_portfolios.config import (
    BenchmarkPortfolioConfig,
    BondHoldingConfig,
    BondPortfolioConfig,
    BondType,
    MaturityType,
    RebalancingConfig,
)

BENCHMARK_REGISTRY: dict[str, BenchmarkPortfolioConfig] = {}


def register_benchmark(config: BenchmarkPortfolioConfig) -> None:
    """Register a named benchmark.

    Args:
        config: Benchmark portfolio configuration.
    """
    BENCHMARK_REGISTRY[config.name] = config


def get_benchmark(name: str) -> BenchmarkPortfolioConfig:
    """Look up a benchmark by name.

    Args:
        name: Canonical benchmark name.

    Returns:
        The registered benchmark configuration.

    Raises:
        KeyError: If the benchmark is not registered.
    """
    if name not in BENCHMARK_REGISTRY:
        msg = (
            f"Benchmark '{name}' not found. "
            f"Available: {sorted(BENCHMARK_REGISTRY.keys())}"
        )
        raise KeyError(msg)
    return BENCHMARK_REGISTRY[name]


def list_benchmarks() -> list[str]:
    """List all registered benchmark names.

    Returns:
        Sorted list of benchmark names.
    """
    return sorted(BENCHMARK_REGISTRY.keys())


# ── Helper builders ──────────────────────────────────────────────────


def _govt_holding(
    maturity: float,
    weight: float,
    economy: str = "GBP",
    maturity_type: MaturityType = MaturityType.FIXED,
) -> BondHoldingConfig:
    """Build a government bond holding."""
    return BondHoldingConfig(
        maturity=maturity,
        maturity_type=maturity_type,
        weight=weight,
        coupon_rate=0.0,
        coupon_frequency=0,
        is_at_par=False,
        bond_type=BondType.GOVERNMENT,
        credit_class=None,
        liquidity=Liquidity.HIGH,
        economy=economy,
    )


def _swap_holding(
    maturity: float,
    weight: float,
    economy: str = "GBP",
) -> BondHoldingConfig:
    """Build a swap-curve bond holding."""
    return BondHoldingConfig(
        maturity=maturity,
        maturity_type=MaturityType.FIXED,
        weight=weight,
        coupon_rate=0.0,
        coupon_frequency=0,
        is_at_par=False,
        bond_type=BondType.SWAP_CURVE,
        credit_class=None,
        liquidity=Liquidity.HIGH,
        economy=economy,
    )


def _ilg_holding(
    maturity: float,
    weight: float,
    economy: str = "GBP",
) -> BondHoldingConfig:
    """Build an index-linked government bond holding."""
    return BondHoldingConfig(
        maturity=maturity,
        maturity_type=MaturityType.FIXED,
        weight=weight,
        coupon_rate=0.0,
        coupon_frequency=0,
        is_at_par=False,
        bond_type=BondType.INDEX_LINKED,
        credit_class=None,
        liquidity=Liquidity.HIGH,
        economy=economy,
    )


def _corp_holding(
    maturity: float,
    weight: float,
    credit_class: CreditClass,
    liquidity: Liquidity = Liquidity.HIGH,
    economy: str = "GBP",
) -> BondHoldingConfig:
    """Build a corporate bond holding."""
    return BondHoldingConfig(
        maturity=maturity,
        maturity_type=MaturityType.FIXED,
        weight=weight,
        coupon_rate=0.0,
        coupon_frequency=0,
        is_at_par=False,
        bond_type=BondType.CORPORATE,
        credit_class=credit_class,
        liquidity=liquidity,
        economy=economy,
    )


def _make_corp_portfolio(
    name: str,
    maturities: list[float],
    allocations: list[float],
    credit_classes: list[CreditClass],
    liquidity: Liquidity = Liquidity.HIGH,
    economy: str = "GBP",
    n_issues: int = 1,
    nominal_economy: str = "GBP",
    description: str = "",
) -> BenchmarkPortfolioConfig:
    """Build and register a corporate bond benchmark."""
    holdings = [
        _corp_holding(mat, alloc, cc, liquidity, economy)
        for mat, alloc, cc in zip(maturities, allocations, credit_classes, strict=True)
    ]
    portfolio = BondPortfolioConfig(
        name=name,
        holdings=holdings,
        nominal_economy=nominal_economy,
        n_issues_per_tranche=n_issues,
    )
    return BenchmarkPortfolioConfig(
        name=name,
        description=description,
        portfolio=portfolio,
    )


def _make_multi_economy_corp_portfolio(
    name: str,
    economy_weights: dict[str, float],
    maturity: float,
    credit_allocations: dict[CreditClass, float],
    liquidity: Liquidity = Liquidity.HIGH,
    n_issues: int = 1,
    nominal_economy: str = "GBP",
    description: str = "",
) -> BenchmarkPortfolioConfig:
    """Build a multi-economy corporate bond benchmark.

    Creates one holding per (economy, credit_class) combination with
    weight = economy_weight × credit_allocation.
    """
    holdings: list[BondHoldingConfig] = []
    for econ, econ_w in economy_weights.items():
        for cc, cc_w in credit_allocations.items():
            holdings.append(
                _corp_holding(maturity, econ_w * cc_w, cc, liquidity, econ)
            )
    portfolio = BondPortfolioConfig(
        name=name,
        holdings=holdings,
        nominal_economy=nominal_economy,
        n_issues_per_tranche=n_issues,
    )
    return BenchmarkPortfolioConfig(
        name=name,
        description=description,
        portfolio=portfolio,
    )


def _make_govt_basket(
    name: str,
    maturities: list[float],
    allocations: list[float],
    economy: str = "GBP",
    description: str = "",
) -> BenchmarkPortfolioConfig:
    """Build a government bond basket benchmark."""
    holdings = [
        _govt_holding(mat, alloc, economy)
        for mat, alloc in zip(maturities, allocations, strict=True)
    ]
    portfolio = BondPortfolioConfig(
        name=name,
        holdings=holdings,
        nominal_economy=economy,
    )
    return BenchmarkPortfolioConfig(
        name=name,
        description=description,
        portfolio=portfolio,
    )


# ── Registration ─────────────────────────────────────────────────────


def _register_all_benchmarks() -> None:
    """Register all 44+ benchmarks. Called at module import."""
    _register_2022_mc_portfolios()
    _register_pre2022_portfolios()
    _register_trw_portfolios()


def _register_2022_mc_portfolios() -> None:
    """Register 2022 MC benchmark portfolios."""

    # 1. Sterling Corporate Index GBP
    register_benchmark(
        _make_corp_portfolio(
            "Sterling Corporate Index GBP",
            maturities=[9.0, 9.0, 9.0],
            allocations=[0.10, 0.40, 0.50],
            credit_classes=[CreditClass.AA, CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=2,
            description="ICE BofA Sterling Corporate Index",
        )
    )

    # 2. US Leveraged Loan GBP
    register_benchmark(
        _make_corp_portfolio(
            "US Leveraged Loan GBP",
            maturities=[5.0, 5.0, 5.0, 5.0],
            allocations=[0.05, 0.30, 0.55, 0.10],
            credit_classes=[
                CreditClass.BBB, CreditClass.BB, CreditClass.B, CreditClass.CCC,
            ],
            liquidity=Liquidity.MEDIUM,
            economy="USD",
            n_issues=2,
        )
    )

    # 3. US Leveraged Loan Illiquid GBP
    register_benchmark(
        _make_corp_portfolio(
            "US Leveraged Loan Illiquid GBP",
            maturities=[5.0, 5.0, 5.0, 5.0],
            allocations=[0.05, 0.30, 0.55, 0.10],
            credit_classes=[
                CreditClass.BBB, CreditClass.BB, CreditClass.B, CreditClass.CCC,
            ],
            liquidity=Liquidity.LOW,
            economy="USD",
            n_issues=2,
        )
    )

    # 4. US Leveraged Loan GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "US Leveraged Loan GBP Hedged",
            maturities=[5.0, 5.0, 5.0, 5.0],
            allocations=[0.05, 0.30, 0.55, 0.10],
            credit_classes=[
                CreditClass.BBB, CreditClass.BB, CreditClass.B, CreditClass.CCC,
            ],
            liquidity=Liquidity.MEDIUM,
            economy="GBP",
            n_issues=2,
        )
    )

    # 5. US Leveraged Loan Illiquid GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "US Leveraged Loan Illiquid GBP Hedged",
            maturities=[5.0, 5.0, 5.0, 5.0],
            allocations=[0.05, 0.30, 0.55, 0.10],
            credit_classes=[
                CreditClass.BBB, CreditClass.BB, CreditClass.B, CreditClass.CCC,
            ],
            liquidity=Liquidity.LOW,
            economy="GBP",
            n_issues=2,
        )
    )

    # 6. EU Leveraged Loan GBP
    register_benchmark(
        _make_corp_portfolio(
            "EU Leveraged Loan GBP",
            maturities=[5.0, 5.0, 5.0],
            allocations=[0.25, 0.60, 0.15],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.MEDIUM,
            economy="EUR",
            n_issues=2,
        )
    )

    # 7. EU Leveraged Loan Illiquid GBP
    register_benchmark(
        _make_corp_portfolio(
            "EU Leveraged Loan Illiquid GBP",
            maturities=[5.0, 5.0, 5.0],
            allocations=[0.25, 0.60, 0.15],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.LOW,
            economy="EUR",
            n_issues=2,
        )
    )

    # 8. EU Leveraged Loan GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "EU Leveraged Loan GBP Hedged",
            maturities=[5.0, 5.0, 5.0],
            allocations=[0.25, 0.60, 0.15],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.MEDIUM,
            economy="GBP",
            n_issues=2,
        )
    )

    # 9. EU Leveraged Loan Illiquid GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "EU Leveraged Loan Illiquid GBP Hedged",
            maturities=[5.0, 5.0, 5.0],
            allocations=[0.25, 0.60, 0.15],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.LOW,
            economy="GBP",
            n_issues=2,
        )
    )

    # 10. US Distressed Debt GBP
    register_benchmark(
        _make_corp_portfolio(
            "US Distressed Debt GBP",
            maturities=[3.0, 3.0],
            allocations=[0.20, 0.80],
            credit_classes=[CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.LOW,
            economy="USD",
            n_issues=3,
        )
    )

    # 11. US Distressed Debt GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "US Distressed Debt GBP Hedged",
            maturities=[3.0, 3.0],
            allocations=[0.20, 0.80],
            credit_classes=[CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.LOW,
            economy="GBP",
            n_issues=3,
        )
    )

    # 12. Global Distressed Debt GBP
    register_benchmark(
        _make_multi_economy_corp_portfolio(
            "Global Distressed Debt GBP",
            economy_weights={"EUR": 0.15, "USD": 0.85},
            maturity=3.0,
            credit_allocations={CreditClass.CCC: 1.0},
            liquidity=Liquidity.LOW,
            n_issues=3,
        )
    )

    # 13. Global Distressed Debt GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "Global Distressed Debt GBP Hedged",
            maturities=[3.0],
            allocations=[1.0],
            credit_classes=[CreditClass.CCC],
            liquidity=Liquidity.LOW,
            economy="GBP",
            n_issues=3,
        )
    )

    # 14. Emerging Market Debt (hard currency) GBP
    register_benchmark(
        _make_corp_portfolio(
            "Emerging Market Debt (hard currency) GBP",
            maturities=[8.0],
            allocations=[1.0],
            credit_classes=[CreditClass.BB],
            liquidity=Liquidity.HIGH,
            economy="USD",
            n_issues=6,
        )
    )

    # 15. Emerging Market Debt (hard currency) GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "Emerging Market Debt (hard currency) GBP Hedged",
            maturities=[8.0],
            allocations=[1.0],
            credit_classes=[CreditClass.BB],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=6,
        )
    )

    # 16. Emerging Market Debt (local currency) GBP — government bond
    _em_holdings = [
        _govt_holding(5.0, 1.0, economy="EM"),
    ]
    _em_portfolio = BondPortfolioConfig(
        name="Emerging Market Debt (local currency) GBP",
        holdings=_em_holdings,
        nominal_economy="GBP",
    )
    register_benchmark(
        BenchmarkPortfolioConfig(
            name="Emerging Market Debt (local currency) GBP",
            portfolio=_em_portfolio,
        )
    )

    # 17. US High Yield Debt GBP
    register_benchmark(
        _make_corp_portfolio(
            "US High Yield Debt GBP",
            maturities=[4.0, 4.0, 4.0],
            allocations=[0.55, 0.35, 0.10],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.HIGH,
            economy="USD",
            n_issues=2,
        )
    )

    # 18. US High Yield Debt GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "US High Yield Debt GBP Hedged",
            maturities=[4.0, 4.0, 4.0],
            allocations=[0.55, 0.35, 0.10],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=2,
        )
    )

    # 19. EU High Yield Debt GBP
    register_benchmark(
        _make_corp_portfolio(
            "EU High Yield Debt GBP",
            maturities=[3.0, 3.0, 3.0],
            allocations=[0.70, 0.25, 0.05],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.HIGH,
            economy="EUR",
            n_issues=2,
        )
    )

    # 20. EU High Yield Debt GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "EU High Yield Debt GBP Hedged",
            maturities=[3.0, 3.0, 3.0],
            allocations=[0.70, 0.25, 0.05],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=2,
        )
    )

    # 21. Global High Yield Debt GBP
    register_benchmark(
        _make_multi_economy_corp_portfolio(
            "Global High Yield Debt GBP",
            economy_weights={"EUR": 0.20, "USD": 0.80},
            maturity=4.0,
            credit_allocations={
                CreditClass.BB: 0.60,
                CreditClass.B: 0.30,
                CreditClass.CCC: 0.10,
            },
            liquidity=Liquidity.HIGH,
            n_issues=1,
        )
    )

    # 22. Global High Yield Debt GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "Global High Yield Debt GBP Hedged",
            maturities=[4.0, 4.0, 4.0],
            allocations=[0.60, 0.30, 0.10],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=1,
        )
    )

    # 23. UK Infrastructure Debt GBP
    register_benchmark(
        _make_corp_portfolio(
            "UK Infrastructure Debt GBP",
            maturities=[10.0, 10.0],
            allocations=[0.25, 0.75],
            credit_classes=[CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=3,
        )
    )

    # 24. US Infrastructure Debt GBP
    register_benchmark(
        _make_corp_portfolio(
            "US Infrastructure Debt GBP",
            maturities=[10.0, 10.0, 10.0],
            allocations=[0.05, 0.30, 0.65],
            credit_classes=[CreditClass.AA, CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="USD",
            n_issues=2,
        )
    )

    # 25. US Infrastructure Debt GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "US Infrastructure Debt GBP Hedged",
            maturities=[10.0, 10.0, 10.0],
            allocations=[0.05, 0.30, 0.65],
            credit_classes=[CreditClass.AA, CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=2,
        )
    )

    # 26. EU Infrastructure Debt GBP
    register_benchmark(
        _make_corp_portfolio(
            "EU Infrastructure Debt GBP",
            maturities=[6.0, 6.0],
            allocations=[0.30, 0.70],
            credit_classes=[CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="EUR",
            n_issues=3,
        )
    )

    # 27. EU Infrastructure Debt GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "EU Infrastructure Debt GBP Hedged",
            maturities=[6.0, 6.0],
            allocations=[0.30, 0.70],
            credit_classes=[CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=3,
        )
    )

    # 28. Global Corporate FI GBP
    register_benchmark(
        _make_multi_economy_corp_portfolio(
            "Global Corporate FI GBP",
            economy_weights={"GBP": 0.05, "EUR": 0.25, "USD": 0.70},
            maturity=7.0,
            credit_allocations={
                CreditClass.AA: 0.10,
                CreditClass.A: 0.40,
                CreditClass.BBB: 0.50,
            },
            liquidity=Liquidity.HIGH,
            n_issues=1,
        )
    )

    # 29. Global Corporate FI GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "Global Corporate FI GBP Hedged",
            maturities=[7.0, 7.0, 7.0],
            allocations=[0.10, 0.40, 0.50],
            credit_classes=[CreditClass.AA, CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=1,
        )
    )

    # 30. Global Aggregate FI GBP
    register_benchmark(
        _make_multi_economy_corp_portfolio(
            "Global Aggregate FI GBP",
            economy_weights={"GBP": 0.05, "EUR": 0.30, "USD": 0.65},
            maturity=8.0,
            credit_allocations={
                CreditClass.AAA: 0.40,
                CreditClass.AA: 0.15,
                CreditClass.A: 0.30,
                CreditClass.BBB: 0.15,
            },
            liquidity=Liquidity.HIGH,
            n_issues=1,
        )
    )

    # 31. Global Aggregate FI GBP Hedged
    register_benchmark(
        _make_corp_portfolio(
            "Global Aggregate FI GBP Hedged",
            maturities=[8.0, 8.0, 8.0, 8.0],
            allocations=[0.40, 0.15, 0.30, 0.15],
            credit_classes=[
                CreditClass.AAA, CreditClass.AA, CreditClass.A, CreditClass.BBB,
            ],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            n_issues=1,
        )
    )


def _register_pre2022_portfolios() -> None:
    """Register pre-2022 benchmark portfolios."""

    _basket_mats = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    _basket_allocs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.20]

    # 32. NominalGiltsBasket
    register_benchmark(
        _make_govt_basket(
            "NominalGiltsBasket",
            _basket_mats,
            _basket_allocs,
            description="UK nominal gilts basket",
        )
    )

    # 33. NominalSwapsBasket
    _swap_holdings = [
        _swap_holding(mat, alloc)
        for mat, alloc in zip(_basket_mats, _basket_allocs, strict=True)
    ]
    register_benchmark(
        BenchmarkPortfolioConfig(
            name="NominalSwapsBasket",
            description="UK nominal swaps basket",
            portfolio=BondPortfolioConfig(
                name="NominalSwapsBasket",
                holdings=_swap_holdings,
                nominal_economy="GBP",
            ),
        )
    )

    # 34. RealRPIBasket
    _rpi_holdings = [
        _ilg_holding(mat, alloc)
        for mat, alloc in zip(_basket_mats, _basket_allocs, strict=True)
    ]
    register_benchmark(
        BenchmarkPortfolioConfig(
            name="RealRPIBasket",
            description="UK real RPI-linked basket",
            portfolio=BondPortfolioConfig(
                name="RealRPIBasket",
                holdings=_rpi_holdings,
                nominal_economy="GBP",
            ),
        )
    )

    # 35. RealCPIBasket
    _cpi_holdings = [
        _ilg_holding(mat, alloc)
        for mat, alloc in zip(_basket_mats, _basket_allocs, strict=True)
    ]
    register_benchmark(
        BenchmarkPortfolioConfig(
            name="RealCPIBasket",
            description="UK real CPI-linked basket",
            portfolio=BondPortfolioConfig(
                name="RealCPIBasket",
                holdings=_cpi_holdings,
                nominal_economy="GBP",
            ),
        )
    )

    # 36. CorpBasket
    register_benchmark(
        _make_corp_portfolio(
            "CorpBasket",
            maturities=_basket_mats,
            allocations=_basket_allocs,
            credit_classes=[CreditClass.A] * len(_basket_mats),
            liquidity=Liquidity.HIGH,
            economy="GBP",
            description="UK corporate bond basket (A-rated)",
        )
    )

    # 37. 17yrCorpBasket
    register_benchmark(
        _make_corp_portfolio(
            "17yrCorpBasket",
            maturities=[17.0],
            allocations=[1.0],
            credit_classes=[CreditClass.A],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            description="UK 17-year corporate bond (A-rated)",
        )
    )

    # 38. SterlingInvestmentGradeIndex
    register_benchmark(
        _make_corp_portfolio(
            "SterlingInvestmentGradeIndex",
            maturities=[8.0, 8.0, 8.0],
            allocations=[0.15, 0.40, 0.45],
            credit_classes=[CreditClass.AA, CreditClass.A, CreditClass.BBB],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            description="Sterling investment grade index",
        )
    )

    # 39. GlobalHighYieldIndex
    register_benchmark(
        _make_corp_portfolio(
            "GlobalHighYieldIndex",
            maturities=[4.0, 4.0, 4.0],
            allocations=[0.55, 0.35, 0.10],
            credit_classes=[CreditClass.BB, CreditClass.B, CreditClass.CCC],
            liquidity=Liquidity.HIGH,
            economy="GBP",
            description="Global high yield index",
        )
    )

    # 40. Rolling1yrGovtBond
    _rolling_holdings = [
        _govt_holding(1.0, 1.0, maturity_type=MaturityType.ROLLING),
    ]
    register_benchmark(
        BenchmarkPortfolioConfig(
            name="Rolling1yrGovtBond",
            description="1-year rolling government bond",
            portfolio=BondPortfolioConfig(
                name="Rolling1yrGovtBond",
                holdings=_rolling_holdings,
                nominal_economy="GBP",
                rebalancing=RebalancingConfig(
                    strategy="maturity_and_allocation",
                    frequency=12,
                    rebalance_to_initial_maturity=True,
                ),
            ),
        )
    )

    # 41. 4yrGiltsBond
    _4yr_holdings = [_govt_holding(4.0, 1.0)]
    register_benchmark(
        BenchmarkPortfolioConfig(
            name="4yrGiltsBond",
            description="4-year UK gilt",
            portfolio=BondPortfolioConfig(
                name="4yrGiltsBond",
                holdings=_4yr_holdings,
                nominal_economy="GBP",
            ),
        )
    )

    # 42. 8yrGiltsBond
    _8yr_holdings = [_govt_holding(8.0, 1.0)]
    register_benchmark(
        BenchmarkPortfolioConfig(
            name="8yrGiltsBond",
            description="8-year UK gilt",
            portfolio=BondPortfolioConfig(
                name="8yrGiltsBond",
                holdings=_8yr_holdings,
                nominal_economy="GBP",
            ),
        )
    )


def _register_trw_portfolios() -> None:
    """Register TRW (Total Return World) variant portfolios."""

    # 43. CorpUltraShort_B_Medium
    register_benchmark(
        _make_corp_portfolio(
            "CorpUltraShort_B_Medium",
            maturities=[2.0],
            allocations=[1.0],
            credit_classes=[CreditClass.B],
            liquidity=Liquidity.MEDIUM,
            economy="GBP",
            n_issues=6,
            description="Ultra-short B-rated corporate (medium liquidity)",
        )
    )

    # 44. CorpUltraShort_B_Low
    register_benchmark(
        _make_corp_portfolio(
            "CorpUltraShort_B_Low",
            maturities=[2.0],
            allocations=[1.0],
            credit_classes=[CreditClass.B],
            liquidity=Liquidity.LOW,
            economy="GBP",
            n_issues=6,
            description="Ultra-short B-rated corporate (low liquidity)",
        )
    )


# Populate registry on import
_register_all_benchmarks()


__all__ = [
    "BENCHMARK_REGISTRY",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
]
