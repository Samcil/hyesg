# hyesg — Python/JAX Economic Scenario Generator

## Product Requirements Document

**Date:** 2026-05-11
**Status:** Ready for Implementation
**Review Cycles:** 6 adversarial reviews, 0 critical issues remaining

---

## Problem Statement

Hymans Robertson's Economic Scenario Generator (ESG) is implemented across 6 C# repositories (~200+ files, 50+ financial models) spanning .NET Standard 2.0 and .NET Framework 4.7.2. The system generates Monte Carlo simulations of financial and economic variables (interest rates, inflation, credit spreads, equity returns, exchange rates) used in actuarial and investment decision-making.

The current C# implementation has several limitations:

1. **Hardware lock-in**: Runs only on CPU. No GPU/TPU acceleration for the compute-intensive Monte Carlo engine (5,000 trials × 1,200 timesteps × 50+ models).
2. **Platform lock-in**: Tightly coupled to Windows/.NET, Azure Functions, Azure ServiceBus, and Entity Framework — limiting deployment flexibility.
3. **Architectural debt**: Pull-based lazy evaluation (`LazySequence`) with mutable state makes parallelisation difficult. MEF-based plugin discovery is opaque.
4. **Maintenance burden**: 6 repositories with cross-cutting dependencies make changes slow and error-prone.
5. **Testing friction**: Mutable state and deep coupling make isolated unit testing difficult.
6. **Knowledge concentration**: Complex, undocumented financial mathematics spread across repositories.

## Solution

Build `hyesg` — a single Python package that reimplements the full ESG using JAX for hardware-transparent numerical computation. The package:

- **Runs identically on CPU, GPU, and TPU** ("write once, run anywhere") via JAX's XLA compilation
- **Consolidates 6 repositories into one package** with clean module boundaries
- **Replaces pull-based lazy evaluation with push-based `jax.lax.scan`** — the single most important architectural change
- **Uses Protocol-based contracts** (structural typing) instead of deep inheritance hierarchies
- **Provides Pydantic v2 configuration** with validation, serialisation, and fail-fast error handling
- **Decouples infrastructure** via pluggable protocols for storage, messaging, and market data
- **Documents all financial mathematics** with explicit SDEs, discretisation schemes, and pricing formulas

This is a **clean redesign**, not a faithful C# mirror. Mathematical logic is preserved exactly; architecture is rebuilt from first principles following SOLID and DRY.

---

## User Stories

### Core Simulation

1. As a **quant developer**, I want to define a simulation configuration in Python (models, parameters, time grid, trial count), so that I can run ESG simulations without C#/.NET.
2. As a **quant developer**, I want the simulation to run identically on CPU, GPU, and TPU without code changes, so that I can scale compute to match workload.
3. As a **quant developer**, I want to run 5,000 trials × 1,200 monthly timesteps × 50+ correlated models in a single batch, so that I can generate full 100-year projection sets.
4. As a **quant developer**, I want to configure multi-regime simulations (e.g., 3 regimes with different parameters), so that I can model economic regime switching.
5. As a **quant developer**, I want antithetic variance reduction applied automatically (trial pairs with negated/complemented shocks), so that I get tighter confidence intervals with fewer trials.
6. As a **quant developer**, I want reproducible results via deterministic PRNG seeding (JAX ThreeFry), so that I can verify and debug simulations.
7. As a **quant developer**, I want the simulation engine to topologically sort model dependencies and evaluate them in correct order, so that dependent models always see up-to-date parent state.

### Interest Rate Models

8. As a **quant developer**, I want to simulate CIR (Cox-Ingersoll-Ross) short rate processes with floor-at-zero and Feller condition checking, so that I can model mean-reverting positive rates.
9. As a **quant developer**, I want to simulate CIR++ (CIR with deterministic shift φ(t)) that exactly fits the initial market term structure, so that I can produce arbitrage-free rate paths.
10. As a **quant developer**, I want to simulate CIR2++ (two-factor CIR++ with blending), so that I can model both risk-neutral and real-world rate dynamics.
11. As a **quant developer**, I want to simulate G1++ (one-factor Gaussian short rate with analytical φ), so that I can model rates that can go negative.
12. As a **quant developer**, I want to simulate G2++ (two-factor correlated Gaussian short rate), so that I can capture richer term structure dynamics.
13. As a **quant developer**, I want to simulate Vasicek (Ornstein-Uhlenbeck without shift), so that I can model simple mean-reverting rates.
14. As a **quant developer**, I want analytical zero-coupon bond pricing from G1++, G2++, and CIR++ states, so that I can derive yield curves at every timestep without numerical integration.
15. As a **quant developer**, I want the φ(t) shift function pre-computed and memoised at all time grid points, so that the scan loop indexes into an array rather than recomputing.

### Inflation Models

16. As a **quant developer**, I want to simulate real rates using the same model types (CIR++, G++) as nominal rates, so that I have a consistent modelling framework.
17. As a **quant developer**, I want inflation derived from the nominal/real rate differential via the exchange rate analogy, so that inflation is arbitrage-free relative to nominal and real rates.
18. As a **quant developer**, I want Fourier seasonality (2 harmonics, 4 coefficients) applied to inflation paths, so that I can model seasonal CPI/RPI patterns.
19. As a **quant developer**, I want both CPI and RPI index outputs from the inflation model chain, so that I can price index-linked instruments.

### Credit Models

20. As a **quant developer**, I want to simulate credit default intensities using CIR++ processes, so that I can model time-varying credit risk.
21. As a **quant developer**, I want Cox survival probabilities derived from integrated default intensity, so that I can price defaultable bonds.
22. As a **quant developer**, I want credit spreads computed from survival probabilities and recovery rates, so that I can output market-observable spread curves.
23. As a **quant developer**, I want multiple credit quality tiers (AAA through CCC) simulated with correlated default intensities, so that I can model diversified credit portfolios.

### Equity and Property Models

24. As a **quant developer**, I want to simulate equity total return indices using GBM (Geometric Brownian Motion), so that I can model equity accumulation paths.
25. As a **quant developer**, I want dividend yield and earnings yield outputs, so that I can decompose total returns.
26. As a **quant developer**, I want property returns modelled as an asset class, so that I can include real estate in multi-asset projections.

### Exchange Rate Models

27. As a **quant developer**, I want exchange rates simulated using the Foreign Currency Analogy (FCA) framework, so that I get arbitrage-free FX paths consistent with domestic and foreign interest rates.
28. As a **quant developer**, I want quanto adjustments applied to foreign-currency assets, so that domestic-currency returns are correctly risk-adjusted.
29. As a **quant developer**, I want multi-currency chains (e.g., GBP→USD→EUR derived from GBP→USD and USD→EUR), so that I can handle cross rates without redundant models.

### Correlation and Copula

30. As a **quant developer**, I want Cholesky-decomposed correlation applied to all model shocks, so that I get correctly correlated paths across all models.
31. As a **quant developer**, I want Gaussian copula support for specified model subsets, so that I can model tail dependence differently from linear correlation.
32. As a **quant developer**, I want Student-t copula with configurable degrees of freedom, so that I can model heavier tail dependence than Gaussian copula.
33. As a **quant developer**, I want separate correlation matrices for copula-participating and non-copula models, so that there is no double-correlation.
34. As a **quant developer**, I want Higham's alternating projections algorithm for nearest-correlation-matrix, so that user-specified matrices are always valid positive semi-definite.

### Configuration and Validation

35. As a **quant developer**, I want to define simulation configurations using Pydantic models with full validation, so that invalid parameters are caught at config time, not during simulation.
36. As a **quant developer**, I want to serialise/deserialise configurations to/from JSON and YAML, so that I can version-control and share simulation setups.
37. As a **quant developer**, I want Feller condition checking on CIR parameters with configurable strictness (warn vs error), so that I can catch numerically unstable configurations early.
38. As a **quant developer**, I want μ=0 enforcement on G1++/G2++ parameters at config validation time, so that the shift-function formulation is always correct.
39. As a **quant developer**, I want time grid configuration supporting monthly, annual, and custom timestep schedules, so that I can match different projection requirements.

### Output and Post-Processing

40. As a **quant developer**, I want to specify which model outputs to capture (e.g., short rate, ZCB prices, yield curves at specific tenors), so that I control memory usage.
41. As a **quant developer**, I want output post-processors (e.g., yield curve extraction, total return computation), so that derived quantities are computed efficiently inside the scan loop.
42. As a **quant developer**, I want results returned as structured JAX arrays with labelled dimensions (trial, timestep, output), so that I can slice and analyse results easily.
43. As a **quant developer**, I want to export results to Parquet, CSV, and HDF5 formats, so that downstream tools can consume simulation output.

### Yield Curve Operations

44. As a **quant developer**, I want ParametricCurve algebra (add, subtract, multiply, scalar operations) preserved from C#, so that I can compose yield curve transformations.
45. As a **quant developer**, I want yield curve transforms (zero-to-forward, forward-to-zero, continuous-to-discrete compounding), so that I can work in any rate convention.
46. As a **quant developer**, I want interpolation methods (Akima, cubic spline, Nelson-Siegel, flat extrapolation), so that I can construct smooth curves from market data.
47. As a **quant developer**, I want bond pricing (ZCB, coupon, index-linked) from yield curves, so that I can value fixed-income instruments.

### Calibration

48. As a **quant developer**, I want to calibrate CIR/OU parameters to market yield curves using Levenberg-Marquardt optimisation, so that models fit observed term structures.
49. As a **quant developer**, I want calibration to support multiple regimes with shared and regime-specific parameters, so that I can fit regime-switching models.
50. As a **quant developer**, I want calibration results (parameters, residuals, convergence diagnostics) returned as structured objects, so that I can assess fit quality.

### Market Data

51. As a **quant developer**, I want a pluggable market data interface (Protocol-based), so that I can source data from any provider without changing simulation code.
52. As a **quant developer**, I want a reference implementation for file-based market data (JSON/CSV), so that I can run simulations without a live data service.
53. As a **quant developer**, I want market data cached and validated before simulation starts, so that missing data causes a clear error, not a mid-simulation failure.

### Orchestration

54. As a **quant developer**, I want batch execution of multiple simulation configurations (e.g., different calibration dates or parameter sets), so that I can run production workloads.
55. As a **quant developer**, I want pluggable infrastructure protocols for storage and messaging, so that orchestration is not locked to any cloud provider.
56. As a **quant developer**, I want progress reporting during long simulations, so that I can monitor batch execution.

### Portfolio and LSMC (Phase 3)

57. As a **quant developer**, I want portfolio models that aggregate asset class returns with configurable weights, so that I can project multi-asset portfolio outcomes.
58. As a **quant developer**, I want LSMC (Least-Squares Monte Carlo) for path-dependent option pricing within the simulation, so that I can value guarantees and options embedded in insurance products.

### Testing and Parity

59. As a **quant developer**, I want golden-master parity tests comparing Python output to C# output at configurable tolerances, so that I can verify the port is mathematically correct.
60. As a **quant developer**, I want per-model unit tests with analytical solutions (where available), so that I can verify each model independently.
61. As a **quant developer**, I want statistical distribution tests (KS test, moment matching) for stochastic models, so that I can verify distributional properties even when exact trial-by-trial parity is impossible (due to different RNG).
62. As a **quant developer**, I want a `match_csharp` flag that switches intentional improvements back to C# behaviour, so that I can isolate port correctness from deliberate enhancements.

### Developer Experience

63. As a **quant developer**, I want a debug mode that disables JIT, enables NaN checking, and returns all intermediate states, so that I can step through simulations.
64. As a **quant developer**, I want single-trial execution without vmap for breakpoint debugging, so that I can inspect model state at any timestep.
65. As a **quant developer**, I want clear, structured error messages with 3-phase error handling (config/JIT/runtime), so that I can quickly identify and fix problems.
66. As a **quant developer**, I want a model registry with decorator-based registration, so that I can add new models without modifying engine code (Open/Closed Principle).
67. As a **quant developer**, I want comprehensive docstrings on all public APIs with SDE formulas in LaTeX notation, so that mathematical intent is clear.

---

## Implementation Decisions

### Architecture

1. **Single package, 9 modules**: `hyesg.core`, `hyesg.math`, `hyesg.models`, `hyesg.engine`, `hyesg.config`, `hyesg.calibration`, `hyesg.market_data`, `hyesg.orchestration`, `hyesg.io`. Each module has a single clear responsibility.

2. **Protocol-based contracts** (not ABC): All inter-module contracts use `typing.Protocol` for structural typing. This avoids inheritance hierarchies and works naturally with JAX's functional paradigm. Key protocols: `Model`, `InterestRateModel`, `CurrencyAnalogy`, `ExchangeRateModel`, `CreditModel`, `StochasticProcess`, `PostProcessor`, `YieldCurveProtocol`.

3. **NamedTuples for model state**: All model states (e.g., `OUState(x, short_rate)`, `G2State(x1, x2, short_rate)`, `CIRState(x, short_rate)`) are `NamedTuple` subclasses — immutable, typed, and automatically JAX pytree-compatible.

4. **Push-based `jax.lax.scan` replaces pull-based `LazySequence`**: This is the most important architectural change. C# uses lazy evaluation with mutable forward-fill cache; Python uses a single forward scan with immutable pytree carry containing all model states. Models are evaluated in topological dependency order within each timestep.

5. **`functools.partial` for static configuration**: Model parameters, time grid, and other static data are closed over via `functools.partial` on the scan step function — NOT included in the scan carry. This keeps the carry small (states only) and avoids retracing.

6. **Separate scan runs per regime**: Rather than a single scan with regime-indexed parameters (complex), each regime runs its own `jax.lax.scan`. Final results are concatenated.

7. **Model registry**: Decorator-based (`@register_model("cir_pp")`) explicit registration replaces C#'s MEF discovery. The registry maps string names to model classes, resolved at config validation time.

8. **Pydantic v2 for all configuration**: Replaces C#'s fluent builder + JSON templates. Provides validation, serialisation, and clear error messages. Config validation is fail-fast — all errors caught before JIT compilation.

### Engine Pipeline

9. **Split correlation pipeline**: Non-copula models get standard Cholesky correlation. Copula-participating models get a SEPARATE copula-only Cholesky, followed by CDF transform to uniforms, then marginal inverse-CDF. The two pipelines never mix — preventing double-correlation.

10. **Antithetic variance reduction**: Non-copula streams negate normals pre-correlation (`-z`). Copula streams complement post-CDF uniforms (`1-u`). These are mutually exclusive transforms — applying both cancels out (Φ(-z) = 1-Φ(z)).

11. **Shock convention**: Models receive raw N(0,1) shocks and multiply by √dt themselves. This differs from C# (which pre-scales by √dt) but simplifies the interface and makes the discretisation explicit in each model.

12. **Dict-based carry with pre-allocated keys**: The scan carry is `SimulationState = dict[str, ModelState]`. Keys are pre-allocated from the initial state dict; each step creates a shallow copy then updates values. This guarantees pytree structure stability across scan iterations.

13. **Phi memoisation**: For CIR++ and G++, the φ(t) shift function is pre-evaluated at all time grid points and stored as a JAX array. The scan loop indexes into this array by step number — no recomputation.

### Financial Mathematics

14. **CIR++ phi**: Numerical computation `φ(t) = f_market(0,t) - f_CIR(0,t; x₀)` with non-negativity constraint (clamp if between -1e-4 and 0, error if more negative). Uses CIR A/B functions evaluated from time 0.

15. **CIR++ phi integral**: Uses `A(T)/A(t)` and `B(T)-B(t)` where A/B are evaluated at maturities T and t FROM TIME 0. NOT `A(T-t)` and `B(T-t)` — CIR A/B functions are nonlinear, so these are not equivalent.

16. **G1++ phi**: Analytical: `φ(t) = f_market(0,t) + (σ²/2α²)(1-e^{-αt})²`.

17. **G++ ZCB pricing**: `ln P(t,T) = -IntegralPhi(t,T) - M(t,T) + ½V²(t,T)` with explicit formulas for M (expectation of integral), V² (variance of integral), and IntegralPhi (market curve contribution).

18. **Quanto adjustment**: `drift_adjustment -= ρ × σ_fx × √dt` for raw N(0,1) shocks. No σ_model term — the correlation with FX is the only adjustment.

19. **Inflation seasonality**: Fourier with 2 harmonics (4 coefficients), subtracted (not added), with 0.5 constant shift: `shift = (month-1 + day/daysInMonth)/12 + 0.5`.

20. **μ=0 enforcement**: G1++ and G2++ models enforce μ=0 at config validation time via Pydantic `@model_validator`. This is required for the analytical phi formula to be correct.

### Intentional Deviations from C#

21. **Copula antithetic (D1)**: C# applies `1-z` to raw normals pre-correlation (producing N(1,1) — a shifted distribution, not a proper antithetic). hyesg applies `1-u` to post-CDF uniforms, which is the mathematically correct complement. Parity tests for copula-antithetic trials will differ.

22. **BOverDt Taylor (D2)**: C# uses linear Taylor `1-y/2`; hyesg adds quadratic `y²/6` for improved accuracy. Tiny differences in G++ ZCB prices near y≈0. A `match_csharp` flag can revert to C# behaviour for parity testing.

### Numerical Stability

23. **CIR floor at zero**: After each Euler step, `x = max(x, 0)`. Feller condition checked at config time.
24. **Correlation matrix projection**: Higham's alternating projections (2002), 100 max iterations, 1e-10 tolerance.
25. **Bond yield solver**: `jax.lax.fori_loop` bounded Newton-Raphson (max 50 iterations).
26. **BOverDt**: Taylor expansion `1 - y/2 + y²/6` for `|y| < 1e-8` to avoid 0/0.

---

## Testing Decisions

### What Makes a Good Test

Tests should verify **external behaviour** — given specific inputs (parameters, shocks, market data), does the model produce the correct outputs? Tests should NOT test internal implementation details (e.g., how the scan carry is structured, which JAX primitives are used).

### Testing Tiers

1. **Analytical parity tests** (where closed-form solutions exist):
   - CIR ZCB pricing against analytical A/B formulas
   - G1++ phi against analytical formula
   - Black option pricing against known formulas
   - Bond pricing against standard formulations
   - **Tolerance**: rtol=1e-10 (machine precision)

2. **C# golden-master parity tests** (numerical agreement with C#):
   - Generate golden masters from C# with known seeds, parameters, and shocks
   - Compare Python output at each timestep
   - **Tolerance tiers**: Exact (1e-12) for deterministic math, Tight (1e-8) for accumulated stepping, Statistical (KS p>0.05) for RNG-dependent paths
   - Use `match_csharp=True` flag to disable intentional deviations

3. **Statistical distribution tests** (for stochastic correctness):
   - KS tests on marginal distributions
   - Moment matching (mean, variance, skewness, kurtosis)
   - Correlation matrix recovery from simulated paths
   - Antithetic variance reduction effectiveness (variance ratio < 1)

4. **Property-based tests** (invariants):
   - ZCB prices in (0, 1] for positive rates
   - Yield curves smooth and monotone where expected
   - CIR paths non-negative
   - Correlation matrix positive semi-definite after projection

5. **Integration tests**:
   - Full pipeline: config → validate → simulate → output
   - Multi-regime simulation
   - Copula + non-copula mixed pipeline
   - Debug mode produces complete intermediate states

### Modules Under Test

| Module | Test Type | Priority |
|--------|-----------|----------|
| `hyesg.math.curves` | Analytical parity | P0 |
| `hyesg.math.pricing` | Analytical parity | P0 |
| `hyesg.math.transforms` | Analytical parity | P0 |
| `hyesg.models.short_rates` (all 6) | Golden master + analytical | P0 |
| `hyesg.models.inflation` | Golden master + statistical | P0 |
| `hyesg.models.credit` | Golden master + statistical | P0 |
| `hyesg.models.exchange_rates` | Golden master | P0 |
| `hyesg.engine.simulator` | Integration | P0 |
| `hyesg.engine.correlation` | Analytical (eigenvalue check) | P0 |
| `hyesg.engine.copula` | Statistical (marginal uniformity) | P0 |
| `hyesg.engine.rng` | Statistical (uniformity, independence) | P1 |
| `hyesg.config` | Validation (reject bad configs) | P1 |
| `hyesg.calibration` | Golden master | P1 |
| `hyesg.io` | Round-trip (write → read → compare) | P2 |
| `hyesg.orchestration` | Integration | P2 |

---

## Out of Scope

1. **Azure-specific infrastructure**: No Azure Functions, ServiceBus, Entity Framework, or Blob Storage implementations. The package provides Protocol-based interfaces; cloud adapters are separate packages.
2. **Web API / REST endpoints**: No HTTP server. The ESS orchestration layer's ASP.NET MVC is not ported — `hyesg` is a library, not a service.
3. **Excel integration**: No ParameterReader for Excel. Market data and calibration parameters come via JSON/CSV/Parquet or the pluggable market data protocol.
4. **GUI / visualisation**: No dashboards or plotting. Users consume JAX arrays with their preferred tools (matplotlib, plotly, etc.).
5. **MersenneTwister RNG**: JAX uses ThreeFry PRNG. Exact trial-by-trial parity with C# is not achievable and not attempted — only statistical distributional parity.
6. **MEF plugin system**: Replaced by explicit decorator-based model registry.
7. **Legacy .NET Framework 4.7.2 patterns**: Entity Framework, WCF, ASMX — none ported.
8. **Real-time streaming**: The engine runs batch simulations. No streaming/live-update mode.

---

## Further Notes

### Implementation Phases

The implementation should proceed in four phases:

- **Phase 0 (Foundation)**: `core`, `math`, `config` modules — protocols, state types, curves, transforms, pricing, configuration validation. No simulation engine yet, but all building blocks tested independently.
- **Phase 1 (Engine + Models)**: `engine` (scan loop, correlation, copula, RNG, antithetic), all `models` (short rates, inflation, credit, equity, FX). End-to-end simulation running with parity tests.
- **Phase 2 (Production)**: `calibration`, `market_data`, `io`, `orchestration`. Production-ready with batch execution, data I/O, and calibration.
- **Phase 3 (Advanced)**: Portfolio models, LSMC, neural network components (equinox-based).

### Key Risk Areas

1. **JAX JIT with dict carry**: The scan carry uses `dict[str, ModelState]` — keys must be fixed across all iterations. Pre-allocation from initial state enforces this, but any model that tries to add/remove keys will break JIT. Extensive testing needed.
2. **GPU memory at full scale**: 5,000 trials × 1,200 steps × 50 models × float64 ≈ 120GB. Chunked vmap (process trials in batches) is essential for GPU deployment.
3. **JIT compilation time**: First run with 50+ models in the unrolled step loop may take 5-10 minutes to compile. Subsequent runs use cached compilation.
4. **CIR2++ blending**: The 2×2 linear system for blending risk-neutral and real-world CIR parameters is numerically subtle. Extensive parity testing required.
5. **RNG sequence differences**: JAX ThreeFry vs C# MersenneTwister produce different sequences. Parity tests must use injected shocks or statistical comparison, never exact trial-by-trial matching.

### Specification Reference

The complete technical specification (120K chars, 8 sections, verified through 6 adversarial review cycles) contains:
- All SDE discretisation formulas with derivations
- Explicit phi function formulas for every model
- G++ ZCB pricing with full variance formulas (verified against C# source)
- JAX scan architecture with pseudocode
- Correlation and copula pipeline details
- Antithetic variance reduction implementation
- Complete tolerance table from C# source
- Error handling philosophy (3-phase)
- Debugging and JIT compilation guidance
- Intentional deviations from C# with rationale

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| jax | ≥0.4.30 | Core numerical computation, scan, vmap, JIT |
| jaxlib | matches JAX | XLA backend (CPU/GPU/TPU) |
| jaxtyping | ≥0.2.28 | Array shape/dtype annotations |
| pydantic | ≥2.5 | Configuration validation and serialisation |
| equinox | ≥0.11 | Optional: neural network components (Phase 3) |
| pytest | ≥7.0 | Testing framework |
| hypothesis | ≥6.0 | Property-based testing |
