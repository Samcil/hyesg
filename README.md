# hyesg

**Python/JAX Economic Scenario Generator**

A high-performance Economic Scenario Generator (ESG) built on JAX for hardware-transparent Monte Carlo simulation across CPU, GPU, and TPU.

## Overview

`hyesg` reimplements Hymans Robertson's C# ESG as a clean Python package using JAX for numerical computation. It simulates correlated financial and economic variables (interest rates, inflation, credit, equity, FX) over long time horizons for actuarial and investment modelling.

## Key Features

- **Write once, run anywhere** — identical code on CPU, GPU, and TPU via JAX/XLA
- **50+ financial models** — CIR, CIR++, CIR2++, G1++, G2++, Vasicek, GBM, FCA exchange rates, credit, inflation
- **High-performance Monte Carlo** — 5,000 trials × 1,200 timesteps via `jax.lax.scan` and `vmap`
- **Protocol-based architecture** — SOLID principles, clean module boundaries
- **Pydantic v2 configuration** — validated, serialisable, fail-fast
- **Gaussian and Student-t copulas** — for tail dependence modelling

## Documentation

- [Product Requirements Document](docs/hyesg-prd.md)

## License

Proprietary — Hymans Robertson LLP
