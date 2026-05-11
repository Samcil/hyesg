"""Core ESG simulation engine.

Uses ``jax.lax.scan`` for efficient timestep evolution and ``jax.vmap``
for batching across Monte Carlo trials.  Static config (params, Cholesky L)
is bound via ``functools.partial``, NOT included in the scan carry dict.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from hyesg.config.models import (
    CorrelationEntry,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.core.registry import get_model
from hyesg.core.types import OutputSpec
from hyesg.engine.correlation import cholesky_factor, correlate_shocks
from hyesg.engine.output import SimulationResult, combine_regime_results, extract_outputs
from hyesg.engine.rng import create_rng_keys


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def topological_sort(models: dict[str, Any]) -> list[str]:
    """Order models by dependencies using Kahn's algorithm.

    Args:
        models: Dict mapping model name to an object with a
            ``dependencies`` attribute (list of model names).

    Returns:
        List of model names in dependency order (dependencies first).

    Raises:
        ValueError: If cyclic dependencies detected.
    """
    # Build adjacency list and in-degree count
    in_degree: dict[str, int] = {name: 0 for name in models}
    dependents: dict[str, list[str]] = defaultdict(list)

    for name, cfg in models.items():
        deps = getattr(cfg, "dependencies", [])
        for dep in deps:
            if dep not in models:
                raise ValueError(
                    f"Model '{name}' depends on '{dep}' which is not defined"
                )
            dependents[dep].append(name)
            in_degree[name] += 1

    # Kahn's algorithm
    queue = sorted(n for n, d in in_degree.items() if d == 0)
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in sorted(dependents[node]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(models):
        remaining = set(models.keys()) - set(result)
        raise ValueError(f"Cyclic dependencies detected among: {remaining}")

    return result


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------


def build_time_grid(config: TimeGridConfig) -> tuple[Array, Array]:
    """Build time points and dt arrays from config.

    Args:
        config: Time grid configuration.

    Returns:
        Tuple of (time_points, dts) where time_points has shape
        (n_steps+1,) and dts has shape (n_steps,).
    """
    time_points = config.time_points
    dts = jnp.diff(time_points)
    return time_points, dts


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------


class Simulator:
    """Core ESG simulation engine.

    Uses ``jax.lax.scan`` for efficient timestep evolution.
    Static config (params, Cholesky L, curves) is bound via
    ``functools.partial``, NOT included in the scan carry dict.

    Args:
        config: Complete simulation configuration.
        models: Dict of instantiated model objects (name -> model).
            If None, models are created from config using the registry.
    """

    def __init__(
        self,
        config: SimulationConfig,
        models: dict[str, Any] | None = None,
    ) -> None:
        self._config = config

        # Build model config lookup by name
        self._model_configs: dict[str, Any] = {m.name: m for m in config.models}

        # Instantiate or use provided models
        if models is not None:
            self._models = models
        else:
            self._models = self._build_models()

        # Topological sort
        self._model_order = topological_sort(self._model_configs)

        # Build shock layout: map model name -> (start_idx, end_idx) in shock vector
        self._shock_slices: dict[str, tuple[int, int]] = {}
        self._shock_names: list[str] = []
        offset = 0
        for name in self._model_order:
            model = self._models[name]
            n = model.n_shocks
            self._shock_slices[name] = (offset, offset + n)
            # Collect shock names for correlation mapping
            shock_cfg = model.shock_config
            self._shock_names.extend(shock_cfg.names)
            offset += n
        self._total_shocks = offset

        # Build correlation matrix
        self._corr_matrix, self._cholesky_L = self._build_correlation()

        # Time grid
        self._time_points, self._dts = build_time_grid(config.time_grid)
        self._n_steps = len(self._dts)

    def _build_models(self) -> dict[str, Any]:
        """Instantiate models from config using the registry.

        Returns:
            Dict mapping model name to instantiated model object.
        """
        models: dict[str, Any] = {}
        for model_cfg in self._config.models:
            model_cls = get_model(model_cfg.type)
            model = model_cls(params=model_cfg.params, name=model_cfg.name)
            models[model_cfg.name] = model
        return models

    def _build_correlation(self) -> tuple[Array, Array]:
        """Build correlation matrix and Cholesky factor.

        Returns:
            Tuple of (correlation_matrix, cholesky_L).
        """
        n = self._total_shocks
        if n == 0:
            return jnp.eye(1, dtype=jnp.float64), jnp.eye(1, dtype=jnp.float64)

        # Build name -> index mapping
        name_to_idx: dict[str, int] = {}
        for i, sname in enumerate(self._shock_names):
            name_to_idx[sname] = i

        # Start with identity
        corr = jnp.eye(n, dtype=jnp.float64)

        # Fill from correlation entries
        for entry in self._config.correlations:
            if entry.shock_a in name_to_idx and entry.shock_b in name_to_idx:
                i = name_to_idx[entry.shock_a]
                j = name_to_idx[entry.shock_b]
                corr = corr.at[i, j].set(entry.value)
                corr = corr.at[j, i].set(entry.value)

        # Compute Cholesky
        chol_L = cholesky_factor(corr)
        return corr, chol_L

    def _make_step_fn(
        self,
        models: dict[str, Any],
        model_order: list[str],
        shock_slices: dict[str, tuple[int, int]],
        model_deps: dict[str, list[str]],
        cholesky_L: Array,
        total_shocks: int,
    ) -> Callable:
        """Build the scan step function with static data bound via partial.

        Args:
            models: Dict of instantiated model objects.
            model_order: Topologically sorted model names.
            shock_slices: Maps model name to (start, end) in shock vector.
            model_deps: Maps model name to list of dependency names.
            cholesky_L: Lower-triangular Cholesky factor.
            total_shocks: Total number of shock streams.

        Returns:
            Step function: (carry, t_dt) -> (carry, outputs).
        """

        def step_fn(carry: dict, t_dt: tuple[Array, Array]) -> tuple[dict, dict]:
            """Single timestep: evolve all models in dependency order."""
            t, dt = t_dt
            key = carry["rng_key"]
            key, step_key = jax.random.split(key)

            # Generate and correlate shocks for this step
            raw = jax.random.normal(step_key, shape=(total_shocks,))
            correlated = correlate_shocks(
                raw.reshape(1, -1), cholesky_L
            ).reshape(-1)

            # Step each model in topological order
            new_states = {}
            step_outputs: dict[str, dict[str, Array]] = {}

            for name in model_order:
                model = models[name]
                start, end = shock_slices[name]
                model_shocks = jax.lax.dynamic_slice(
                    correlated, (start,), (end - start,)
                )

                # Gather dependency outputs
                deps: dict[str, Any] = {}
                for dep_name in model_deps.get(name, []):
                    # Merge the dep's outputs so the model can access them
                    dep_outputs = step_outputs.get(dep_name, {})
                    deps.update(dep_outputs)

                state = carry["states"][name]
                new_state, outputs = model.step(state, t, dt, model_shocks, deps)
                new_states[name] = new_state
                step_outputs[name] = outputs

            new_carry = {"states": new_states, "rng_key": key}
            return new_carry, step_outputs

        return step_fn

    def run(
        self,
        seed: int | None = None,
        regime_idx: int = 0,
    ) -> SimulationResult:
        """Run simulation for a single regime.

        Steps:
        1. Generate trial keys from seed.
        2. For each trial (via vmap):
           a. Init all model states.
           b. Build scan carry dict.
           c. Run jax.lax.scan over timesteps.
           d. Collect outputs.
        3. Stack results across trials.
        4. Wrap in SimulationResult.

        Args:
            seed: RNG seed. If None, uses regime config seed.
            regime_idx: Index of the regime to use for n_trials/seed.

        Returns:
            SimulationResult with outputs shaped (n_trials, n_steps).
        """
        t_start = time.monotonic()

        # Get regime config
        if self._config.regimes:
            regime = self._config.regimes[regime_idx]
            n_trials = regime.n_trials
            if seed is None:
                seed = regime.seed
        else:
            n_trials = 100
            if seed is None:
                seed = 42

        # Build dependency map
        model_deps: dict[str, list[str]] = {}
        for name, cfg in self._model_configs.items():
            model_deps[name] = list(cfg.dependencies)

        # Build step function
        step_fn = self._make_step_fn(
            models=self._models,
            model_order=self._model_order,
            shock_slices=self._shock_slices,
            model_deps=model_deps,
            cholesky_L=self._cholesky_L,
            total_shocks=self._total_shocks,
        )

        # Initialize all model states
        init_states: dict[str, Any] = {}
        for name in self._model_order:
            model = self._models[name]
            init_states[name] = model.init_state()

        # Generate trial keys
        keys = create_rng_keys(seed, n_trials, 1)[0]  # (n_trials, 2)

        # Time arrays for scan input
        ts = self._time_points[:-1]  # (n_steps,)
        dts = self._dts  # (n_steps,)

        # Single trial function
        def run_single_trial(trial_key: Array) -> dict[str, dict[str, Array]]:
            carry = {"states": init_states, "rng_key": trial_key}
            _final_carry, all_outputs = jax.lax.scan(step_fn, carry, (ts, dts))
            return all_outputs

        # vmap over trials
        all_trial_outputs = jax.vmap(run_single_trial)(keys)

        # all_trial_outputs: {model_name: {field: Array(n_trials, n_steps)}}
        # Extract outputs
        output_specs: list[OutputSpec] | None = None
        extracted = extract_outputs(all_trial_outputs, output_specs)

        elapsed = time.monotonic() - t_start
        metadata = {
            "seed": seed,
            "n_trials": n_trials,
            "n_steps": self._n_steps,
            "regime_idx": regime_idx,
            "elapsed_seconds": elapsed,
            "model_order": self._model_order,
        }

        return SimulationResult(
            outputs=extracted,
            time_grid=self._time_points,
            metadata=metadata,
        )

    def run_all_regimes(self) -> SimulationResult:
        """Run all regimes and combine results.

        For each regime: run(seed=regime.seed, regime_idx=i)
        Then combine/concatenate results across regimes.

        Returns:
            Combined SimulationResult with all regimes.
        """
        if not self._config.regimes:
            return self.run()

        results = []
        for i, regime in enumerate(self._config.regimes):
            result = self.run(seed=regime.seed, regime_idx=i)
            results.append(result)

        return combine_regime_results(results)

    @property
    def model_order(self) -> list[str]:
        """Topologically sorted model names."""
        return list(self._model_order)

    @property
    def total_shocks(self) -> int:
        """Total number of shock streams across all models."""
        return self._total_shocks

    @property
    def shock_slices(self) -> dict[str, tuple[int, int]]:
        """Map of model name to (start, end) shock indices."""
        return dict(self._shock_slices)

    @property
    def correlation_matrix(self) -> Array:
        """Correlation matrix used for shock generation."""
        return self._corr_matrix

    @property
    def cholesky_L(self) -> Array:
        """Cholesky factor of the correlation matrix."""
        return self._cholesky_L

    @property
    def n_steps(self) -> int:
        """Number of simulation timesteps."""
        return self._n_steps

    @property
    def config(self) -> SimulationConfig:
        """The simulation configuration."""
        return self._config
