"""Configuration validation for hyesg.

Validates a SimulationConfig for semantic correctness beyond
what Pydantic field validators can check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from hyesg.core.registry import list_models

if TYPE_CHECKING:
    from hyesg.config.models import SimulationConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        msg = f"{len(errors)} validation error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


def build_dep_graph(
    config: SimulationConfig,
) -> dict[str, list[str]]:
    """Build adjacency list from model dependencies.

    Args:
        config: Simulation configuration.

    Returns:
        Dict mapping model name to list of dependencies.
    """
    graph: dict[str, list[str]] = {}
    for model in config.models:
        graph[model.name] = list(model.dependencies)
    return graph


def has_cycles(graph: dict[str, list[str]]) -> bool:
    """Check if dependency graph contains cycles.

    Args:
        graph: Adjacency list.

    Returns:
        True if a cycle exists.
    """
    return find_cycle_path(graph) is not None


def find_cycle_path(
    graph: dict[str, list[str]],
) -> list[str] | None:
    """Find a cycle path in the dependency graph.

    Uses DFS with colouring: WHITE=unvisited, GREY=in-progress,
    BLACK=complete.

    Args:
        graph: Adjacency list.

    Returns:
        List of nodes forming a cycle, or None.
    """
    white, grey, black = 0, 1, 2
    colour: dict[str, int] = {n: white for n in graph}
    parent: dict[str, str | None] = {n: None for n in graph}

    def _dfs(node: str) -> list[str] | None:
        colour[node] = grey
        for dep in graph.get(node, []):
            if dep not in colour:
                continue
            if colour[dep] == grey:
                # Reconstruct cycle
                cycle = [dep, node]
                current = node
                while parent[current] is not None and parent[current] != dep:
                    current = parent[current]  # type: ignore[assignment]
                    cycle.append(current)
                cycle.reverse()
                return cycle
            if colour[dep] == white:
                parent[dep] = node
                result = _dfs(dep)
                if result is not None:
                    return result
        colour[node] = black
        return None

    for node in graph:
        if colour[node] == white:
            result = _dfs(node)
            if result is not None:
                return result
    return None


def validate_config(
    config: SimulationConfig,
) -> list[str]:
    """Validate a SimulationConfig for semantic correctness.

    Checks beyond Pydantic validation:
    - Model types exist in registry
    - Dependencies exist
    - No circular dependencies
    - Correlation matrix is positive semi-definite
    - CIR stability (dt < 2/alpha)
    - Trial count warnings

    Args:
        config: The configuration to validate.

    Returns:
        List of warning/error messages (empty = valid).
    """
    errors: list[str] = []
    model_names = {m.name for m in config.models}
    registered = set(list_models())

    # Check model types exist in registry
    for model in config.models:
        if registered and model.type not in registered:
            errors.append(
                f"Model type '{model.type}' for '{model.name}' "
                f"not found in registry. "
                f"Available: {sorted(registered)}"
            )

    # Check dependencies exist
    for model in config.models:
        for dep in model.dependencies:
            if dep not in model_names:
                errors.append(
                    f"Model '{model.name}' depends on '{dep}' which is not defined"
                )

    # Check for circular dependencies
    dep_graph = build_dep_graph(config)
    cycle = find_cycle_path(dep_graph)
    if cycle is not None:
        errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")

    # Check correlation matrix is PSD
    if config.correlations:
        shock_names: list[str] = []
        for corr in config.correlations:
            if corr.shock_a not in shock_names:
                shock_names.append(corr.shock_a)
            if corr.shock_b not in shock_names:
                shock_names.append(corr.shock_b)

        n = len(shock_names)
        name_to_idx = {name: i for i, name in enumerate(shock_names)}
        matrix = jnp.eye(n)
        for corr in config.correlations:
            i = name_to_idx[corr.shock_a]
            j = name_to_idx[corr.shock_b]
            matrix = matrix.at[i, j].set(corr.value)
            matrix = matrix.at[j, i].set(corr.value)

        eigenvalues = jnp.linalg.eigvalsh(matrix)
        if jnp.any(eigenvalues < -1e-10):
            min_eig = float(jnp.min(eigenvalues))
            errors.append(
                f"Correlation matrix is not positive "
                f"semi-definite (min eigenvalue={min_eig:.6e})"
            )

    # CIR stability check
    dt = config.time_grid.dt
    for model in config.models:
        if model.type in ("cir", "cir2pp"):
            alpha = model.params.get("alpha", None)
            alpha1 = model.params.get("alpha1", None)
            alpha2 = model.params.get("alpha2", None)
            for a_val in [alpha, alpha1, alpha2]:
                if a_val is not None and a_val > 0 and dt >= 2.0 / a_val:
                    errors.append(
                        f"CIR stability: dt={dt:.4f} >= "
                        f"2/alpha={2.0 / a_val:.4f} for "
                        f"model '{model.name}'. "
                        f"Reduce timestep or alpha."
                    )

    # Trial count warnings
    total_trials = sum(r.n_trials for r in config.regimes)
    if total_trials == 0:
        errors.append("No trials configured across regimes")
    elif total_trials < 100:
        errors.append(
            f"Low trial count ({total_trials}). "
            f"Consider at least 1000 for stable results."
        )

    return errors
