"""Topological ordering of models within and across economies.

The ``ModelDependencyGraph`` resolves execution order for all models
in a multi-economy simulation, ensuring that each model runs only
after its dependencies have completed.

.. todo::
    **F40 Integration Path** — ``ModelDependencyGraph`` duplicates
    the topological sort in :func:`~hyesg.engine.simulator.topological_sort`.
    Future integration:

    1. Consolidate into a single ``topological_sort`` utility that both
       the ``Simulator`` and ``ModelDependencyGraph`` call.
    2. ``ModelDependencyGraph`` should accept ``Economy`` objects and
       produce the ``ModelConfig.dependencies`` lists that
       ``SimulationConfig`` expects, bridging the economy-oriented
       and flat config systems.
    3. Cross-economy dependencies (e.g. FX referencing a domestic
       nominal rate) should be resolved here, not in
       ``to_simulation_config()``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

from hyesg.config.economy import Economy


class ModelDependencyGraph:
    """Topological ordering of models within and across economies.

    Builds a directed acyclic graph from the structural rules:

    - **Nominal rate**: no dependencies.
    - **FX**: depends on domestic nominal + own nominal.
    - **Real rate**: depends on own nominal.
    - **Inflation**: depends on own real rate.
    - **Equity**: depends on own nominal.
    - **Credit**: depends on own nominal.
    - **Salary**: depends on own real rate.

    Attributes:
        economies: The economies used to build the graph.
    """

    def __init__(self, economies: Sequence[Economy]) -> None:
        """Initialize and build the dependency graph.

        Args:
            economies: Sequence of economy specifications. Exactly one
                must have ``is_domestic=True``.

        Raises:
            ValueError: If no domestic economy is found.
        """
        self._economies = list(economies)
        self._graph: dict[str, list[str]] = defaultdict(list)
        self._in_degree: dict[str, int] = {}
        self._all_labels: list[str] = []
        self._build_graph()

    @property
    def economies(self) -> list[Economy]:
        """The economies used to build the graph."""
        return list(self._economies)

    @property
    def graph(self) -> dict[str, list[str]]:
        """Adjacency list: label → list of labels that depend on it."""
        return dict(self._graph)

    def _find_domestic(self) -> Economy:
        """Find the single domestic economy.

        Returns:
            The domestic economy.

        Raises:
            ValueError: If no domestic economy exists.
        """
        for econ in self._economies:
            if econ.is_domestic:
                return econ
        raise ValueError("No domestic economy found (is_domestic=True)")

    def _build_graph(self) -> None:
        """Build dependency graph from economy specifications.

        Rules:
        - Nominal rate: no deps.
        - FX: depends on domestic nominal + own nominal.
        - Real rate: depends on own nominal.
        - Inflation: depends on real rate.
        - Equity: depends on own nominal.
        - Credit: depends on own nominal.
        - Salary: depends on real rate.
        """
        domestic = self._find_domestic()
        domestic_nominal = domestic.nominal_rate_model.label

        # Collect all labels first
        for econ in self._economies:
            for model in econ.all_models:
                self._all_labels.append(model.label)
                self._in_degree[model.label] = 0

        # Build edges
        for econ in self._economies:
            own_nominal = econ.nominal_rate_model.label

            # FX depends on domestic nominal + own nominal
            if econ.fx_model:
                self._add_edge(domestic_nominal, econ.fx_model.label)
                if own_nominal != domestic_nominal:
                    self._add_edge(own_nominal, econ.fx_model.label)

            # Real rate depends on own nominal
            if econ.real_rate_model:
                self._add_edge(own_nominal, econ.real_rate_model.label)

            # Inflation depends on real rate
            if econ.inflation_model and econ.real_rate_model:
                self._add_edge(
                    econ.real_rate_model.label, econ.inflation_model.label
                )

            # Equities depend on own nominal
            for eq in econ.equity_models:
                self._add_edge(own_nominal, eq.label)

            # Credit depends on own nominal
            if econ.credit_pool:
                self._add_edge(own_nominal, econ.credit_pool.label)

            # Salary depends on real rate
            if econ.salary_model and econ.real_rate_model:
                self._add_edge(
                    econ.real_rate_model.label, econ.salary_model.label
                )

    def _add_edge(self, from_label: str, to_label: str) -> None:
        """Add a directed edge: from_label must execute before to_label."""
        self._graph[from_label].append(to_label)
        self._in_degree[to_label] = self._in_degree.get(to_label, 0) + 1

    def topological_order(self) -> list[str]:
        """Return model labels in valid execution order.

        Uses Kahn's algorithm (BFS-based topological sort).

        Returns:
            List of model labels in dependency-safe order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        in_deg = dict(self._in_degree)
        queue: deque[str] = deque()

        for label in self._all_labels:
            if in_deg.get(label, 0) == 0:
                queue.append(label)

        result: list[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbour in self._graph.get(node, []):
                in_deg[neighbour] -= 1
                if in_deg[neighbour] == 0:
                    queue.append(neighbour)

        if len(result) != len(self._all_labels):
            raise ValueError(
                "Dependency graph contains a cycle — cannot determine "
                "execution order"
            )
        return result

    def execution_layers(self) -> list[list[str]]:
        """Group models into parallel execution layers.

        Models within the same layer have no mutual dependencies and
        can be executed concurrently.

        Returns:
            List of layers, each a list of model labels.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        in_deg = dict(self._in_degree)
        queue: deque[str] = deque()

        for label in self._all_labels:
            if in_deg.get(label, 0) == 0:
                queue.append(label)

        layers: list[list[str]] = []
        processed = 0

        while queue:
            layer: list[str] = list(queue)
            queue.clear()
            for node in layer:
                processed += 1
                for neighbour in self._graph.get(node, []):
                    in_deg[neighbour] -= 1
                    if in_deg[neighbour] == 0:
                        queue.append(neighbour)
            layers.append(layer)

        if processed != len(self._all_labels):
            raise ValueError(
                "Dependency graph contains a cycle — cannot determine "
                "execution layers"
            )
        return layers
