"""Tests for ModelDependencyGraph."""

from __future__ import annotations

import pytest

from hyesg.config.economies.foreign import build_usd_economy
from hyesg.config.economies.gbp import build_gbp_economy
from hyesg.config.economy import Economy, EconomyModelConfig
from hyesg.orchestration.dependency_graph import ModelDependencyGraph


@pytest.fixture()
def gbp_economy() -> Economy:
    """GBP domestic economy."""
    return build_gbp_economy()


@pytest.fixture()
def usd_economy() -> Economy:
    """USD foreign economy."""
    return build_usd_economy()


@pytest.fixture()
def multi_economy_graph(
    gbp_economy: Economy, usd_economy: Economy
) -> ModelDependencyGraph:
    """Dependency graph for GBP + USD."""
    return ModelDependencyGraph([gbp_economy, usd_economy])


class TestDependencyGraphConstruction:
    """Tests for graph construction."""

    def test_requires_domestic_economy(self) -> None:
        """Raises ValueError if no domestic economy."""
        foreign = Economy(
            name="USD",
            is_domestic=False,
            nominal_rate_model=EconomyModelConfig(
                model_type="cir2pp", label="usd_nominal"
            ),
            fx_model=EconomyModelConfig(
                model_type="fx_gbm", label="usd_fx"
            ),
        )
        with pytest.raises(ValueError, match="No domestic economy"):
            ModelDependencyGraph([foreign])

    def test_single_domestic_economy(self, gbp_economy: Economy) -> None:
        """Graph builds successfully with single domestic economy."""
        graph = ModelDependencyGraph([gbp_economy])
        assert len(graph.topological_order()) == len(gbp_economy.all_models)


class TestTopologicalOrder:
    """Tests for topological ordering."""

    def test_nominal_before_fx(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """Nominal rates appear before FX in topological order."""
        order = multi_economy_graph.topological_order()
        gbp_nom_idx = order.index("gbp_nominal")
        usd_fx_idx = order.index("usd_fx")
        assert gbp_nom_idx < usd_fx_idx

    def test_own_nominal_before_fx(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """Each economy's nominal appears before its own FX."""
        order = multi_economy_graph.topological_order()
        usd_nom_idx = order.index("usd_nominal")
        usd_fx_idx = order.index("usd_fx")
        assert usd_nom_idx < usd_fx_idx

    def test_nominal_before_real(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """Nominal appears before real rate."""
        order = multi_economy_graph.topological_order()
        gbp_nom_idx = order.index("gbp_nominal")
        gbp_real_idx = order.index("gbp_real")
        assert gbp_nom_idx < gbp_real_idx

    def test_real_before_inflation(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """Real rate appears before inflation."""
        order = multi_economy_graph.topological_order()
        gbp_real_idx = order.index("gbp_real")
        gbp_infl_idx = order.index("gbp_inflation")
        assert gbp_real_idx < gbp_infl_idx

    def test_nominal_before_equities(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """Nominal appears before all equities."""
        order = multi_economy_graph.topological_order()
        gbp_nom_idx = order.index("gbp_nominal")
        for label in order:
            if "eq" in label and label.startswith("gbp"):
                assert gbp_nom_idx < order.index(label)

    def test_real_before_salary(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """Real rate appears before salary."""
        order = multi_economy_graph.topological_order()
        gbp_real_idx = order.index("gbp_real")
        gbp_sal_idx = order.index("gbp_salary")
        assert gbp_real_idx < gbp_sal_idx

    def test_all_models_present(
        self,
        multi_economy_graph: ModelDependencyGraph,
        gbp_economy: Economy,
        usd_economy: Economy,
    ) -> None:
        """All model labels appear in topological order."""
        order = multi_economy_graph.topological_order()
        expected_labels = {m.label for m in gbp_economy.all_models} | {
            m.label for m in usd_economy.all_models
        }
        assert set(order) == expected_labels

    def test_no_duplicates(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """No duplicate labels in topological order."""
        order = multi_economy_graph.topological_order()
        assert len(order) == len(set(order))


class TestExecutionLayers:
    """Tests for parallel execution layers."""

    def test_layers_cover_all_models(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """All models appear exactly once across all layers."""
        layers = multi_economy_graph.execution_layers()
        all_labels = [label for layer in layers for label in layer]
        assert len(all_labels) == len(set(all_labels))
        order = multi_economy_graph.topological_order()
        assert set(all_labels) == set(order)

    def test_first_layer_contains_nominals(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """First layer contains nominal rates (no dependencies)."""
        layers = multi_economy_graph.execution_layers()
        first_layer = set(layers[0])
        assert "gbp_nominal" in first_layer
        assert "usd_nominal" in first_layer

    def test_fx_not_in_first_layer(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """FX models are not in the first layer."""
        layers = multi_economy_graph.execution_layers()
        first_layer = set(layers[0])
        assert "usd_fx" not in first_layer

    def test_multiple_layers(
        self, multi_economy_graph: ModelDependencyGraph
    ) -> None:
        """Graph produces more than one execution layer."""
        layers = multi_economy_graph.execution_layers()
        assert len(layers) > 1
