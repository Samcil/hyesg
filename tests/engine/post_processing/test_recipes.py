"""Tests for PostProcessingRecipe and CompositeProcessor."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.engine.post_processing.protocol import PostProcessor, SimulationResults
from hyesg.engine.post_processing.processors import (
    CustomProcessor,
    OutputFormattingProcessor,
    PathStatisticsProcessor,
    PortfolioAggregationProcessor,
)
from hyesg.engine.post_processing.recipes import (
    CompositeProcessor,
    PostProcessingRecipe,
)


def _make_results(n_trials: int = 10, n_steps: int = 5) -> SimulationResults:
    key = jax.random.PRNGKey(99)
    return SimulationResults(
        paths={"equity": jax.random.normal(key, shape=(n_trials, n_steps)) + 1.0},
        time_grid=jnp.linspace(0.0, 1.0, n_steps),
        n_trials=n_trials,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# PostProcessingRecipe
# ---------------------------------------------------------------------------


class TestPostProcessingRecipe:
    def test_empty_recipe(self):
        recipe = PostProcessingRecipe()
        res = _make_results()
        out = recipe.execute(res)
        assert out.raw is res
        assert out.processed == res.paths

    def test_single_processor(self):
        recipe = PostProcessingRecipe([PathStatisticsProcessor()])
        res = _make_results()
        out = recipe.execute(res)
        assert "statistics" in out.statistics or "statistics" in out.metadata

    def test_chained_processors(self):
        recipe = (
            PostProcessingRecipe()
            .add(PathStatisticsProcessor())
            .add(OutputFormattingProcessor(decimal_places=2))
        )
        res = _make_results()
        out = recipe.execute(res)
        assert "processors_applied" in out.metadata
        assert len(out.metadata["processors_applied"]) == 2

    def test_fluent_add_returns_self(self):
        recipe = PostProcessingRecipe()
        returned = recipe.add(PathStatisticsProcessor())
        assert returned is recipe

    def test_validate_empty(self):
        recipe = PostProcessingRecipe()
        warnings = recipe.validate()
        assert any("no processors" in w.lower() for w in warnings)

    def test_validate_duplicates(self):
        recipe = PostProcessingRecipe([
            PathStatisticsProcessor(),
            PathStatisticsProcessor(),
        ])
        warnings = recipe.validate()
        assert any("duplicate" in w.lower() for w in warnings)

    def test_validate_ok(self):
        recipe = PostProcessingRecipe([
            PathStatisticsProcessor(),
            OutputFormattingProcessor(),
        ])
        warnings = recipe.validate()
        assert warnings == []

    def test_len(self):
        recipe = PostProcessingRecipe()
        assert len(recipe) == 0
        recipe.add(PathStatisticsProcessor())
        assert len(recipe) == 1

    def test_processors_property(self):
        proc = PathStatisticsProcessor()
        recipe = PostProcessingRecipe([proc])
        assert recipe.processors == [proc]

    def test_execute_preserves_raw(self):
        recipe = PostProcessingRecipe([
            CustomProcessor(fn=lambda r: r.model_copy(update={"paths": {}})),
        ])
        res = _make_results()
        out = recipe.execute(res)
        # Raw should still have original paths
        assert "equity" in out.raw.paths

    def test_multi_model_pipeline(self):
        models = {
            "equity": jnp.ones((10, 5)) * 2.0,
            "bond": jnp.ones((10, 5)) * 3.0,
        }
        res = SimulationResults(
            paths=models,
            time_grid=jnp.linspace(0.0, 1.0, 5),
            n_trials=10,
            n_steps=5,
        )
        recipe = (
            PostProcessingRecipe()
            .add(PortfolioAggregationProcessor({"equity": 0.6, "bond": 0.4}))
            .add(PathStatisticsProcessor())
        )
        out = recipe.execute(res)
        assert "portfolio" in out.processed


# ---------------------------------------------------------------------------
# CompositeProcessor
# ---------------------------------------------------------------------------


class TestCompositeProcessor:
    def test_basic_composite(self):
        inner = CompositeProcessor(
            name="stats_and_format",
            processors=[
                PathStatisticsProcessor(),
                OutputFormattingProcessor(decimal_places=3),
            ],
        )
        res = _make_results()
        out = inner.process(res)
        assert "statistics" in out.metadata
        assert "output_format" in out.metadata

    def test_composite_name(self):
        comp = CompositeProcessor(name="my_pipeline", processors=[])
        assert comp.name == "my_pipeline"

    def test_composite_in_recipe(self):
        inner = CompositeProcessor(
            name="inner",
            processors=[PathStatisticsProcessor()],
        )
        recipe = PostProcessingRecipe([inner])
        res = _make_results()
        out = recipe.execute(res)
        assert "statistics" in out.metadata

    def test_nested_composite(self):
        inner = CompositeProcessor(
            name="level1",
            processors=[PathStatisticsProcessor()],
        )
        outer = CompositeProcessor(
            name="level0",
            processors=[inner, OutputFormattingProcessor(decimal_places=2)],
        )
        res = _make_results()
        out = outer.process(res)
        assert "statistics" in out.metadata
        assert "output_format" in out.metadata

    def test_empty_composite(self):
        comp = CompositeProcessor(name="empty", processors=[])
        res = _make_results()
        out = comp.process(res)
        assert jnp.allclose(out.paths["equity"], res.paths["equity"])
