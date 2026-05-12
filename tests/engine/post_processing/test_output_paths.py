"""Tests for OutputPathSpec resolution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.engine.post_processing.output_paths import OutputPathSpec
from hyesg.engine.post_processing.protocol import SimulationResults


def _make_results() -> SimulationResults:
    key = jax.random.PRNGKey(7)
    return SimulationResults(
        paths={
            "equity": jnp.abs(jax.random.normal(key, shape=(10, 5))) + 0.5,
        },
        time_grid=jnp.linspace(0.0, 1.0, 5),
        n_trials=10,
        n_steps=5,
    )


class TestOutputPathSpec:
    def test_resolve_no_transform(self):
        spec = OutputPathSpec(model="equity", field="paths")
        res = _make_results()
        arr = spec.resolve(res)
        assert jnp.allclose(arr, res.paths["equity"])

    def test_resolve_log_transform(self):
        spec = OutputPathSpec(model="equity", field="paths", transform="log")
        res = _make_results()
        arr = spec.resolve(res)
        expected = jnp.log(jnp.maximum(res.paths["equity"], 1e-12))
        assert jnp.allclose(arr, expected)

    def test_resolve_cumulative_transform(self):
        spec = OutputPathSpec(model="equity", field="paths", transform="cumulative")
        res = _make_results()
        arr = spec.resolve(res)
        expected = jnp.cumsum(res.paths["equity"], axis=-1)
        assert jnp.allclose(arr, expected)

    def test_resolve_annualised_transform(self):
        spec = OutputPathSpec(model="equity", field="paths", transform="annualised")
        res = _make_results()
        arr = spec.resolve(res)
        assert arr.shape == res.paths["equity"].shape

    def test_resolve_missing_model_raises(self):
        spec = OutputPathSpec(model="missing", field="paths")
        res = _make_results()
        with pytest.raises(KeyError, match="missing"):
            spec.resolve(res)

    def test_label_attribute(self):
        spec = OutputPathSpec(model="equity", field="paths", label="Equity Returns")
        assert spec.label == "Equity Returns"

    def test_default_label_empty(self):
        spec = OutputPathSpec(model="equity", field="paths")
        assert spec.label == ""

    def test_transform_none(self):
        spec = OutputPathSpec(model="equity", field="paths", transform=None)
        res = _make_results()
        arr = spec.resolve(res)
        assert jnp.allclose(arr, res.paths["equity"])
