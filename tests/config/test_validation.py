"""Tests for hyesg.config.validation."""

from __future__ import annotations

import pytest

from hyesg.config.models import (
    CorrelationEntry,
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.validation import (
    build_dep_graph,
    find_cycle_path,
    has_cycles,
    validate_config,
)
from hyesg.core.registry import clear_registry, register_model


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _make_config(**overrides) -> SimulationConfig:
    """Helper to build a SimulationConfig with defaults."""
    defaults = {
        "name": "test",
        "models": [],
        "regimes": [RegimeConfig(name="r1", n_trials=1000)],
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestBuildDepGraph:
    def test_empty(self) -> None:
        cfg = _make_config()
        assert build_dep_graph(cfg) == {}

    def test_with_deps(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(type="a", name="m1"),
                ModelConfig(
                    type="b",
                    name="m2",
                    dependencies=["m1"],
                ),
            ]
        )
        g = build_dep_graph(cfg)
        assert g == {"m1": [], "m2": ["m1"]}


class TestCycleDetection:
    def test_no_cycle(self) -> None:
        graph = {"a": ["b"], "b": ["c"], "c": []}
        assert not has_cycles(graph)
        assert find_cycle_path(graph) is None

    def test_simple_cycle(self) -> None:
        graph = {"a": ["b"], "b": ["a"]}
        assert has_cycles(graph)
        cycle = find_cycle_path(graph)
        assert cycle is not None
        assert len(cycle) >= 2

    def test_self_cycle(self) -> None:
        graph = {"a": ["a"]}
        assert has_cycles(graph)

    def test_diamond_no_cycle(self) -> None:
        graph = {
            "a": ["b", "c"],
            "b": ["d"],
            "c": ["d"],
            "d": [],
        }
        assert not has_cycles(graph)


class TestValidateConfig:
    def test_valid_config(self) -> None:
        register_model("cir2pp")(type("FakeCIR", (), {}))
        cfg = _make_config(
            models=[
                ModelConfig(type="cir2pp", name="nominal"),
            ]
        )
        errors = validate_config(cfg)
        assert errors == []

    def test_missing_model_type(self) -> None:
        register_model("cir2pp")(type("FakeCIR", (), {}))
        cfg = _make_config(
            models=[
                ModelConfig(type="nonexistent", name="bad"),
            ]
        )
        errors = validate_config(cfg)
        assert any("not found in registry" in e for e in errors)

    def test_missing_dependency_caught_by_pydantic(self) -> None:
        """Pydantic validator catches missing deps at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="not defined"):
            SimulationConfig(
                name="test",
                models=[
                    ModelConfig(type="a", name="m1"),
                    ModelConfig(
                        type="b",
                        name="m3",
                        dependencies=["m1", "missing"],
                    ),
                ],
                regimes=[RegimeConfig(name="r1", n_trials=1000)],
            )

    def test_validate_config_catches_dependency_in_graph(
        self,
    ) -> None:
        """validate_config also checks dependencies via graph."""
        # Build a config with valid deps, then mutate to test
        cfg = _make_config(
            models=[
                ModelConfig(type="a", name="m1"),
                ModelConfig(
                    type="b",
                    name="m2",
                    dependencies=["m1"],
                ),
            ]
        )
        # Manually inject bad dep to bypass Pydantic
        cfg.models[1].dependencies.append("ghost")
        errors = validate_config(cfg)
        assert any("ghost" in e for e in errors)

    def test_circular_dependency(self) -> None:
        cfg = SimulationConfig(
            name="test",
            models=[
                ModelConfig(
                    type="a",
                    name="m1",
                    dependencies=["m2"],
                ),
                ModelConfig(
                    type="b",
                    name="m2",
                    dependencies=["m1"],
                ),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=1000)],
        )
        errors = validate_config(cfg)
        assert any("Circular" in e for e in errors)

    def test_non_psd_correlation(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(type="a", name="m1"),
            ],
            correlations=[
                CorrelationEntry(shock_a="z1", shock_b="z2", value=0.9),
                CorrelationEntry(shock_a="z1", shock_b="z3", value=0.9),
                CorrelationEntry(shock_a="z2", shock_b="z3", value=-0.9),
            ],
        )
        errors = validate_config(cfg)
        assert any("not positive semi-definite" in e for e in errors)

    def test_valid_psd_correlation(self) -> None:
        register_model("a")(type("FakeA", (), {}))
        cfg = _make_config(
            models=[
                ModelConfig(type="a", name="m1"),
            ],
            correlations=[
                CorrelationEntry(shock_a="z1", shock_b="z2", value=0.3),
            ],
        )
        errors = validate_config(cfg)
        assert not any("semi-definite" in e for e in errors)

    def test_cir_stability_warning(self) -> None:
        cfg = _make_config(
            time_grid=TimeGridConfig(
                start_year=0.0,
                end_year=1.0,
                frequency="annual",
            ),
            models=[
                ModelConfig(
                    type="cir",
                    name="fast_cir",
                    params={"alpha": 100.0},
                ),
            ],
        )
        errors = validate_config(cfg)
        assert any("CIR stability" in e for e in errors)

    def test_no_trials_warning(self) -> None:
        cfg = SimulationConfig(
            name="test",
            models=[],
            regimes=[],
        )
        errors = validate_config(cfg)
        assert any("No trials" in e for e in errors)

    def test_low_trial_warning(self) -> None:
        cfg = _make_config(regimes=[RegimeConfig(name="r1", n_trials=10)])
        errors = validate_config(cfg)
        assert any("Low trial count" in e for e in errors)
