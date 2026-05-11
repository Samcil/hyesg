"""Tests for hyesg.config.builder."""

from __future__ import annotations

from hyesg.config.builder import SimulationBuilder


class TestSimulationBuilder:
    def test_minimal_build(self) -> None:
        cfg = SimulationBuilder("test").build()
        assert cfg.name == "test"

    def test_fluent_chain(self) -> None:
        cfg = (
            SimulationBuilder("fluent")
            .description("A test sim")
            .time_grid(0.0, 50.0, "quarterly")
            .add_model("cir2pp", "nominal", alpha1=0.1)
            .add_model(
                "g2pp",
                "real",
                dependencies=["nominal"],
            )
            .correlate("z1", "z2", 0.3)
            .add_regime("r1", n_trials=1000, seed=42)
            .output_models("nominal", "real")
            .build()
        )
        assert cfg.description == "A test sim"
        assert cfg.time_grid.frequency == "quarterly"
        assert len(cfg.models) == 2
        assert cfg.models[0].params["alpha1"] == 0.1
        assert cfg.models[1].dependencies == ["nominal"]
        assert len(cfg.correlations) == 1
        assert cfg.correlations[0].value == 0.3
        assert len(cfg.regimes) == 1
        assert cfg.regimes[0].n_trials == 1000
        assert cfg.output_models == ["nominal", "real"]

    def test_multiple_regimes(self) -> None:
        cfg = (
            SimulationBuilder("multi")
            .add_regime("r1", n_trials=1667, seed=1)
            .add_regime("r2", n_trials=1667, seed=2)
            .add_regime("r3", n_trials=1666, seed=3)
            .build()
        )
        assert len(cfg.regimes) == 3
        total = sum(r.n_trials for r in cfg.regimes)
        assert total == 5000

    def test_model_with_outputs(self) -> None:
        cfg = (
            SimulationBuilder("outputs")
            .add_model(
                "cir2pp",
                "nominal",
                outputs=["ShortRate", "SpotRate"],
                output_maturities=[1.0, 5.0, 10.0],
            )
            .build()
        )
        m = cfg.models[0]
        assert m.outputs == ["ShortRate", "SpotRate"]
        assert m.output_maturities == [1.0, 5.0, 10.0]
