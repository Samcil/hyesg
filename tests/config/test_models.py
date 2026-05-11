"""Tests for hyesg.config.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hyesg.config.models import (
    CorrelationEntry,
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)


class TestTimeGridConfig:
    def test_monthly(self) -> None:
        tg = TimeGridConfig(start_year=0.0, end_year=1.0, frequency="monthly")
        assert tg.n_steps == 12
        assert tg.dt == pytest.approx(1.0 / 12.0)
        pts = tg.time_points
        assert pts.shape == (13,)
        assert float(pts[0]) == pytest.approx(0.0)
        assert float(pts[-1]) == pytest.approx(1.0)

    def test_quarterly(self) -> None:
        tg = TimeGridConfig(start_year=0.0, end_year=1.0, frequency="quarterly")
        assert tg.n_steps == 4
        assert tg.dt == pytest.approx(0.25)

    def test_annual(self) -> None:
        tg = TimeGridConfig(start_year=0.0, end_year=5.0, frequency="annual")
        assert tg.n_steps == 5
        pts = tg.time_points
        assert pts.shape == (6,)

    def test_custom_times(self) -> None:
        tg = TimeGridConfig(custom_times=[0.0, 0.5, 1.0, 2.0, 5.0])
        assert tg.n_steps == 4
        pts = tg.time_points
        assert pts.shape == (5,)
        assert float(pts[0]) == 0.0
        assert float(pts[-1]) == 5.0

    def test_custom_times_not_increasing_raises(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            TimeGridConfig(custom_times=[0.0, 1.0, 0.5])

    def test_custom_times_too_short_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 2"):
            TimeGridConfig(custom_times=[0.0])


class TestModelConfig:
    def test_minimal(self) -> None:
        m = ModelConfig(type="cir2pp", name="nominal")
        assert m.type == "cir2pp"
        assert m.dependencies == []

    def test_with_params(self) -> None:
        m = ModelConfig(
            type="cir2pp",
            name="nominal",
            params={"alpha": 0.1, "mu": 0.03},
        )
        assert m.params["alpha"] == 0.1


class TestCorrelationEntry:
    def test_valid(self) -> None:
        c = CorrelationEntry(shock_a="z1", shock_b="z2", value=0.3)
        assert c.value == 0.3

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            CorrelationEntry(shock_a="z1", shock_b="z2", value=1.5)


class TestRegimeConfig:
    def test_defaults(self) -> None:
        r = RegimeConfig(name="base")
        assert r.n_trials == 1667
        assert r.seed == 27
        assert r.use_antithetic is True

    def test_zero_trials_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RegimeConfig(name="bad", n_trials=0)


class TestSimulationConfig:
    def test_valid_construction(self) -> None:
        cfg = SimulationConfig(
            name="test",
            models=[
                ModelConfig(type="cir2pp", name="nominal"),
                ModelConfig(
                    type="g2pp",
                    name="real",
                    dependencies=["nominal"],
                ),
            ],
            regimes=[RegimeConfig(name="r1")],
            output_models=["nominal"],
        )
        assert cfg.name == "test"
        assert len(cfg.models) == 2

    def test_missing_dependency_raises(self) -> None:
        with pytest.raises(ValidationError, match="not defined"):
            SimulationConfig(
                name="bad",
                models=[
                    ModelConfig(
                        type="fca",
                        name="inflation",
                        dependencies=["nonexistent"],
                    ),
                ],
            )

    def test_missing_output_model_raises(self) -> None:
        with pytest.raises(ValidationError, match="not defined"):
            SimulationConfig(
                name="bad",
                models=[
                    ModelConfig(type="cir2pp", name="nominal"),
                ],
                output_models=["nonexistent"],
            )

    def test_json_roundtrip(self) -> None:
        cfg = SimulationConfig(
            name="roundtrip",
            models=[
                ModelConfig(type="cir2pp", name="nominal"),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=100)],
        )
        json_str = cfg.model_dump_json()
        cfg2 = SimulationConfig.model_validate_json(json_str)
        assert cfg2.name == "roundtrip"
        assert len(cfg2.models) == 1
        assert cfg2.regimes[0].n_trials == 100
