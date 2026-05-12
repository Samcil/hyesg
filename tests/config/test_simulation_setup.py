"""Tests for SimulationSetup, SimulationSetupBuilder, and default ESS setup."""

from __future__ import annotations

import pytest

from hyesg.config.default_setup import build_default_ess_setup
from hyesg.config.economy import Economy, EconomyModelConfig
from hyesg.config.simulation_setup import (
    SetupRegimeConfig,
    SimulationSetup,
    SimulationSetupBuilder,
)

# ── SetupRegimeConfig ──────────────────────────────────────────────


class TestSetupRegimeConfig:
    """Tests for SetupRegimeConfig."""

    def test_create_with_defaults(self) -> None:
        rc = SetupRegimeConfig(name="Strong", trials=2500)
        assert rc.name == "Strong"
        assert rc.trials == 2500
        assert rc.weight == 0.0
        assert rc.calibration_params == {}

    def test_create_with_calibration_params(self) -> None:
        rc = SetupRegimeConfig(
            name="Weak", trials=1000, calibration_params={"alpha": 0.5}
        )
        assert rc.calibration_params == {"alpha": 0.5}

    def test_weight_mutable(self) -> None:
        rc = SetupRegimeConfig(name="Test", trials=100)
        rc.weight = 0.42
        assert rc.weight == pytest.approx(0.42)


# ── SimulationSetup ────────────────────────────────────────────────


class TestSimulationSetup:
    """Tests for SimulationSetup dataclass-like properties."""

    def test_defaults(self) -> None:
        setup = SimulationSetup()
        assert setup.seed == 27
        assert setup.horizon == 100
        assert setup.inverse_dt == 12

    def test_n_steps(self) -> None:
        setup = SimulationSetup(horizon=100, inverse_dt=12)
        assert setup.n_steps == 1200

    def test_n_steps_custom(self) -> None:
        setup = SimulationSetup(horizon=50, inverse_dt=4)
        assert setup.n_steps == 200

    def test_dt(self) -> None:
        setup = SimulationSetup(inverse_dt=12)
        assert setup.dt == pytest.approx(1.0 / 12.0)

    def test_dt_annual(self) -> None:
        setup = SimulationSetup(inverse_dt=1)
        assert setup.dt == pytest.approx(1.0)

    def test_total_trials_empty(self) -> None:
        setup = SimulationSetup()
        assert setup.total_trials == 0

    def test_total_trials_with_regimes(self) -> None:
        regimes = [
            SetupRegimeConfig(name="A", trials=100),
            SetupRegimeConfig(name="B", trials=200),
        ]
        setup = SimulationSetup(regimes=regimes)
        assert setup.total_trials == 300

    def test_validate_no_regimes(self) -> None:
        setup = SimulationSetup()
        errors = setup.validate_setup()
        assert "No regimes defined" in errors

    def test_validate_no_economies(self) -> None:
        regimes = [SetupRegimeConfig(name="R1", trials=100)]
        setup = SimulationSetup(regimes=regimes)
        errors = setup.validate_setup()
        assert "No economies defined" in errors

    def test_validate_zero_trials(self) -> None:
        regimes = [SetupRegimeConfig(name="R1", trials=0)]
        setup = SimulationSetup(regimes=regimes)
        errors = setup.validate_setup()
        assert "Total trials is 0" in errors

    def test_validate_valid_setup(self) -> None:
        regimes = [SetupRegimeConfig(name="R1", trials=100)]
        setup = SimulationSetup(regimes=regimes, economies=["dummy"])
        errors = setup.validate_setup()
        assert errors == []

    def test_arbitrary_types_allowed(self) -> None:
        """Correlation and funds can be any type."""
        setup = SimulationSetup(correlation="some_matrix", funds=42)
        assert setup.correlation == "some_matrix"
        assert setup.funds == 42


# ── SimulationSetupBuilder ─────────────────────────────────────────


class TestSimulationSetupBuilder:
    """Tests for the fluent builder API."""

    def test_builder_seed(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .seed(42)
            .add_regime("R", trials=10)
            .build(validate=False)
        )
        assert setup.seed == 42

    def test_builder_time_grid(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .time_grid(horizon=50, inverse_dt=4)
            .add_regime("R", trials=10)
            .build(validate=False)
        )
        assert setup.horizon == 50
        assert setup.inverse_dt == 4
        assert setup.n_steps == 200

    def test_builder_add_regime(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("Strong", trials=2500)
            .add_regime("Weak", trials=1000)
            .build(validate=False)
        )
        assert len(setup.regimes) == 2
        assert setup.regimes[0].name == "Strong"
        assert setup.regimes[1].name == "Weak"

    def test_builder_regime_weights_computed(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("A", trials=500)
            .add_regime("B", trials=500)
            .build(validate=False)
        )
        assert setup.regimes[0].weight == pytest.approx(0.5)
        assert setup.regimes[1].weight == pytest.approx(0.5)

    def test_builder_add_economy(self) -> None:
        economy = Economy(
            name="GBP",
            is_domestic=True,
            nominal_rate_model=EconomyModelConfig(
                model_type="cir2pp", label="gbp_nominal"
            ),
        )
        setup = (
            SimulationSetupBuilder()
            .add_regime("R", trials=100)
            .add_economy(economy)
            .build()
        )
        assert len(setup.economies) == 1
        assert setup.economies[0].name == "GBP"

    def test_builder_correlate(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("R", trials=10)
            .correlate("corr_matrix_placeholder")
            .build(validate=False)
        )
        assert setup.correlation == "corr_matrix_placeholder"

    def test_builder_fund_catalogue(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("R", trials=10)
            .add_fund_catalogue({"fund_a": 0.5})
            .build(validate=False)
        )
        assert setup.funds == {"fund_a": 0.5}

    def test_builder_post_processing(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("R", trials=10)
            .post_processing({"recipe": "sabr"})
            .build(validate=False)
        )
        assert setup.post_processing == {"recipe": "sabr"}

    def test_builder_fluent_chaining(self) -> None:
        """All builder methods return the builder itself."""
        builder = SimulationSetupBuilder()
        assert builder.seed(1) is builder
        assert builder.time_grid() is builder
        assert builder.add_regime("R", trials=1) is builder
        assert builder.add_economy("E") is builder
        assert builder.correlate(None) is builder
        assert builder.add_fund_catalogue(None) is builder
        assert builder.post_processing(None) is builder

    def test_builder_validate_raises(self) -> None:
        """Builder with validate=True raises on invalid setup."""
        with pytest.raises(ValueError, match="No economies defined"):
            SimulationSetupBuilder().add_regime("R", trials=10).build(validate=True)

    def test_builder_validate_false_no_raise(self) -> None:
        """Builder with validate=False does not raise on missing economies."""
        setup = (
            SimulationSetupBuilder().add_regime("R", trials=10).build(validate=False)
        )
        assert len(setup.economies) == 0

    def test_builder_no_regimes_validate_raises(self) -> None:
        with pytest.raises(ValueError, match="No regimes defined"):
            SimulationSetupBuilder().build(validate=True)

    def test_builder_regime_calibration_params(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("R", trials=100, alpha=0.5, beta=0.3)
            .build(validate=False)
        )
        assert setup.regimes[0].calibration_params == {"alpha": 0.5, "beta": 0.3}


# ── Default ESS Setup ──────────────────────────────────────────────


class TestDefaultESSSetup:
    """Tests that build_default_ess_setup matches C# constants."""

    def test_seed(self) -> None:
        setup = build_default_ess_setup()
        assert setup.seed == 27

    def test_n_steps(self) -> None:
        setup = build_default_ess_setup()
        assert setup.n_steps == 1200

    def test_dt(self) -> None:
        setup = build_default_ess_setup()
        assert setup.dt == pytest.approx(1.0 / 12.0)

    def test_total_trials(self) -> None:
        setup = build_default_ess_setup()
        assert setup.total_trials == 5000

    def test_three_regimes(self) -> None:
        setup = build_default_ess_setup()
        assert len(setup.regimes) == 3

    def test_regime_names(self) -> None:
        setup = build_default_ess_setup()
        names = [r.name for r in setup.regimes]
        assert names == ["Strong", "Moderate", "Weak"]

    def test_regime_trials(self) -> None:
        setup = build_default_ess_setup()
        trials = [r.trials for r in setup.regimes]
        assert trials == [2500, 1500, 1000]

    def test_regime_weights(self) -> None:
        setup = build_default_ess_setup()
        weights = [r.weight for r in setup.regimes]
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(0.3)
        assert weights[2] == pytest.approx(0.2)

    def test_horizon(self) -> None:
        setup = build_default_ess_setup()
        assert setup.horizon == 100

    def test_inverse_dt(self) -> None:
        setup = build_default_ess_setup()
        assert setup.inverse_dt == 12
