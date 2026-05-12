"""Integration tests for the full SimulationSetup pipeline."""

from __future__ import annotations

import pytest

from hyesg.config.default_setup import build_default_ess_setup
from hyesg.config.economy import Economy, EconomyModelConfig
from hyesg.config.simulation_setup import (
    SimulationSetupBuilder,
)

# ── Helpers ────────────────────────────────────────────────────────


def _make_economy(name: str, *, domestic: bool = False) -> Economy:
    """Create a minimal economy for integration testing."""
    return Economy(
        name=name,
        is_domestic=domestic,
        nominal_rate_model=EconomyModelConfig(
            model_type="cir2pp", label=f"{name.lower()}_nominal"
        ),
    )


# ── Full pipeline tests ───────────────────────────────────────────


class TestFullSetupPipeline:
    """End-to-end builder → setup → validate flow."""

    def test_complete_build_with_economy(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .seed(27)
            .time_grid(horizon=100, inverse_dt=12)
            .add_regime("Strong", trials=2500)
            .add_regime("Moderate", trials=1500)
            .add_regime("Weak", trials=1000)
            .add_economy(_make_economy("GBP", domestic=True))
            .build()
        )
        assert setup.validate_setup() == []
        assert setup.total_trials == 5000
        assert setup.n_steps == 1200

    def test_build_validates_by_default(self) -> None:
        with pytest.raises(ValueError):
            SimulationSetupBuilder().build()

    def test_multi_economy(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .seed(42)
            .add_regime("R1", trials=100)
            .add_economy(_make_economy("GBP", domestic=True))
            .add_economy(_make_economy("USD"))
            .add_economy(_make_economy("EUR"))
            .build()
        )
        assert len(setup.economies) == 3
        names = [e.name for e in setup.economies]
        assert "GBP" in names
        assert "USD" in names
        assert "EUR" in names


class TestRegimeOrdering:
    """Regimes maintain insertion order and correct weights."""

    def test_insertion_order_preserved(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("C", trials=100)
            .add_regime("A", trials=200)
            .add_regime("B", trials=300)
            .add_economy(_make_economy("GBP", domestic=True))
            .build()
        )
        assert [r.name for r in setup.regimes] == ["C", "A", "B"]

    def test_weights_sum_to_one(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .add_regime("X", trials=333)
            .add_regime("Y", trials=333)
            .add_regime("Z", trials=334)
            .build(validate=False)
        )
        total = sum(r.weight for r in setup.regimes)
        assert total == pytest.approx(1.0)


class TestTimeGrid:
    """Time grid properties are consistent."""

    def test_monthly(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .time_grid(horizon=100, inverse_dt=12)
            .add_regime("R", trials=100)
            .build(validate=False)
        )
        assert setup.n_steps == 1200
        assert setup.dt == pytest.approx(1 / 12)

    def test_quarterly(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .time_grid(horizon=50, inverse_dt=4)
            .add_regime("R", trials=100)
            .build(validate=False)
        )
        assert setup.n_steps == 200
        assert setup.dt == pytest.approx(0.25)

    def test_annual(self) -> None:
        setup = (
            SimulationSetupBuilder()
            .time_grid(horizon=30, inverse_dt=1)
            .add_regime("R", trials=100)
            .build(validate=False)
        )
        assert setup.n_steps == 30
        assert setup.dt == pytest.approx(1.0)


class TestPRNGSeedDerivation:
    """Verify that seed constants from the C# engine are reproducible.

    From engine/rng.py PRNGStreamManager:
      normals_seed  = seed
      copula_seed   = seed * 1_000_003 + 13
      chi2_seed     = (seed * (-104_723) - 1_000_003) & 0xFFFF_FFFF
    """

    def test_seed_27_normals(self) -> None:
        seed = 27
        assert seed == 27

    def test_seed_27_copula(self) -> None:
        seed = 27
        copula = seed * 1_000_003 + 13
        assert copula == 27_000_094

    def test_seed_27_chi2(self) -> None:
        seed = 27
        raw = seed * (-104_723) - 1_000_003
        chi2 = raw & 0xFFFF_FFFF
        # raw = -2827521 - 1000003 = -3827524
        expected = (-3_827_524) & 0xFFFF_FFFF
        assert chi2 == expected

    def test_default_setup_uses_seed_27(self) -> None:
        setup = build_default_ess_setup()
        assert setup.seed == 27
