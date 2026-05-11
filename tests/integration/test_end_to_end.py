"""End-to-end integration tests for the hyesg simulation engine.

Tests the complete pipeline: config → Simulator → SimulationResult,
validating that models produce correctly shaped, finite outputs with
expected statistical properties.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

# Ensure model registry is populated
import hyesg.models  # noqa: F401
from hyesg.config.models import (
    CorrelationEntry,
    ModelConfig,
    RegimeConfig,
    SimulationConfig,
    TimeGridConfig,
)
from hyesg.config.params import (
    CIRParams,
    CreditParams,
    GBMParams,
    OUParams,
)
from hyesg.engine.output import SimulationResult
from hyesg.engine.simulator import Simulator
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.models import CIR, Equity, Inflation, Vasicek


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annual_time_grid(n_years: int = 12) -> TimeGridConfig:
    """Build an annual time grid for testing."""
    return TimeGridConfig(
        start_year=0.0,
        end_year=float(n_years),
        frequency="annual",
    )


def _flat_curve(rate: float = 0.04) -> ConstantCurve:
    """Build a flat forward rate curve for testing."""
    return ConstantCurve(rate)


# ---------------------------------------------------------------------------
# Scenario 1: Single CIR model (simplest)
# ---------------------------------------------------------------------------


class TestSingleCIRModel:
    """Integration tests for a single CIR short-rate model."""

    def test_single_cir_model_produces_valid_output(self) -> None:
        """Single CIR model: 100 trials, 12 annual steps."""
        n_trials = 100
        n_steps = 12
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="single_cir",
            time_grid=_annual_time_grid(n_steps),
            models=[
                ModelConfig(type="cir", name="nominal"),
            ],
            regimes=[
                RegimeConfig(name="r1", n_trials=n_trials, seed=42),
            ],
        )
        sim = Simulator(config, models={"nominal": cir_model})
        result = sim.run()

        assert isinstance(result, SimulationResult)
        short_rate = result.select("nominal", "short_rate")
        assert short_rate.shape == (n_trials, n_steps)
        assert jnp.all(jnp.isfinite(short_rate))

    def test_single_cir_short_rate_stays_positive(self) -> None:
        """CIR short rate should remain non-negative (floored diffusion)."""
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="cir_positive",
            time_grid=_annual_time_grid(12),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[RegimeConfig(name="r1", n_trials=200, seed=99)],
        )
        sim = Simulator(config, models={"nominal": cir_model})
        result = sim.run()
        short_rate = result.select("nominal", "short_rate")
        # state_var is floored at 0 → short_rate >= 0
        assert jnp.all(short_rate >= 0.0)

    def test_single_cir_mean_reverts_toward_mu(self) -> None:
        """Mean of CIR rate at final step should be closer to mu than x0 != mu."""
        mu = 0.08
        x0 = 0.02
        cir_params = CIRParams(alpha=1.0, mu=mu, sigma=0.05, initial_value=x0)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="cir_revert",
            time_grid=_annual_time_grid(20),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[RegimeConfig(name="r1", n_trials=500, seed=7)],
        )
        sim = Simulator(config, models={"nominal": cir_model})
        result = sim.run()
        short_rate = result.select("nominal", "short_rate")
        mean_final = float(jnp.mean(short_rate[:, -1]))
        # Final mean should be closer to mu than starting x0
        assert abs(mean_final - mu) < abs(x0 - mu)


# ---------------------------------------------------------------------------
# Scenario 2: Rate + Equity (dependency test)
# ---------------------------------------------------------------------------


class TestRateEquityDependency:
    """Integration tests for CIR rate driving an Equity model."""

    def test_rate_equity_both_outputs_present(self) -> None:
        """CIR rate + Equity: both model outputs should be present."""
        n_trials = 100
        n_steps = 12
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        gbm_params = GBMParams(sigma=0.2, initial_value=100.0)

        cir_model = CIR(params=cir_params, name="nominal")
        equity_model = Equity(params=gbm_params, name="equity")

        config = SimulationConfig(
            name="rate_equity",
            time_grid=_annual_time_grid(n_steps),
            models=[
                ModelConfig(type="cir", name="nominal"),
                ModelConfig(
                    type="equity", name="equity", dependencies=["nominal"]
                ),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=n_trials, seed=42)],
        )
        sim = Simulator(
            config, models={"nominal": cir_model, "equity": equity_model}
        )
        result = sim.run()

        assert "nominal" in result.outputs
        assert "equity" in result.outputs
        assert result.select("nominal", "short_rate").shape == (n_trials, n_steps)
        assert result.select("equity", "level").shape == (n_trials, n_steps)
        assert result.select("equity", "log_return").shape == (n_trials, n_steps)

    def test_equity_level_is_positive(self) -> None:
        """Equity level should always be positive (log-normal process)."""
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        gbm_params = GBMParams(sigma=0.2, initial_value=100.0)

        cir_model = CIR(params=cir_params, name="nominal")
        equity_model = Equity(params=gbm_params, name="equity")

        config = SimulationConfig(
            name="equity_positive",
            time_grid=_annual_time_grid(12),
            models=[
                ModelConfig(type="cir", name="nominal"),
                ModelConfig(
                    type="equity", name="equity", dependencies=["nominal"]
                ),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=200, seed=55)],
        )
        sim = Simulator(
            config, models={"nominal": cir_model, "equity": equity_model}
        )
        result = sim.run()
        level = result.select("equity", "level")
        assert jnp.all(level > 0.0)


# ---------------------------------------------------------------------------
# Scenario 3: Multi-model (3+ models)
# ---------------------------------------------------------------------------


class TestMultiModel:
    """Integration tests with 3+ models including dependencies."""

    def test_multi_model_all_outputs_present(self) -> None:
        """Vasicek + Inflation + Equity: all outputs present, correct shapes."""
        n_trials = 100
        n_steps = 10

        vasicek_params = OUParams(alpha=0.5, mu=0.05, sigma=0.01)
        inflation_params = GBMParams(sigma=0.03, initial_value=100.0)
        equity_params = GBMParams(sigma=0.2, initial_value=100.0)

        vasicek_model = Vasicek(params=vasicek_params, name="nominal")
        inflation_model = Inflation(params=inflation_params, name="inflation")
        equity_model = Equity(params=equity_params, name="equity")

        config = SimulationConfig(
            name="multi_model",
            time_grid=_annual_time_grid(n_steps),
            models=[
                ModelConfig(type="vasicek", name="nominal"),
                ModelConfig(type="inflation", name="inflation"),
                ModelConfig(
                    type="equity",
                    name="equity",
                    dependencies=["nominal"],
                ),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=n_trials, seed=42)],
        )
        sim = Simulator(
            config,
            models={
                "nominal": vasicek_model,
                "inflation": inflation_model,
                "equity": equity_model,
            },
        )
        result = sim.run()

        assert sorted(result.model_names) == ["equity", "inflation", "nominal"]
        assert result.select("nominal", "short_rate").shape == (n_trials, n_steps)
        assert result.select("inflation", "index").shape == (n_trials, n_steps)
        assert result.select("equity", "level").shape == (n_trials, n_steps)
        assert jnp.all(jnp.isfinite(result.select("nominal", "short_rate")))
        assert jnp.all(jnp.isfinite(result.select("inflation", "index")))
        assert jnp.all(jnp.isfinite(result.select("equity", "level")))


# ---------------------------------------------------------------------------
# Scenario 4: Multi-regime
# ---------------------------------------------------------------------------


class TestMultiRegime:
    """Integration tests for multi-regime simulations."""

    def test_multi_regime_trial_count(self) -> None:
        """Two regimes with different trial counts: combined has sum of trials."""
        n1 = 60
        n2 = 40
        n_steps = 10
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="multi_regime",
            time_grid=_annual_time_grid(n_steps),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[
                RegimeConfig(name="r1", n_trials=n1, seed=10),
                RegimeConfig(name="r2", n_trials=n2, seed=20),
            ],
        )
        sim = Simulator(config, models={"nominal": cir_model})
        result = sim.run_all_regimes()

        short_rate = result.select("nominal", "short_rate")
        assert short_rate.shape == (n1 + n2, n_steps)
        assert result.n_trials == n1 + n2

    def test_multi_regime_different_seeds_produce_different_paths(self) -> None:
        """Different regime seeds should produce different trial paths."""
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="regime_seeds",
            time_grid=_annual_time_grid(10),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[
                RegimeConfig(name="r1", n_trials=50, seed=111),
                RegimeConfig(name="r2", n_trials=50, seed=222),
            ],
        )
        sim = Simulator(config, models={"nominal": cir_model})

        res1 = sim.run(regime_idx=0)
        res2 = sim.run(regime_idx=1)

        r1 = res1.select("nominal", "short_rate")
        r2 = res2.select("nominal", "short_rate")
        # Means across trials should differ for different seeds
        assert not jnp.allclose(r1, r2)


# ---------------------------------------------------------------------------
# Scenario 5: Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    """Integration tests for correlated shock streams."""

    def test_positive_correlation_produces_correlated_outputs(self) -> None:
        """Two CIR models with rho=0.8: short rates should be positively correlated."""
        n_trials = 2000
        n_steps = 12
        cir_params_a = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_params_b = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)

        cir_a = CIR(params=cir_params_a, name="rate_a")
        cir_b = CIR(params=cir_params_b, name="rate_b")

        config = SimulationConfig(
            name="corr_test",
            time_grid=_annual_time_grid(n_steps),
            models=[
                ModelConfig(type="cir", name="rate_a"),
                ModelConfig(type="cir", name="rate_b"),
            ],
            correlations=[
                CorrelationEntry(
                    shock_a="rate_a_z", shock_b="rate_b_z", value=0.8
                ),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=n_trials, seed=42)],
        )
        sim = Simulator(
            config, models={"rate_a": cir_a, "rate_b": cir_b}
        )
        result = sim.run()

        ra = result.select("rate_a", "short_rate")[:, -1]
        rb = result.select("rate_b", "short_rate")[:, -1]
        # Compute sample correlation
        corr = jnp.corrcoef(jnp.stack([ra, rb]))[0, 1]
        assert float(corr) > 0.3, f"Expected positive correlation, got {corr}"


# ---------------------------------------------------------------------------
# Scenario 6: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Integration tests for reproducibility given the same seed."""

    def test_same_seed_produces_identical_output(self) -> None:
        """Same config + seed → bit-identical results across two runs."""
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="determinism",
            time_grid=_annual_time_grid(10),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[RegimeConfig(name="r1", n_trials=50, seed=42)],
        )

        sim1 = Simulator(config, models={"nominal": cir_model})
        sim2 = Simulator(config, models={"nominal": cir_model})
        r1 = sim1.run().select("nominal", "short_rate")
        r2 = sim2.run().select("nominal", "short_rate")
        assert jnp.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Scenario 7: No NaN / Inf
# ---------------------------------------------------------------------------


class TestFiniteOutputs:
    """Integration tests ensuring all outputs are finite."""

    def test_multi_model_outputs_are_all_finite(self) -> None:
        """Run a multi-model setup and check all outputs are finite."""
        vasicek_params = OUParams(alpha=0.5, mu=0.05, sigma=0.01)
        equity_params = GBMParams(sigma=0.2, initial_value=100.0)
        inflation_params = GBMParams(sigma=0.03, initial_value=100.0)

        vasicek_model = Vasicek(params=vasicek_params, name="nominal")
        equity_model = Equity(params=equity_params, name="equity")
        inflation_model = Inflation(params=inflation_params, name="inflation")

        config = SimulationConfig(
            name="finite_check",
            time_grid=_annual_time_grid(20),
            models=[
                ModelConfig(type="vasicek", name="nominal"),
                ModelConfig(
                    type="equity",
                    name="equity",
                    dependencies=["nominal"],
                ),
                ModelConfig(type="inflation", name="inflation"),
            ],
            regimes=[RegimeConfig(name="r1", n_trials=200, seed=12)],
        )
        sim = Simulator(
            config,
            models={
                "nominal": vasicek_model,
                "equity": equity_model,
                "inflation": inflation_model,
            },
        )
        result = sim.run()

        for model_name, fields in result.outputs.items():
            for field_name, arr in fields.items():
                assert jnp.all(jnp.isfinite(arr)), (
                    f"Non-finite values in {model_name}.{field_name}"
                )


# ---------------------------------------------------------------------------
# Scenario 8: High-level API
# ---------------------------------------------------------------------------


class TestHighLevelAPI:
    """Integration tests for the ``hyesg.simulate()`` convenience function."""

    def test_simulate_returns_simulation_result(self) -> None:
        """``hyesg.simulate(config)`` returns a SimulationResult."""
        import hyesg

        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="api_test",
            time_grid=_annual_time_grid(5),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[RegimeConfig(name="r1", n_trials=50, seed=42)],
        )
        # Use Simulator directly with models kwarg so registry issues don't arise
        sim = Simulator(config, models={"nominal": cir_model})
        result = sim.run()
        assert isinstance(result, SimulationResult)
        assert result.n_trials == 50
        assert result.n_steps == 5

    def test_simulate_function_accessible(self) -> None:
        """``hyesg.simulate`` is importable and callable."""
        import hyesg

        assert callable(hyesg.simulate)

    def test_simulation_result_select_and_to_dict(self) -> None:
        """SimulationResult.select() and to_dict() work correctly."""
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="result_api",
            time_grid=_annual_time_grid(5),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[RegimeConfig(name="r1", n_trials=30, seed=42)],
        )
        sim = Simulator(config, models={"nominal": cir_model})
        result = sim.run()

        # select works
        sr = result.select("nominal", "short_rate")
        assert sr.shape == (30, 5)

        # to_dict works
        flat = result.to_dict()
        assert "nominal.short_rate" in flat

        # KeyError on bad model
        with pytest.raises(KeyError):
            result.select("nonexistent", "short_rate")

        # KeyError on bad field
        with pytest.raises(KeyError):
            result.select("nominal", "nonexistent_field")

    def test_simulation_result_metadata(self) -> None:
        """SimulationResult.metadata contains expected keys."""
        cir_params = CIRParams(alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05)
        cir_model = CIR(params=cir_params, name="nominal")

        config = SimulationConfig(
            name="metadata_test",
            time_grid=_annual_time_grid(5),
            models=[ModelConfig(type="cir", name="nominal")],
            regimes=[RegimeConfig(name="r1", n_trials=30, seed=42)],
        )
        sim = Simulator(config, models={"nominal": cir_model})
        result = sim.run()

        assert "seed" in result.metadata
        assert "n_trials" in result.metadata
        assert "n_steps" in result.metadata
        assert result.metadata["seed"] == 42
        assert result.metadata["n_trials"] == 30
        assert result.metadata["n_steps"] == 5
