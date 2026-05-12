"""Tests for the 14 post-processor types."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.engine.post_processing.protocol import PostProcessor, SimulationResults
from hyesg.engine.post_processing.processors import (
    AnnualisationProcessor,
    ConditionalExpectationProcessor,
    CurrencyConversionProcessor,
    CustomProcessor,
    EquilibriumSwapRateProcessor,
    EstimateFilterProcessor,
    FeeDeductionProcessor,
    InflationAdjustmentProcessor,
    LSMCRegressionProcessor,
    OutputFormattingProcessor,
    PathStatisticsProcessor,
    PercentileExtractionProcessor,
    PortfolioAggregationProcessor,
    SABRCalibrationProcessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(
    n_trials: int = 100,
    n_steps: int = 24,
    models: dict[str, jnp.ndarray] | None = None,
) -> SimulationResults:
    """Build a minimal ``SimulationResults`` for testing."""
    key = jax.random.PRNGKey(42)
    if models is None:
        models = {
            "equity": jax.random.normal(key, shape=(n_trials, n_steps)) * 0.1 + 1.0,
        }
    tg = jnp.linspace(0.0, 2.0, n_steps)
    return SimulationResults(
        paths=models,
        time_grid=tg,
        n_trials=n_trials,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Every concrete processor must satisfy the PostProcessor protocol."""

    @pytest.mark.parametrize(
        "cls",
        [
            CustomProcessor,
            PathStatisticsProcessor,
            PercentileExtractionProcessor,
            PortfolioAggregationProcessor,
            FeeDeductionProcessor,
            InflationAdjustmentProcessor,
            AnnualisationProcessor,
            CurrencyConversionProcessor,
            OutputFormattingProcessor,
            EstimateFilterProcessor,
            LSMCRegressionProcessor,
            SABRCalibrationProcessor,
            EquilibriumSwapRateProcessor,
        ],
    )
    def test_is_post_processor(self, cls):
        assert hasattr(cls, "process")


# ---------------------------------------------------------------------------
# 1. CustomProcessor
# ---------------------------------------------------------------------------


class TestCustomProcessor:
    def test_applies_function(self):
        def double_paths(r: SimulationResults) -> SimulationResults:
            new_paths = {k: v * 2.0 for k, v in r.paths.items()}
            return r.model_copy(update={"paths": new_paths})

        proc = CustomProcessor(fn=double_paths)
        res = _make_results()
        out = proc.process(res)
        assert jnp.allclose(out.paths["equity"], res.paths["equity"] * 2.0)

    def test_identity_function(self):
        proc = CustomProcessor(fn=lambda r: r)
        res = _make_results()
        out = proc.process(res)
        assert jnp.allclose(out.paths["equity"], res.paths["equity"])


# ---------------------------------------------------------------------------
# 2. PathStatisticsProcessor
# ---------------------------------------------------------------------------


class TestPathStatisticsProcessor:
    def test_computes_mean_and_std(self):
        proc = PathStatisticsProcessor()
        res = _make_results()
        out = proc.process(res)
        stats = out.metadata["statistics"]
        assert "equity" in stats
        assert "mean" in stats["equity"]
        assert "std" in stats["equity"]

    def test_computes_percentiles(self):
        proc = PathStatisticsProcessor(percentiles=(0.1, 0.5, 0.9))
        res = _make_results()
        out = proc.process(res)
        stats = out.metadata["statistics"]["equity"]
        assert "p10" in stats
        assert "p50" in stats
        assert "p90" in stats

    def test_output_shape(self):
        proc = PathStatisticsProcessor()
        res = _make_results(n_trials=50, n_steps=10)
        out = proc.process(res)
        mean = out.metadata["statistics"]["equity"]["mean"]
        assert mean.shape == (10,)


# ---------------------------------------------------------------------------
# 3. PercentileExtractionProcessor
# ---------------------------------------------------------------------------


class TestPercentileExtractionProcessor:
    def test_adds_percentile_paths(self):
        proc = PercentileExtractionProcessor(percentiles=(0.25, 0.75))
        res = _make_results()
        out = proc.process(res)
        assert "equity_p25" in out.paths
        assert "equity_p75" in out.paths

    def test_filters_by_model(self):
        models = {
            "equity": jnp.ones((10, 5)),
            "bond": jnp.ones((10, 5)),
        }
        proc = PercentileExtractionProcessor(percentiles=(0.5,), models=["equity"])
        res = _make_results(models=models)
        out = proc.process(res)
        assert "equity_p50" in out.paths
        assert "bond_p50" not in out.paths


# ---------------------------------------------------------------------------
# 4. ConditionalExpectationProcessor
# ---------------------------------------------------------------------------


class TestConditionalExpectationProcessor:
    def test_conditional_mean(self):
        key = jax.random.PRNGKey(0)
        equity = jax.random.normal(key, shape=(200, 12)) * 0.1 + 1.0
        models = {"equity": equity, "bond": jnp.ones((200, 12))}
        proc = ConditionalExpectationProcessor(
            condition_model="equity",
            condition_field="paths",
            condition_fn=lambda x: x[:, -1] > 1.0,
            target_models=["bond"],
        )
        res = _make_results(n_trials=200, n_steps=12, models=models)
        out = proc.process(res)
        assert "conditional_expectations" in out.metadata
        assert "bond" in out.metadata["conditional_expectations"]

    def test_no_matching_trials(self):
        models = {"equity": jnp.ones((10, 5)) * -100.0}
        proc = ConditionalExpectationProcessor(
            condition_model="equity",
            condition_field="paths",
            condition_fn=lambda x: x[:, -1] > 999.0,
            target_models=["equity"],
        )
        res = _make_results(n_trials=10, n_steps=5, models=models)
        out = proc.process(res)
        # Should not crash — divides by max(n_valid, 1)
        assert "conditional_expectations" in out.metadata


# ---------------------------------------------------------------------------
# 5. PortfolioAggregationProcessor
# ---------------------------------------------------------------------------


class TestPortfolioAggregationProcessor:
    def test_weighted_sum(self):
        models = {
            "equity": jnp.ones((10, 5)) * 2.0,
            "bond": jnp.ones((10, 5)) * 3.0,
        }
        proc = PortfolioAggregationProcessor(portfolio_weights={"equity": 0.6, "bond": 0.4})
        res = _make_results(n_trials=10, n_steps=5, models=models)
        out = proc.process(res)
        expected = 0.6 * 2.0 + 0.4 * 3.0
        assert jnp.allclose(out.paths["portfolio"], jnp.full((10, 5), expected), atol=1e-6)

    def test_single_asset(self):
        models = {"equity": jnp.ones((5, 3)) * 4.0}
        proc = PortfolioAggregationProcessor(portfolio_weights={"equity": 1.0})
        res = _make_results(n_trials=5, n_steps=3, models=models)
        out = proc.process(res)
        assert jnp.allclose(out.paths["portfolio"], jnp.full((5, 3), 4.0))


# ---------------------------------------------------------------------------
# 6. FeeDeductionProcessor
# ---------------------------------------------------------------------------


class TestFeeDeductionProcessor:
    def test_reduces_values(self):
        models = {"equity": jnp.ones((10, 5))}
        proc = FeeDeductionProcessor(fee_schedule={"equity": 50.0})
        tg = jnp.linspace(0.0, 1.0, 5)
        res = SimulationResults(
            paths=models, time_grid=tg, n_trials=10, n_steps=5,
        )
        out = proc.process(res)
        # Fees should reduce values over time
        assert float(jnp.mean(out.paths["equity"][:, -1])) < 1.0

    def test_zero_fee(self):
        models = {"equity": jnp.ones((10, 5))}
        proc = FeeDeductionProcessor(fee_schedule={"equity": 0.0})
        res = _make_results(n_trials=10, n_steps=5, models=models)
        out = proc.process(res)
        assert jnp.allclose(out.paths["equity"], jnp.ones((10, 5)))


# ---------------------------------------------------------------------------
# 7. InflationAdjustmentProcessor
# ---------------------------------------------------------------------------


class TestInflationAdjustmentProcessor:
    def test_creates_real_paths(self):
        models = {
            "equity": jnp.ones((10, 5)) * 2.0,
            "inflation": jnp.ones((10, 5)) * 1.5,
        }
        proc = InflationAdjustmentProcessor(inflation_model="inflation")
        res = _make_results(n_trials=10, n_steps=5, models=models)
        out = proc.process(res)
        assert "equity_real" in out.paths

    def test_no_inflation_model(self):
        models = {"equity": jnp.ones((10, 5))}
        proc = InflationAdjustmentProcessor(inflation_model="inflation")
        res = _make_results(n_trials=10, n_steps=5, models=models)
        out = proc.process(res)
        # Should return unchanged
        assert "equity_real" not in out.paths


# ---------------------------------------------------------------------------
# 8. AnnualisationProcessor
# ---------------------------------------------------------------------------


class TestAnnualisationProcessor:
    def test_geometric_annualisation(self):
        models = {"equity": jnp.ones((10, 24)) * 1.01}
        proc = AnnualisationProcessor(frequency=12, method="geometric")
        res = _make_results(n_trials=10, n_steps=24, models=models)
        out = proc.process(res)
        assert "equity_annual" in out.paths
        annual = out.paths["equity_annual"]
        assert annual.shape == (10, 2)

    def test_arithmetic_annualisation(self):
        models = {"equity": jnp.ones((10, 12)) * 5.0}
        proc = AnnualisationProcessor(frequency=12, method="arithmetic")
        res = _make_results(n_trials=10, n_steps=12, models=models)
        out = proc.process(res)
        assert "equity_annual" in out.paths
        assert jnp.allclose(out.paths["equity_annual"], 5.0)


# ---------------------------------------------------------------------------
# 9. CurrencyConversionProcessor
# ---------------------------------------------------------------------------


class TestCurrencyConversionProcessor:
    def test_converts_currency(self):
        models = {
            "equity_usd": jnp.ones((10, 5)) * 100.0,
            "fx_usd_gbp": jnp.ones((10, 5)) * 0.8,
        }
        proc = CurrencyConversionProcessor(fx_model="fx_usd_gbp", target_models=["equity_usd"])
        res = _make_results(n_trials=10, n_steps=5, models=models)
        out = proc.process(res)
        assert "equity_usd_base_ccy" in out.paths
        assert jnp.allclose(out.paths["equity_usd_base_ccy"], 80.0)

    def test_missing_fx_model(self):
        models = {"equity": jnp.ones((10, 5))}
        proc = CurrencyConversionProcessor(fx_model="fx_missing", target_models=["equity"])
        res = _make_results(n_trials=10, n_steps=5, models=models)
        out = proc.process(res)
        assert "equity_base_ccy" not in out.paths


# ---------------------------------------------------------------------------
# 10. OutputFormattingProcessor
# ---------------------------------------------------------------------------


class TestOutputFormattingProcessor:
    def test_rounds_values(self):
        models = {"equity": jnp.array([[1.123456789]])}
        proc = OutputFormattingProcessor(decimal_places=2)
        res = _make_results(n_trials=1, n_steps=1, models=models)
        out = proc.process(res)
        assert jnp.allclose(out.paths["equity"], jnp.array([[1.12]]), atol=1e-8)

    def test_sets_format_metadata(self):
        proc = OutputFormattingProcessor(format="csv", decimal_places=4)
        res = _make_results()
        out = proc.process(res)
        assert out.metadata["output_format"] == "csv"
        assert out.metadata["decimal_places"] == 4


# ---------------------------------------------------------------------------
# 11. EstimateFilterProcessor
# ---------------------------------------------------------------------------


class TestEstimateFilterProcessor:
    def test_zeros_below_threshold(self):
        models = {"equity": jnp.array([[1.0, -1.0, 2.0, -0.5]])}
        proc = EstimateFilterProcessor(filter_model="equity", threshold=0.0)
        res = _make_results(n_trials=1, n_steps=4, models=models)
        out = proc.process(res)
        expected = jnp.array([[1.0, 0.0, 2.0, 0.0]])
        assert jnp.allclose(out.paths["equity"], expected)

    def test_missing_filter_model(self):
        models = {"equity": jnp.ones((5, 3))}
        proc = EstimateFilterProcessor(filter_model="missing")
        res = _make_results(n_trials=5, n_steps=3, models=models)
        out = proc.process(res)
        assert jnp.allclose(out.paths["equity"], jnp.ones((5, 3)))


# ---------------------------------------------------------------------------
# 12. LSMCRegressionProcessor
# ---------------------------------------------------------------------------


class TestLSMCRegressionProcessor:
    def test_produces_coefficients(self):
        key = jax.random.PRNGKey(1)
        models = {"equity": jax.random.normal(key, shape=(50, 24)) + 1.0}
        proc = LSMCRegressionProcessor(basis_degree=2, n_exercise_dates=4)
        res = _make_results(n_trials=50, n_steps=24, models=models)
        out = proc.process(res)
        assert "lsmc" in out.metadata
        assert "equity" in out.metadata["lsmc"]
        assert "coefficients" in out.metadata["lsmc"]["equity"]

    def test_skips_1d_paths(self):
        models = {"scalar": jnp.ones((10,))}
        proc = LSMCRegressionProcessor()
        res = _make_results(n_trials=10, n_steps=10, models=models)
        out = proc.process(res)
        assert "scalar" not in out.metadata.get("lsmc", {})


# ---------------------------------------------------------------------------
# 13. SABRCalibrationProcessor
# ---------------------------------------------------------------------------


class TestSABRCalibrationProcessor:
    def test_produces_sabr_params(self):
        key = jax.random.PRNGKey(2)
        # Ensure positive values for log
        models = {"swaption_vol": jnp.abs(jax.random.normal(key, shape=(50, 12))) + 0.5}
        proc = SABRCalibrationProcessor(expiries=[1.0, 2.0], tenors=[5.0, 10.0])
        res = _make_results(n_trials=50, n_steps=12, models=models)
        out = proc.process(res)
        assert "sabr_calibration" in out.metadata
        assert "swaption_vol" in out.metadata["sabr_calibration"]
        params = out.metadata["sabr_calibration"]["swaption_vol"]
        assert "alpha" in params
        assert "nu" in params


# ---------------------------------------------------------------------------
# 14. EquilibriumSwapRateProcessor
# ---------------------------------------------------------------------------


class TestEquilibriumSwapRateProcessor:
    def test_produces_swap_rate(self):
        # Simulate discount factors declining from 1.0
        t = jnp.linspace(0.0, 1.0, 24)
        df = jnp.exp(-0.03 * t)
        models = {"zcb": jnp.tile(df, (20, 1))}
        proc = EquilibriumSwapRateProcessor(tenor=2.0, fixed_frequency=2)
        res = _make_results(n_trials=20, n_steps=24, models=models)
        out = proc.process(res)
        assert "zcb_swap_rate" in out.paths

    def test_skips_1d_paths(self):
        models = {"scalar": jnp.ones((10,))}
        proc = EquilibriumSwapRateProcessor()
        res = _make_results(n_trials=10, n_steps=10, models=models)
        out = proc.process(res)
        assert "scalar_swap_rate" not in out.paths
