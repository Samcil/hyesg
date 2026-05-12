"""Tests for the portfolio aggregation module."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from hyesg.engine.output import SimulationResult
from hyesg.models.portfolio.analytics import PortfolioAnalytics
from hyesg.models.portfolio.portfolio import Portfolio
from hyesg.models.portfolio.result import PortfolioConfig, PortfolioResult

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_simulation_result(
    asset_returns: dict[str, jax.Array],
    n_steps: int | None = None,
    return_field: str = "LogReturn",
) -> SimulationResult:
    """Build a minimal SimulationResult from asset return arrays."""
    outputs: dict[str, dict[str, jax.Array]] = {}
    for name, arr in asset_returns.items():
        outputs[name] = {return_field: arr}
    if n_steps is None:
        first = next(iter(asset_returns.values()))
        n_steps = first.shape[1]
    time_grid = jnp.linspace(0.0, float(n_steps), n_steps + 1)
    return SimulationResult(outputs=outputs, time_grid=time_grid)


# ------------------------------------------------------------------
# PortfolioConfig validation
# ------------------------------------------------------------------


class TestPortfolioConfig:
    def test_valid_config(self) -> None:
        cfg = PortfolioConfig(weights={"a": 0.6, "b": 0.4})
        assert cfg.rebalance == "buy_and_hold"
        assert cfg.initial_value == 1.0

    def test_empty_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PortfolioConfig(weights={})

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            PortfolioConfig(weights={"a": -0.5, "b": 1.5})

    def test_invalid_rebalance_raises(self) -> None:
        with pytest.raises(ValueError, match="rebalance must be one of"):
            PortfolioConfig(weights={"a": 1.0}, rebalance="invalid")

    def test_zero_rebalance_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="rebalance_frequency must be >= 1"):
            PortfolioConfig(
                weights={"a": 1.0}, rebalance="periodic", rebalance_frequency=0
            )

    def test_weights_not_summing_to_one_warns(self) -> None:
        """Portfolio.aggregate should warn when weights don't sum to 1."""
        cfg = PortfolioConfig(weights={"a": 0.3, "b": 0.3})
        portfolio = Portfolio(cfg)
        result = _make_simulation_result(
            {"a": jnp.zeros((2, 3)), "b": jnp.zeros((2, 3))}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            portfolio.aggregate(result)
        assert any("sum to" in str(w.message) for w in caught)


# ------------------------------------------------------------------
# Constant-mix returns
# ------------------------------------------------------------------


class TestConstantMix:
    def test_two_asset_manual(self) -> None:
        """Weighted return matches hand calculation."""
        r_a = jnp.array([[0.10, 0.05, -0.02]])
        r_b = jnp.array([[0.02, -0.01, 0.04]])
        w_a, w_b = 0.6, 0.4

        expected = w_a * r_a + w_b * r_b  # (1, 3)

        cfg = PortfolioConfig(weights={"a": w_a, "b": w_b}, rebalance="constant_mix")
        portfolio = Portfolio(cfg)
        result = _make_simulation_result({"a": r_a, "b": r_b})
        out = portfolio.aggregate(result)

        assert jnp.allclose(out.returns, expected, atol=1e-12)
        assert out.weights_history is None

    def test_single_asset(self) -> None:
        """Single asset with weight 1.0 returns the asset return itself."""
        r = jnp.array([[0.05, -0.03, 0.02]])
        cfg = PortfolioConfig(weights={"x": 1.0}, rebalance="constant_mix")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"x": r}))
        assert jnp.allclose(out.returns, r, atol=1e-12)

    def test_all_zero_returns(self) -> None:
        """All-zero returns produce zero portfolio returns."""
        zeros = jnp.zeros((5, 10))
        cfg = PortfolioConfig(weights={"a": 0.5, "b": 0.5}, rebalance="constant_mix")
        result = _make_simulation_result({"a": zeros, "b": zeros})
        out = Portfolio(cfg).aggregate(result)
        assert jnp.allclose(out.returns, 0.0, atol=1e-12)

    def test_multiple_trials(self) -> None:
        """Works correctly with multiple trials."""
        r_a = jnp.array([[0.10, 0.05], [0.02, -0.01]])
        r_b = jnp.array([[0.02, -0.01], [0.04, 0.03]])
        cfg = PortfolioConfig(weights={"a": 0.7, "b": 0.3}, rebalance="constant_mix")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r_a, "b": r_b}))
        expected = 0.7 * r_a + 0.3 * r_b
        assert jnp.allclose(out.returns, expected, atol=1e-12)


# ------------------------------------------------------------------
# Buy-and-hold
# ------------------------------------------------------------------


class TestBuyAndHold:
    def test_weights_drift(self) -> None:
        """After positive returns on asset A, its weight should increase."""
        r_a = jnp.array([[0.10, 0.10]])
        r_b = jnp.array([[0.00, 0.00]])
        cfg = PortfolioConfig(weights={"a": 0.5, "b": 0.5}, rebalance="buy_and_hold")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r_a, "b": r_b}))

        assert out.weights_history is not None
        # After step 0: asset A grew by 10%, B stayed same
        # A value = 0.5 * 1.1 = 0.55, B value = 0.5 * 1.0 = 0.50
        # weight_A = 0.55 / 1.05
        expected_w_a_step0 = 0.55 / 1.05
        assert jnp.isclose(out.weights_history[0, 0, 0], expected_w_a_step0, atol=1e-10)

    def test_returns_match_manual(self) -> None:
        """Buy-and-hold returns match step-by-step manual calculation."""
        r_a = jnp.array([[0.10, 0.05]])
        r_b = jnp.array([[-0.05, 0.02]])
        w_a, w_b = 0.6, 0.4

        # Step 0: weights = (0.6, 0.4), return = 0.6*0.10 + 0.4*(-0.05) = 0.04
        step0 = w_a * 0.10 + w_b * (-0.05)
        # After step 0: A_val = 0.6*1.10 = 0.66, B_val = 0.4*0.95 = 0.38
        # total = 1.04, new_w = (0.66/1.04, 0.38/1.04)
        new_w_a = 0.66 / 1.04
        new_w_b = 0.38 / 1.04
        step1 = new_w_a * 0.05 + new_w_b * 0.02

        cfg = PortfolioConfig(weights={"a": w_a, "b": w_b}, rebalance="buy_and_hold")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r_a, "b": r_b}))
        assert jnp.isclose(out.returns[0, 0], step0, atol=1e-10)
        assert jnp.isclose(out.returns[0, 1], step1, atol=1e-10)


# ------------------------------------------------------------------
# Periodic rebalance
# ------------------------------------------------------------------


class TestPeriodicRebalance:
    def test_rebalance_resets_weights(self) -> None:
        """Weights reset to target at rebalance frequency."""
        n_steps = 6
        freq = 3
        r_a = jnp.array([[0.10] * n_steps])
        r_b = jnp.array([[0.00] * n_steps])

        cfg = PortfolioConfig(
            weights={"a": 0.5, "b": 0.5},
            rebalance="periodic",
            rebalance_frequency=freq,
        )
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r_a, "b": r_b}))

        assert out.weights_history is not None
        # At step 0 (rebalance) and step 3 (rebalance): weights = target
        assert jnp.isclose(out.weights_history[0, 0, 0], 0.5, atol=1e-10)
        assert jnp.isclose(out.weights_history[0, 3, 0], 0.5, atol=1e-10)

        # Between rebalances, weights should drift
        # Steps 1, 2 should have drifted (asset A growing)
        assert out.weights_history[0, 1, 0] > 0.5
        assert out.weights_history[0, 2, 0] > 0.5

    def test_frequency_one_equals_constant_mix(self) -> None:
        """Periodic rebalance with frequency=1 should match constant-mix."""
        r_a = jnp.array([[0.10, 0.05, -0.02]])
        r_b = jnp.array([[0.02, -0.01, 0.04]])
        weights = {"a": 0.6, "b": 0.4}

        cfg_periodic = PortfolioConfig(
            weights=weights, rebalance="periodic", rebalance_frequency=1
        )
        cfg_constant = PortfolioConfig(weights=weights, rebalance="constant_mix")

        result = _make_simulation_result({"a": r_a, "b": r_b})
        out_p = Portfolio(cfg_periodic).aggregate(result)
        out_c = Portfolio(cfg_constant).aggregate(result)

        assert jnp.allclose(out_p.returns, out_c.returns, atol=1e-10)


# ------------------------------------------------------------------
# Portfolio value
# ------------------------------------------------------------------


class TestPortfolioValue:
    def test_initial_value(self) -> None:
        """First element of values should equal initial_value."""
        r = jnp.array([[0.05, 0.03]])
        cfg = PortfolioConfig(
            weights={"a": 1.0}, rebalance="constant_mix", initial_value=100.0
        )
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r}))
        assert jnp.isclose(out.values[0, 0], 100.0, atol=1e-12)

    def test_value_always_positive_long_only(self) -> None:
        """For long-only portfolio with > -100% returns, values stay positive."""
        r_a = jnp.array([[0.10, -0.20, 0.05, -0.30]])
        cfg = PortfolioConfig(weights={"a": 1.0}, rebalance="constant_mix")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r_a}))
        assert jnp.all(out.values > 0)

    def test_value_matches_manual(self) -> None:
        """Value path matches manual compounding."""
        r = jnp.array([[0.10, -0.05, 0.02]])
        cfg = PortfolioConfig(
            weights={"a": 1.0}, rebalance="constant_mix", initial_value=1.0
        )
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r}))
        # v0=1.0, v1=1.10, v2=1.10*0.95=1.045, v3=1.045*1.02=1.0659
        expected = jnp.array([[1.0, 1.10, 1.045, 1.045 * 1.02]])
        assert jnp.allclose(out.values, expected, atol=1e-10)

    def test_values_shape(self) -> None:
        """Values shape is (n_trials, n_steps + 1)."""
        n_trials, n_steps = 4, 10
        r = jnp.zeros((n_trials, n_steps))
        cfg = PortfolioConfig(weights={"a": 1.0}, rebalance="constant_mix")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r}))
        assert out.values.shape == (n_trials, n_steps + 1)


# ------------------------------------------------------------------
# Currency adjustment
# ------------------------------------------------------------------


class TestCurrencyAdjustment:
    def test_fx_adjustment_adds_fx_return(self) -> None:
        """Unhedged foreign return = asset return + FX return."""
        r_asset = jnp.array([[0.10, 0.05]])
        r_fx = jnp.array([[0.02, -0.01]])

        outputs = {
            "foreign_eq": {"LogReturn": r_asset},
            "fx_foreign_eq": {"LogReturn": r_fx},
        }
        time_grid = jnp.linspace(0.0, 2.0, 3)
        sim_result = SimulationResult(outputs=outputs, time_grid=time_grid)

        cfg = PortfolioConfig(
            weights={"foreign_eq": 1.0},
            rebalance="constant_mix",
            currency_base="domestic",
        )
        out = Portfolio(cfg).aggregate(sim_result)
        expected = r_asset + r_fx
        assert jnp.allclose(out.returns, expected, atol=1e-12)

    def test_no_fx_model_uses_raw_returns(self) -> None:
        """Without an FX model, asset returns are used as-is (domestic)."""
        r_asset = jnp.array([[0.10, 0.05]])
        sim_result = _make_simulation_result({"eq": r_asset})

        cfg = PortfolioConfig(
            weights={"eq": 1.0},
            rebalance="constant_mix",
            currency_base="domestic",
        )
        out = Portfolio(cfg).aggregate(sim_result)
        assert jnp.allclose(out.returns, r_asset, atol=1e-12)

    def test_hedged_vs_unhedged_differ(self) -> None:
        """Hedged (no currency_base) vs unhedged returns should differ."""
        r_asset = jnp.array([[0.10, 0.05]])
        r_fx = jnp.array([[0.02, -0.01]])
        outputs = {
            "eq": {"LogReturn": r_asset},
            "fx_eq": {"LogReturn": r_fx},
        }
        time_grid = jnp.linspace(0.0, 2.0, 3)
        sim_result = SimulationResult(outputs=outputs, time_grid=time_grid)

        cfg_hedged = PortfolioConfig(weights={"eq": 1.0}, rebalance="constant_mix")
        cfg_unhedged = PortfolioConfig(
            weights={"eq": 1.0}, rebalance="constant_mix", currency_base="domestic"
        )

        out_hedged = Portfolio(cfg_hedged).aggregate(sim_result)
        out_unhedged = Portfolio(cfg_unhedged).aggregate(sim_result)

        assert not jnp.allclose(out_hedged.returns, out_unhedged.returns)


# ------------------------------------------------------------------
# Analytics
# ------------------------------------------------------------------


class TestAnalytics:
    def test_cumulative_return(self) -> None:
        """Cumulative return: (1+r1)*(1+r2)*... - 1."""
        r = jnp.array([[0.10, 0.05, -0.02]])
        cum = PortfolioAnalytics.cumulative_return(r)
        expected = jnp.array(
            [[1.10 - 1.0, 1.10 * 1.05 - 1.0, 1.10 * 1.05 * 0.98 - 1.0]]
        )
        assert jnp.allclose(cum, expected, atol=1e-10)

    def test_portfolio_value(self) -> None:
        """Value path = initial * cumprod(1+r)."""
        r = jnp.array([[0.10, -0.05]])
        vals = PortfolioAnalytics.portfolio_value(100.0, r)
        expected = jnp.array([[100.0, 110.0, 110.0 * 0.95]])
        assert jnp.allclose(vals, expected, atol=1e-10)

    def test_drawdown(self) -> None:
        """Drawdown at peak is 0, after decline is positive."""
        vals = jnp.array([[100.0, 110.0, 90.0, 95.0]])
        dd = PortfolioAnalytics.drawdown(vals)
        # running max: 100, 110, 110, 110
        # dd: 0, 0, 20/110, 15/110
        expected = jnp.array([[0.0, 0.0, 20.0 / 110.0, 15.0 / 110.0]])
        assert jnp.allclose(dd, expected, atol=1e-10)

    def test_max_drawdown(self) -> None:
        """Max drawdown picks the worst drawdown per trial."""
        vals = jnp.array([[100.0, 110.0, 90.0, 95.0]])
        mdd = PortfolioAnalytics.max_drawdown(vals)
        expected = 20.0 / 110.0
        assert jnp.isclose(mdd[0], expected, atol=1e-10)

    def test_sharpe_ratio(self) -> None:
        """Sharpe = mean(excess) / std(excess)."""
        r = jnp.array([[0.10, 0.05, 0.08, 0.12]])
        rf = 0.02
        excess = r - rf
        expected = jnp.mean(excess, axis=1) / jnp.std(excess, axis=1)
        sharpe = PortfolioAnalytics.sharpe_ratio(r, risk_free=rf)
        assert jnp.allclose(sharpe, expected, atol=1e-10)

    def test_volatility(self) -> None:
        """Volatility = std of returns per trial."""
        r = jnp.array([[0.10, 0.05, 0.08, 0.12]])
        vol = PortfolioAnalytics.volatility(r)
        expected = jnp.std(r, axis=1)
        assert jnp.allclose(vol, expected, atol=1e-12)

    def test_drawdown_zero_values(self) -> None:
        """Drawdown handles zero values without division errors."""
        vals = jnp.array([[0.0, 0.0, 0.0]])
        dd = PortfolioAnalytics.drawdown(vals)
        assert jnp.all(jnp.isfinite(dd))


# ------------------------------------------------------------------
# Integration with SimulationResult
# ------------------------------------------------------------------


class TestIntegration:
    def test_with_simulation_result(self) -> None:
        """Full pipeline: SimulationResult → Portfolio → PortfolioResult."""
        n_trials, n_steps = 10, 20
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        r_a = 0.01 * jax.random.normal(k1, (n_trials, n_steps))
        r_b = 0.02 * jax.random.normal(k2, (n_trials, n_steps))

        sim = _make_simulation_result({"equity": r_a, "bond": r_b})

        cfg = PortfolioConfig(
            weights={"equity": 0.6, "bond": 0.4},
            rebalance="constant_mix",
            initial_value=100.0,
        )
        out = Portfolio(cfg).aggregate(sim)

        assert isinstance(out, PortfolioResult)
        assert out.returns.shape == (n_trials, n_steps)
        assert out.values.shape == (n_trials, n_steps + 1)
        assert out.asset_names == ["equity", "bond"]
        assert jnp.all(jnp.isfinite(out.returns))
        assert jnp.all(jnp.isfinite(out.values))

    def test_missing_model_raises(self) -> None:
        """KeyError if a weighted model is missing from the result."""
        sim = _make_simulation_result({"a": jnp.zeros((2, 3))})
        cfg = PortfolioConfig(weights={"a": 0.5, "missing": 0.5})
        with pytest.raises(KeyError, match="missing"):
            Portfolio(cfg).aggregate(sim)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_single_step(self) -> None:
        """Works with a single time step."""
        r = jnp.array([[0.05]])
        cfg = PortfolioConfig(weights={"a": 1.0}, rebalance="constant_mix")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r}))
        assert out.returns.shape == (1, 1)
        assert out.values.shape == (1, 2)

    def test_negative_returns(self) -> None:
        """Handles negative returns correctly."""
        r = jnp.array([[-0.10, -0.20, -0.05]])
        cfg = PortfolioConfig(weights={"a": 1.0}, rebalance="constant_mix")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r}))
        assert jnp.all(out.values > 0)
        assert jnp.all(out.returns < 0)

    def test_buy_and_hold_single_asset(self) -> None:
        """Buy-and-hold with single asset: weights always 1.0."""
        r = jnp.array([[0.10, 0.05, -0.02]])
        cfg = PortfolioConfig(weights={"a": 1.0}, rebalance="buy_and_hold")
        out = Portfolio(cfg).aggregate(_make_simulation_result({"a": r}))
        assert out.weights_history is not None
        assert jnp.allclose(out.weights_history[0, :, 0], 1.0, atol=1e-12)

    def test_result_config_preserved(self) -> None:
        """PortfolioResult retains the config used."""
        cfg = PortfolioConfig(weights={"a": 1.0}, rebalance="constant_mix")
        out = Portfolio(cfg).aggregate(
            _make_simulation_result({"a": jnp.zeros((1, 3))})
        )
        assert out.config is cfg
